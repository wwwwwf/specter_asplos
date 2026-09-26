import torch
from transformers import DynamicCache
from speculative_inference_controller.sampling import norm_logits
from predictive_io_orchestrator.pattern_analyzer import IncrementalRoutingPatternAnalyzer

class ModelWrapper():
    def __init__(self, model : torch.nn.Module, prefetch_controller : IncrementalRoutingPatternAnalyzer = None, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.prefetch_controller = prefetch_controller

    @staticmethod
    def _require_dynamic_cache(cache) -> DynamicCache:
        if not isinstance(cache, DynamicCache):
            raise TypeError(
                "Specter requires models to preserve the DynamicCache "
                f"instance, got {type(cache).__name__}"
            )
        return cache

    def _update_prefetch_routes(self):
        if self.prefetch_controller is None:
            return
        update_routes = getattr(
            self.prefetch_controller,
            "update_from_bound_routes",
            None,
        )
        if update_routes is not None:
            update_routes()
        
    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        torch.cuda.nvtx.range_push("forward")
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(
                input_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
            )
            self._update_prefetch_routes()
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = self._require_dynamic_cache(
                outputs.past_key_values
            )
            #print(outputs.past_key_values)
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cache = self._require_dynamic_cache(self._past_key_values)
            cached_len = cache.get_seq_length()

            last_input_id = input_ids[:, cached_len:]

            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            outputs = self._model(
                last_input_id,
                past_key_values=cache,
                use_cache=True,
            )
            self._update_prefetch_routes()
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = self._require_dynamic_cache(
                outputs.past_key_values
            )
        torch.cuda.nvtx.range_pop()
        return last_q



    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        cache = self._require_dynamic_cache(self._past_key_values)
        cache.crop(end_pos)
        self._prob_history = self._prob_history[:, :end_pos, :]
