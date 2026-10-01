"""Portable B=1 author-adapted SpecMoEOff backend for the baseline runner."""
from dataclasses import asdict, replace
import torch

from .cache import ExpertCache
from .controller import LegacyPrefetchController
from .decoder import greedy_decode


class AdaptedSpecMoEOffBackend:
    def __init__(self, case, device, tokenizer, draft, target, manager, loading_report=None):
        self.case, self.device = case, torch.device(device)
        self.tokenizer, self.draft, self.target, self.manager = tokenizer, draft, target, manager
        self.metadata = {
            'method': 'specmoeoff-adapted',
            'implementation': 'author_transformers_b1_adaptation_v1',
            'official_specmoeoff_implementation': False,
            'model_case': asdict(case),
            'batch_size': 1,
            'decoding': 'greedy; separate draft and target DynamicCache',
            'gamma': case.gamma,
            'prefetch_fractions': [0.2, 0.6],
            'prefetch_token_indices': sorted({int(case.gamma * value) for value in (0.2, 0.6)}),
            'draft': 'INT4 same-family MoE',
            'target_attention': 'GPU attention',
            'scheduler': 'retained author route-frequency Top-M; independent implementation',
            'expert_cache': 'immutable host copies; per-layer LRU; event-guarded resident slots',
            'shared_non_expert_parameters': False,
            'shared_draft_target_kv': False,
            'cache_reset': 'initial residents and both LRU orders restored outside timing',
            'corrections': [
                'Exact greedy acceptance, rejection correction, and requested output length',
                'Independent DynamicCache rollback without padding an all-accepted draft',
                'Protected resident hits and bounded miss loading prevent premature slot overwrite',
                'Transfer-ready and consumer-done events guard slot reuse',
                'Predictive stream is joined before target verification',
                'Pinned-host allocation failures are errors, not silent pageable fallbacks',
            ],
            'limits': ['B=1', 'greedy decoding'],
            'loading_report': loading_report or {},
        }
        manager.snapshot_initial()
        self._refresh_memory_metadata()

    def _refresh_memory_metadata(self):
        self.metadata.update({
            'model_case': asdict(self.case),
            'expert_slot_bytes': int(self.manager.module_size),
            'cache_group_count': len(self.manager.group_infos),
            'resident_per_layer': self.case.num_experts - self.case.offload_per_layer,
            'resident_slot_count': len(self.manager.main_modules),
            'transfer_buffer_count': len(self.manager.device_expert_buffers),
            'memory_matching': 'runner selects fixed residency or measured-memory calibration; actual GPU memory is recorded',
        })

    def resize_residency(self, resident_per_layer):
        self.manager.resize_residency(resident_per_layer)
        self.case = replace(self.case, offload_per_layer=self.case.num_experts - int(resident_per_layer))
        self._refresh_memory_metadata()
        return dict(self.metadata)

    def reset(self):
        self.manager.restore()

    def generate(self, input_ids, num_tokens, *, seed=42, clock=None, target_only=False):
        # Seed/reset/timing belong to the shared runner and stay outside the
        # measured interval. Greedy generation does not draw random samples.
        del seed
        if input_ids.device != self.device:
            raise ValueError('Input IDs must be on the backend device')
        controller = None
        if not target_only:
            layers = self.draft.model.model.layers
            routes = [layer.block_sparse_moe if self.case.name == 'phimoe' else layer.mlp
                      for index, layer in enumerate(layers)
                      if not (self.case.skip_first_layer and index == 0)]
            controller = LegacyPrefetchController(self.case, self.manager, routes)
        with torch.inference_mode():
            result = greedy_decode(self.draft, self.target, input_ids, num_tokens,
                                   gamma=self.case.gamma, controller=controller,
                                   clock=clock, target_only=target_only)
        result['stats']['io'] = dict(self.manager.stats)
        result['stats']['prefetch'] = dict(controller.stats) if controller is not None else {}
        return result

    def close(self):
        torch.cuda.synchronize(self.device)
        self.draft = self.target = self.manager = None


def load_backend(case, device):
    if case.name not in {'dsv2lite', 'qwen2moe', 'phimoe'}:
        raise ValueError('Unsupported model family')
    if case.gamma < 1 or not 0 <= case.offload_per_layer < case.num_experts:
        raise ValueError('Invalid gamma or expert residency')
    from speculative_inference_controller.model_init import HybridPrecisionModelInitializer
    from .loading import load_models
    case = HybridPrecisionModelInitializer.resolve_model_paths(case)
    tokenizer, draft, target = load_models(case, torch.device(device))
    # Only repair expert floating-point buffers for the supported checkpoint
    # format. Do not call load()/share_non_expert_parameters(), and do not
    # replace draft backbone values with target values.
    draft.model.to(dtype=torch.float16)
    precision = HybridPrecisionModelInitializer.prepare_fused_expert_precision(draft.model)
    manager = (target.model.layers[1].block_sparse_moe.experts if case.name == 'phimoe'
               else target.model.layers[1].mlp.experts)
    if not isinstance(manager, ExpertCache):
        raise TypeError('Baseline builder did not install the independent cache')
    return AdaptedSpecMoEOffBackend(case, device, tokenizer, draft, target, manager,
                                   {'fused_expert_precision': precision})
