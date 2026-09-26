import copy
import functools
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.activations import ACT2FN
from typing import Dict, Any

import torch
from torch import nn
from torch.nn import functional as F

#from src.packing import pack_4bit_u8_common, pack_2bit_u8_common, unpack_4bit_u8_common, unpack_2bit_u8_common
import time
from transformers.models.phimoe.configuration_phimoe import PhimoeConfig
#from src.configuration_phimoe import PhimoeConfig


def _specter_async_loading(expert_cache) -> bool:
    return bool(
        getattr(
            expert_cache,
            "specter_async_loading_enabled",
            False,
        )
    )


def _load_target_experts(expert_cache, uids):
    if _specter_async_loading(expert_cache):
        return expert_cache.load_experts_async(*uids, unordered=True)
    return expert_cache.load_experts(*uids, unordered=True)


def _overlap_shared_experts(expert_cache) -> bool:
    return bool(
        getattr(
            expert_cache,
            "specter_async_overlap_shared_experts",
            True,
        )
    )

#copy from dsv2 modeling
class Deepseekv2BlockSparseTop2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return out





def sparsemixer(scores, top_k, jitter_eps):
    assert top_k == 2
    
    ################ first expert ################
    
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask 
    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    selected_experts = max_ind
        
    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
    
    multiplier = multiplier_o

    # masked out first expert 
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float('-inf'),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask 
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
    selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
    
    multiplier_top2 = multiplier_top2_o
    
    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)
    
    return (
        multiplier, 
        selected_experts,
    )

class SparseMoeWrapperPhimoe(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id
        self.router_jitter_noise = config.router_jitter_noise
        self.gate = gate
        self.experts = expert_cache
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # print ( 'moe', self.iter, torch.norm(hidden_states).item())
        router_logits = self.gate(hidden_states)
        #print(f"layer {self.layer_id} router logits: {router_logits}")
        routing_weights, selected_experts = sparsemixer(
            router_logits, 
            top_k=2, 
            jitter_eps=self.router_jitter_noise
        )
        #print(f"layer {self.layer_id} mixer logits: {routing_weights}")
        # match
        # print(f"target layer {self.layer_id} selected_experts: {selected_experts}")
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        # only comp hitted
        expert_indices = {}
        for idx in expert_hitted:
            expert_idx = idx.item()
            mask = expert_mask[expert_idx].squeeze(0)
            idx_e, top_x = torch.where(mask)
            expert_indices[expert_idx] = (idx_e, top_x)
    
        expert_outputs = []
        if len(expert_hitted) > 0:
            uids = tuple(
                (self.layer_id, expert_idx.item())
                for expert_idx in expert_hitted
            )
            for (
                (_layer_index, expert_idx),
                expert_layer,
            ) in _load_target_experts(self.experts, uids):
                idx, top_x = expert_indices[expert_idx]

                assert top_x.shape[0] > 0
                t0=time.time()
                #print(f"layer {self.layer_id} current input {hidden_states[None, top_x_list].reshape(-1, hidden_dim)}")
                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                #print("per com of target expert:", time.time()-t0)
                #print(f"layer {self.layer_id} expert {expert_idx} expert output {current_hidden_states}")
                expert_outputs.append(
                    (
                        expert_idx,
                        top_x,
                        current_hidden_states.to(hidden_states.dtype),
                    )
                )

        # Expert loading remains hot-first; only the reduction order is fixed.
        for _, top_x, current_hidden_states in sorted(
            expert_outputs,
            key=lambda item: item[0],
        ):
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # print ( 'moe', self.iter, torch.norm(final_hidden_states).item())
        #print(f"layer {self.layer_id} total output {final_hidden_states}")
        return final_hidden_states, router_logits

class SparseMoeWrapperShared(nn.Module):
    def __init__(self, config, layer_id, gate, shared_expert, shared_expert_gate, expert_cache):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts # todo
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = gate
        self.experts = expert_cache
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        attn_output = hidden_states
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        #print("routing_weights, selected_experts", routing_weights, selected_experts)
        #print(self.layer_id, selected_experts)
        #print("large model selected_experts", selected_experts)
        #config.norm_topk_prob = false
        #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        #active_experts = selected_experts.flatten().unique().tolist()
        
        # reorder
        # expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        # expert_hitted_set = set(expert_hitted.flatten().tolist())
        # group_infos = self.experts.group_infos[self.layer_id]
        # cached_items = list(group_infos.main_infos.items())
        # for cached_uid, main_info in cached_items:
        #     if main_info.eviction_group != self.layer_id:
        #         continue
        #     layer_idx, cached_expert = cached_uid
        #     if cached_expert in expert_hitted_set:
        #         idx, top_x = torch.where(expert_mask[cached_expert].squeeze(0))
        #         current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        #         (cacheuid, expert_layer) = next(self.experts.load_experts((self.layer_id, cached_expert)))
        #         #print(f"hit self.layer_id, expert_idx: {self.layer_id} {cached_expert}")
        #         current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        #         # However `index_add_` only support torch tensors for indexing so we'll use
        #         # the `top_x` tensor here.
        #         final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        #         expert_mask[cached_expert] = 0 
        
        # process uncached        
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()     
        expert_indices = {}
        for idx in expert_hitted:
            expert_idx = idx.item()
            mask = expert_mask[expert_idx].squeeze(0)
            idx_e, top_x = torch.where(mask)
            expert_indices[expert_idx] = (idx_e, top_x)   
        async_loading = _specter_async_loading(self.experts)
        expert_iter = None
        if len(expert_hitted) > 0:
            uids = tuple(
                (self.layer_id, expert_idx.item())
                for expert_idx in expert_hitted
            )
            expert_iter = _load_target_experts(self.experts, uids)

        # Async H2D is now in flight; shared experts provide useful overlap.
        shared_expert_output = None
        if async_loading and _overlap_shared_experts(self.experts):
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states))
                * shared_expert_output
            )

        expert_outputs = []
        if expert_iter is not None:
            for (_layer_index, expert_idx), expert_layer in expert_iter:

                idx, top_x = expert_indices[expert_idx]

                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                expert_outputs.append(
                    (
                        expert_idx,
                        top_x,
                        current_hidden_states.to(hidden_states.dtype),
                    )
                )

        # Expert loading remains hot-first; only the reduction order is fixed.
        for _, top_x, current_hidden_states in sorted(
            expert_outputs,
            key=lambda item: item[0],
        ):
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        #print("expert",self.shared_expert.gate_proj.weight.device)
        #print("gate",self.shared_expert_gate.weight.device)
        #print("hidden",hidden_states.device)
        
        #SHARE_EXP
        #mlp_output = final_hidden_states.clone()
        if shared_expert_output is None:
            shared_expert_output = self.shared_expert(hidden_states)
            #sexp_output = shared_expert_output.clone()
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        
        # print("attn_output", attn_output)
        # print("logits",router_logits)
        # print("weights",routing_weights)
        # print("selected experts",selected_experts)
        # print("normal expert_output", final_hidden_states)
        # print("shared_expert_output", shared_expert_output)
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
                
        #print(f"hidden_states {hidden_states} router_logits {router_logits} shared_expert_output {shared_expert_output} mlp_output {mlp_output} sexp_output {sexp_output} final_hidden_states {final_hidden_states}")
        
        return final_hidden_states, router_logits

class SparseMoeWrapperDeepseekv2(nn.Module):
    def __init__(self, config, layer_id, gate, shared_experts, expert_cache):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.n_routed_experts # todo
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = gate
        self.experts = expert_cache
        self.shared_experts = shared_experts
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        selected_experts, routing_weights, _, router_logits = self.gate(
            hidden_states
        )
        identity = hidden_states
        hidden_states = hidden_states.view(-1, hidden_dim)
        token_count = hidden_states.shape[0]
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Build expert ranges on the GPU and transfer only one count vector.
        slot_major = (
            selected_experts.transpose(0, 1)
            .contiguous()
            .reshape(-1)
        )
        positions = torch.arange(
            slot_major.numel(),
            device=slot_major.device,
            dtype=torch.int64,
        )
        keys = (
            slot_major.to(torch.int64) * slot_major.numel()
            + positions
        )
        order = torch.argsort(keys)
        counts = torch.zeros(
            self.num_experts,
            dtype=torch.int64,
            device=slot_major.device,
        )
        counts.scatter_add_(
            0,
            slot_major.to(torch.int64),
            torch.ones_like(slot_major, dtype=torch.int64),
        )
        counts_cpu = counts.cpu().tolist()

        ranges = {}
        uids = []
        offset = 0
        for expert_idx, count in enumerate(counts_cpu):
            if count == 0:
                continue
            end = offset + count
            ranges[expert_idx] = (offset, end)
            uids.append((self.layer_id, expert_idx))
            offset = end

        async_loading = _specter_async_loading(self.experts)
        expert_iter = None
        if uids:
            expert_iter = _load_target_experts(
                self.experts,
                tuple(uids),
            )

        # Submit all immediately safe H2D copies before shared-expert GEMMs.
        shared_expert_output = None
        if async_loading and _overlap_shared_experts(self.experts):
            shared_expert_output = self.shared_experts(identity)

        expert_outputs = []
        if expert_iter is not None:
            for (
                (_layer_index, expert_idx),
                expert_layer,
            ) in expert_iter:
                start, end = ranges[expert_idx]
                expert_positions = order[start:end]
                idx = torch.div(
                    expert_positions,
                    token_count,
                    rounding_mode="floor",
                )
                top_x = torch.remainder(
                    expert_positions,
                    token_count,
                )
                current_state = hidden_states[
                    None, top_x
                ].reshape(-1, hidden_dim)
                torch.cuda.nvtx.range_push(
                    f"[target expert Forward] "
                    f"{_layer_index} {expert_idx}"
                )
                current_hidden_states = (
                    expert_layer(current_state)
                    * routing_weights[top_x, idx, None]
                )
                torch.cuda.nvtx.range_pop()
                expert_outputs.append(
                    (
                        expert_idx,
                        _layer_index,
                        top_x,
                        current_hidden_states.to(
                            hidden_states.dtype
                        ),
                    )
                )

        # Cache hits can be yielded before misses. Expert-id order keeps
        # accumulation deterministic while preserving transfer overlap.
        for (
            expert_idx,
            _layer_index,
            top_x,
            current_hidden_states,
        ) in sorted(expert_outputs, key=lambda item: item[0]):
            torch.cuda.nvtx.range_push(
                f"[index add] {_layer_index} {expert_idx}"
            )
            final_hidden_states.index_add_(
                0,
                top_x,
                current_hidden_states,
            )
            torch.cuda.nvtx.range_pop()

        if shared_expert_output is None:
            shared_expert_output = self.shared_experts(identity)
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits



class PhiMoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: PhimoeConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class QwenmoeBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.ffn_dim = config.moe_intermediate_size
        self.hidden_dim = config.hidden_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        current_hidden_states = self.down_proj(current_hidden_states)
        return current_hidden_states
