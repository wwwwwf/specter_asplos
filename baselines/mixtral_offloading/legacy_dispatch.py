"""Verbatim author DeepSeek dispatch from specoffmoe commit 82d75e3.

See legacy_dispatch.provenance.json; the class body is unmodified.
Original Mixtral-Offloading-derived code: MIT, see LICENSE.
"""
import time

import torch
from torch import nn


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
        t_start = time.time()
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # router_logits: (batch * sequence_length, n_experts)
        selected_experts, routing_weights, aux_loss, router_logits = self.gate(hidden_states)
        #print("routing_weights", routing_weights, selected_experts)
        identity = hidden_states
        hidden_states = hidden_states.view(-1, hidden_dim)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # process uncached        
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()     
        # only comp hitted
        expert_indices = {}
        for idx in expert_hitted:
            expert_idx = idx.item()
            mask = expert_mask[expert_idx].squeeze(0)
            idx_e, top_x = torch.where(mask)
            expert_indices[expert_idx] = (idx_e, top_x)
        # print("expert_indices", expert_indices)
        # print("moe input", hidden_states)
        # print("gate.weight", self.gate.weight)
        # print("routing_weights", routing_weights)
        
        # show buffer data 
        # print("self.experts.expert_module.main_infos")
        # for i, info in enumerate(self.experts.main_infos):
        #     if info is not None:
        #         print(info.uid, info.eviction_group, info.offloaded, info.index)
        #     else:
        #         print(f"Slot {i}: Empty")
        # # for info in self.experts.main_infos:
        # #     print(info.uid, info.eviction_group, info.offloaded, info.index)
        # print("self.experts.expert_module.main_modules")
        # for i, expert in enumerate(self.experts.main_modules):
        #     if expert is not None:
        #         print(torch.tensor(expert.storage, dtype=torch.bfloat16, device='cuda'))
        #     else:
        #         print(f"Slot {i}: Empty")
        
        # Loop over all available experts in the model and perform the computation on each expert
        if len(expert_hitted) > 0:
            for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
                    *((self.layer_id, expert_idx.item()) for expert_idx in expert_hitted), unordered=True):
                #print("not cacheuid", _layer_index, expert_idx)
                #idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                idx, top_x = expert_indices[expert_idx]
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                torch.cuda.nvtx.range_push(f"[target expert Forward] {_layer_index} {expert_idx}")
                #print("(_layer_index, expert_idx), expert_layer", _layer_index, expert_idx, expert_layer)
                
                #torch.cuda.synchronize("cuda:0")
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(f"[index add] {_layer_index} {expert_idx}")
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                torch.cuda.nvtx.range_pop()
                # if _layer_index == 1:
                #     print("=== 专家缓存状态 ===")
                #     self.experts._print_all_expert_states()
        #print("expert output", final_hidden_states)
        shared_expert_output = self.shared_experts(identity)
        final_hidden_states = final_hidden_states + shared_expert_output
        #print("self.shared_experts(identity)", final_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        #print("final_hidden_states, router_logits", final_hidden_states, router_logits)        
        #print(f"layer forward time:{time.time()-t_start}")
        return final_hidden_states, router_logits
