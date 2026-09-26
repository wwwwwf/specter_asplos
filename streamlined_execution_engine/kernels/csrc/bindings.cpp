#include <torch/extension.h>
#include "core/scalar_type.hpp"
#include "core/shared_memory_cache.h"
#include <pybind11/stl.h>
torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> const& c_or_none,
    torch::Tensor& b_q_weight, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights, bool is_ep,
    vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float);
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                         int64_t block_size, torch::Tensor sorted_token_ids,
                         torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad);
void moe_sum(torch::Tensor& input, torch::Tensor& output);
torch::Tensor gptq_marlin_repack(torch::Tensor& weight, torch::Tensor& perm,
                               int64_t k, int64_t n, int64_t bits, int64_t pack_bits);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_gemm", &moe_wna16_marlin_gemm);
  m.def("align_tokens", &moe_align_block_size);
  m.def("sum_experts", &moe_sum);
  m.def("repack", &gptq_marlin_repack);
  m.def("shared_memory_cache_stats", &specter::shared_memory_cache_stats);
}
