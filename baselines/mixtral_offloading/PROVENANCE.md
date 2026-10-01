# Mixtral-Offloading cache and author DeepSeek adapter

Based on [Mixtral-Offloading](https://github.com/dvmazur/mixtral-offloading),
commit `ce545188b804238f0b23a59fc45e6a8f8b390c40` (MIT). The original source
files and license are preserved byte-for-byte in `upstream/`; hashes are in
`upstream/PROVENANCE.json`.

Execution calls the original `ExpertCache` directly: layer-local LRU,
resident-first demand loading, and buffered swaps are unchanged.

The author's DeepSeek adapter is pinned to `specoffmoe` commit
`82d75e3ad3d38fe71c86b0dba290d1e0ef9e66f0`. `legacy_dispatch.py` preserves
`SparseMoeWrapperDeepseekv2` from `src/custom_layers.py` verbatim; source
lines and hashes are in `legacy_dispatch.provenance.json`.

The upstream model is HQQ-quantized Mixtral. The local adapter supplies
DeepSeek-V2-Lite routing, shared experts, checkpoint keys, and FP16 storage.
Expert gate/up/down projections remain unfused, with outputs accumulated
in cache yield order.

`backend.py` supplies greedy decoding and the common token-timing interface.
`original_cache.py` restores initial residency/LRU outside measurement.
The fixed DS presets use 32/20 residents per MoE layer for high/low and
4 global buffers. The original 27-layer allocation formula is retained
for 26 MoE layers, including one layer's unused GPU and host slots.
Input preparation, cache reset, and token timing follow the shared
benchmark protocol.
