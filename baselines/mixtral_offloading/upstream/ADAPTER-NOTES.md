# Vendored upstream source

Files listed in `PROVENANCE.json` are exact Git blobs from
[dvmazur/mixtral-offloading](https://github.com/dvmazur/mixtral-offloading),
commit `ce545188b804238f0b23a59fc45e6a8f8b390c40` (MIT). The manifest records
paths, hashes and blob IDs; `.gitattributes` preserves their original bytes.

The DeepSeek entry imports `src.expert_cache.ExpertCache` directly. Its LRU,
resident-first iterator and buffered swaps are unchanged. The author's early
Mixtral-Offloading cache uses the same demand/swap operations.

`../legacy_dispatch.py` contains the author's historical DeepSeek expert
execution class. `../model.py` supplies FP16 storage and current checkpoint/API
adapters; it does not use the upstream HQQ Mixtral model builder or kernels.
Historical 27-layer allocation is retained, with only 26 registered MoE layers.
The high setting has 32 residents per MoE layer and four global transfer buffers.

`../original_cache.py` restores initial registered slots/LRU outside timing,
leaving historical unused slots untouched. See `../PROVENANCE.md` for the
historical dispatch source and the current measurement protocol.
