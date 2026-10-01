# SpecMoEOff adaptation

Adapted from the author's retained `specoffmoe` B=1 Transformers implementation
(`main._run_baseline2_spec_case`, repository HEAD
`b237896fd8b2d69dcf1294d2239bb6603e71d93c`). The cited system is
[SpecMoEOff, arXiv:2508.21706](https://arxiv.org/abs/2508.21706).

This adapter uses a same-family INT4 draft, an FP16 target, K=3,
route-frequency Top-M prefetching at draft fractions 0.2 and 0.6, and independent
weight/KV storage, decoder, controller, and expert cache. Its scope is the
retained single-request Transformers path; the original system additionally
includes SGLang/EAGLE, CPU chunked attention, and a two-microbatch target pipeline.

Greedy rejection, rollback, and output-length handling follow the common
benchmark contract. Readiness/consumer events protect cache-slot reuse.
Model-format helpers and target routing arithmetic are shared with the artifact.

The cache is derived from [Mixtral-Offloading](https://github.com/dvmazur/mixtral-offloading)
(commit `ce545188b804238f0b23a59fc45e6a8f8b390c40`); its original MIT notice is
in `LICENSE.mixtral-offloading`. Reused model/kernel files retain their notices.
