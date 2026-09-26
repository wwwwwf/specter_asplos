# Design

Specter runs speculative decoding for DeepSeek-V2-Lite, Qwen1.5-MoE, and
Phi-3.5-MoE on a single GPU with batch size one. A quantized draft proposes
tokens; an FP16 target verifies them while expert transfers overlap computation.

## Speculative Inference Controller

`speculative_inference_controller/` coordinates proposal, verification, and
token commitment.

- `model_init.py` connects model construction in `initialization/` to the
  runtime. Routed draft experts retain INT4 weights. Matching non-expert
  parameters, including routers and shared experts, reuse target tensors.
- `depth_profiler.py` selects draft depth by comparing mean decode TPOT across
  candidate depths. Warmup is excluded, and each measurement begins with the
  same resident expert set and cache order.
- `state.py` owns the decoding loop and accepted-token boundary. The target
  stores committed KV; the draft reads that prefix and appends temporary KV.
  Verification crops target KV to the accepted boundary and rebases the draft
  onto it. The correction or bonus token becomes the next uncached input.
- `shared_prefix_cache.py` reuses draft scratch buffers across decoding windows.
  Attention still forms contiguous prefix-plus-scratch tensors when needed.

## Predictive I/O Orchestrator

`predictive_io_orchestrator/` predicts expert demand and manages target expert
residency.

- `pattern_analyzer.py` accumulates draft routing counts on the GPU and ranks
  experts at chunk boundaries to form a compact prefetch plan.
- `prefetch_planner.py` starts with a short observation chunk, then chooses
  subsequent chunks using transfer bandwidth and draft execution time. Its
  cleanup phase stops lookahead and establishes verification dependencies.
- `expert_cache.py` tracks resident experts, replacement order, and slot
  ownership. Verification processes resident experts first, then loads missing
  experts into available slots while protecting queued or in-use weights.
- `routing_hooks.py` connects draft routing observations to the planner.

Expert residency is a per-layer slot budget. Total GPU memory also includes
model state, KV, and transfer buffers; a slot count is not a process memory cap.

## Streamlined Execution Engine

`streamlined_execution_engine/` implements expert computation and coordinates
transfer/compute dependencies.

- `draft_fusion.py` prepares packed draft weights and invokes fused routed
  expert operations. Local kernels in `kernels/` align token groups, perform
  Marlin W4A16 GEMMs, and reduce expert outputs.
- `expert_reorder.py` gathers tokens by expert, orders resident and missing
  expert execution, and scatters results back into token order.
- `overlap_optimizer.py` coordinates transfer streams and readiness/last-use
  events. Native launch metadata is cached by device and kernel to avoid
  repeated attribute queries during warmed execution.

W4A16 applies to routed draft experts: the kernels dequantize INT4 weight tiles
while computing with FP16 activations. The target uses FP16 weights. Required
FP32 normalization and reduction arithmetic remain in their respective models.

## Supporting code

`models/` contains model forward implementations and required helpers;
`initialization/` loads checkpoints and prepares execution state. `data/`
prepares benchmark inputs. `cli/` provides generation and depth profiling,
while `benchmarks/` records performance and output checks. `tests/` covers
runtime behavior and local kernels.

`benchmarks.run_ae` provides a default DeepSeek high-residency experiment and
dispatches optional depth, prefix-length, system, and kernel measurements.
`run_experiments.py` and `experiment_plan.py` define shared input selection,
seeded workload sweeps, and recorded configuration for routing, working sets,
sensitivity, cache policies, partial ablations, latency, memory, route fidelity,
and oracle predictions.

- `routing.py`, `latency.py`, and `memory.py` capture diagnostics in passes
  separate from clean TPOT measurements. Storage accounting includes model,
  expert-pool, and KV aliases; it measures retained capacity rather than
  per-step allocation savings.
- `cache_policies.py` varies prefetch timing. `variants.py` can copy committed
  target KV or serialize demand experts while retaining target verification,
  shared model weights, and fused draft kernels. These are partial mechanism
  ablations.
- `oracle.py` replays recorded target routes with controlled prediction
  perturbations and validates the replay trajectory.
- `kernel_bench.py` validates and times a single local routed projection using
  batched W4A16, sequential W4A16, and dequantized FP16 paths with shared inputs
  and represented weights.

# can ignore
NUMA placement remains caller-controlled. Benchmark entries record actual
placement and validate the configured CPU and preferred-memory nodes only
when `--check-numa` is requested; they do not bind the process.

All asset and output paths come from the global TOML configuration. Checkpoint
weights, prepared overlays, dependency wheels, caches, and experiment outputs
live outside the source directory.
