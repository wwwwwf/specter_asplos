# Specter

Speculative decoding with expert offloading for DeepSeek-V2-Lite, Qwen1.5-MoE,
and Phi-3.5-MoE. The implementation targets a single NVIDIA A100 40 GB with
batch size one. See [Design](docs/design.md) for the modules and mechanisms.

## Setup

Use Linux, Python 3.11.13, a CUDA-capable driver, CUDA Toolkit, a C++ compiler,
and `uv`. `numactl` is optional. The validated environment uses Torch 2.7.0 with CUDA runtime
12.6 and CUDA Toolkit 12.8. Custom GPTQModel, Transformers, FlashAttention, and
vLLM wheels are required; upstream wheels with matching version numbers are
not interchangeable with these builds. Provision those wheels, the target
and draft checkpoints, and the benchmark datasets separately.

From the repository root:

```bash
cp config.example.toml ../specter.local.toml
# Edit the copied configuration to point to your local assets and output directories.
uv run --python 3.11 --no-project environment/install.py --config ../specter.local.toml
uv run --python 3.11 --no-project environment/install.py --config ../specter.local.toml --check
```

The configuration defines model and dataset paths, cache and output directories,
the environment's Python version, virtual environment, and wheelhouse, and
CPU thread count and optional NUMA checks. Relative configuration paths resolve from the
configuration file's parent directory. Keep the local configuration and all
assets outside this repository.

Activate the virtual environment selected by `environment.venv`, then build
the local kernels:

```bash
source /path/to/specter-env/bin/activate
export SPECTER_CONFIG=../specter.local.toml
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
python -B -m streamlined_execution_engine.kernels.build
python -B -m pytest -p no:cacheprovider tests -q
```

Set `CUDA_HOME` to the installed toolkit location. Pass `--config` to each
entry point, or set `SPECTER_CONFIG` once to the local configuration path.

## Default experiment

Run the DeepSeek-V2-Lite high-residency case:

```bash
python -B -m benchmarks.run_ae --config ../specter.local.toml
```

This runs one model/configuration, with all five datasets, five inputs per
dataset, and three repeats (75 records). It uses greedy decoding, K=16,
16 resident experts per layer, and 128 output tokens.

The default result directory is `paths.output/ae/main/`. `records.jsonl`
contains per-input timing and generated tokens; `summary.json` and `summary.csv`
contain TPOT summaries. Configuration, inputs, warmup, and target output checks
are saved separately. An existing experiment directory is not overwritten.
Use `--output ae_run2` for another run.

Two optional supporting experiments reuse the same DeepSeek checkpoints:

```bash
python -B -m benchmarks.run_ae --experiment depth
python -B -m benchmarks.run_ae --experiment prefix
```

These commands use `SPECTER_CONFIG` exported during setup; `--config` can also
be passed explicitly. `depth` sweeps K={2,4,6,8,12,16,20,24,32} over three
GK inputs and three repeats (81 measurements). It writes `ae/depth/`.
`prefix` compares input-prefix caps of 8, 16, and 32 tokens using three GK inputs
and three repeats per length (27 measurements), writing `ae/prefix_8/`,
`ae/prefix_16/`, and `ae/prefix_32/`. Both generate 128 tokens.

Use `--experiment all` with a fresh output directory to run the three experiments
sequentially. Add `--dry-run` to inspect the commands without loading models.

## Output paths and host placement

All results are written outside the source directory. For example, with
`paths.output = "/data/specter-results"`, the default main experiment writes
to `/data/specter-results/ae/main/`. Relative `--output` values resolve under
`paths.output`; absolute external paths are accepted. Relative paths inside
the TOML resolve from the configuration file's directory.

NUMA binding is optional. The default command uses the operating system's
placement. To request a specific placement, choose nodes for your machine and
match them to `runtime.cpu_node` and `runtime.memory_node`:

```bash
numactl --cpunodebind=0 --preferred=0 python -B -m benchmarks.run_ae \
  --config ../specter.local.toml --check-numa --output ae_numa0
```

The configuration does not bind the process; `--check-numa` validates an
explicit binding. CPU affinity and available NUMA policy information are
recorded.

## Additional measurements

The optional experiments use the current model loader, decoder, expert cache,
and local kernels. Run one measurement family with:

```bash
python -B -m benchmarks.run_ae --experiment routing
```

Available families are:

| Experiment | Measurements |
| --- | --- |
| `routing` | Draft/target expert sets, verification-window recall, routing diversity, and cache demand hits across datasets. |
| `working-set` | Expert-set growth and decode TPOT across speculative depths. |
| `sensitivity` | Task domain, input length, and greedy or sampling decoding at temperatures 0.7 and 1.0; includes routing recall. |
| `cache` | Adaptive, no-prefetch, early-prefetch, and late-prefetch policies at the same resident capacity. |
| `ablation` | Cumulative removal of predictive prefetch, shared-prefix KV storage, and demand-transfer overlap. |
| `latency` | CUDA event traces of draft/verification forwards, H2D copies, and observed stream waits. |
| `memory` | Actual model, expert-pool, shared-storage, KV, and scratch allocations with shared or copied prefix KV. |
| `fidelity` | Token-level expert-selection agreement for identical inputs and independent empty caches. |
| `oracle` | Recorded target-route replay with 0%, 25%, 50%, and 75% identity replacement for prefetching. |
| `kernel` | Local batched and sequential W4A16 routed projections against a dequantized FP16 reference. |

Each command writes to `paths.output/ae/<experiment>/`. The existing
`--experiment all` shortcut continues to run `main`, `depth`, and `prefix`.
Use the detailed entry to select another model or change the workload:

```bash
python -B -m benchmarks.run_experiments --experiment sensitivity \
  --model qwen2moe --num-data 3 --repeats 3 --tokens 128 --output qwen_sensitivity
python -B -m benchmarks.run_experiments --experiment cache \
  --residents 4 8 16 --output ds_cache
python -B -m benchmarks.run_experiments --experiment memory \
  --model qwen2moe --prompts-json ../specter-assets/long_prompts.json \
  --prefix-tokens 12000 --num-data 1 --repeats 1 --tokens 128 --output qwen_memory
python -B -m benchmarks.kernel_bench --tokens 1 16 128 \
  --hidden 2048 --features 1408 --experts 64 --top-k 6 --output ds_projection
```

Configuration is read from `SPECTER_CONFIG`, or from an explicit `--config`.
All entries accept `--dry-run`. Diagnostic runs and clean TPOT runs use the same
inputs, seed, and initial cache state. Their generated tokens must match before
the measurements are paired. Length sweeps select the same source prompts long
enough for every requested length; `--allow-short-prefix` retains shorter inputs.
Oracle replay additionally checks target-route and decoding-trajectory agreement.
Raw records, diagnostic traces, input IDs, settings, and summaries are retained.

The ablations isolate the named mechanisms: copied KV still uses target-validated
states, and serial demand still uses the fused draft kernels. Cache capacity is
specified as resident experts per layer. Memory reports count reachable live
storage separately from the allocator peak. Forward event spans include launch
gaps and waits; overlapping intervals are never added as sequential components.
The kernel experiment measures one routed projection, including sequential
gather/scatter and excluding fixed routing preparation and weight packing.

Render the results after a measurement completes:

```bash
python -B -m benchmarks.plot_experiments --input /path/to/results/ae/routing
```

Plots and their source tables are written under the input directory's `figures/`
subdirectory, outside the source tree.

## Other entry points

Generate a response:

```bash
python -B -m cli.run --config ../specter.local.toml --model qwen2moe \
  --prompt 'The capital of France is' --max-new-tokens 128
```

Default decoding is greedy. Default draft depths for DeepSeek, Qwen, and Phi
are 16, 8, and 16. Outputs include generated tokens and text, decode TPOT,
TTFT, and end-to-end time.

Profile draft depth using a JSON array of representative prompt strings:

```bash
python -B -m cli.profile --config ../specter.local.toml --model qwen2moe \
  --prompts-json ../specter-assets/prompts.json --tokens 128 --repeats 3
```

Other model cases remain available through the benchmark entry:

```bash
python -B -m benchmarks.run_tpot \
  --config ../specter.local.toml --protocol-case qwen_high --model qwen2moe \
  --gamma 8 --offload-per-layer 45 --strategy greedy --output qwen_high
```

Case settings are in `benchmarks/protocol.json`. Each case measures
five inputs from each of five datasets across three repeats, producing 75
records. You can choose samples dependent on your time budget. For a custom
input count or repeat count, run `benchmarks.run_tpot` without `--protocol-case`
and set `--num-data` or `--repeats`.
