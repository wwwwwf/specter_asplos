# Specter

Speculative decoding with expert offloading for DeepSeek-V2-Lite, Qwen1.5-MoE,
and Phi-3.5-MoE. The implementation targets a single NVIDIA A100 40 GB with
batch size one. See [Design](docs/design.md) for the modules and mechanisms.

## Setup

For AE, use the source archive from Zenodo downloaded below. Run the installation
and experiment commands from its extracted `$SPECTER_WORK/specter` directory,
which includes the validated installer, resource checks, and matching configuration.

Use Linux x86_64, an NVIDIA A100 40 GB, a CUDA-capable driver, CUDA Toolkit,
a C++ compiler, `curl`, `tar`, and `unzip`. The validated environment uses
Python 3.11.13, Torch 2.7.0 with CUDA runtime 12.6, and CUDA Toolkit 12.8.
`numactl` is optional. Allow about 200 GB of free disk space for downloaded
archives, extracted checkpoints, the environment, and caches.

### Download and extract

The [Zenodo record](https://doi.org/10.5281/zenodo.22937478) supplies:

| File | Contents |
| --- | --- |
| `specter-ae-v1.tar.gz` | Source code, download helper, model manifest, and setup guide. |
| `specter-ae-wheels.tar.gz` | The four validated dependency wheels and their licenses. |
| `specter-ae-datasets.tar.gz` | Prepared GK, WT, HE, GP, and C4 inputs. |
| `specter-ae-v1.manifest.json`, `SHA256SUMS` | File inventory and integrity checks. |

The two DeepSeek checkpoint archives are hosted as numbered parts in the
[GitHub model release](https://github.com/wwwwwf/specter_asplos/releases/tag/ae-models-20260927).
`initialization/model-assets.json` records their URLs, ordered parts, sizes,
and SHA-256 hashes. The download helper uses the Python standard library with
`curl`; system Python 3.8 or newer is sufficient. The release is public and
requires no GitHub login. The isolated inference environment is installed
later with uv.

Choose a new working directory on a filesystem with sufficient free space.
Replace `/path/to/specter-work` below with that directory:

```bash
export SPECTER_WORK=/path/to/specter-work
mkdir -p "$SPECTER_WORK/downloads"
cd "$SPECTER_WORK/downloads"

for file in specter-ae-v1.tar.gz specter-ae-wheels.tar.gz \
  specter-ae-datasets.tar.gz specter-ae-v1.manifest.json SHA256SUMS; do
  curl --fail --location --retry 3 \
    "https://zenodo.org/api/records/22937478/files/${file}/content" \
    --output "$file" || break
done

sha256sum --ignore-missing -c SHA256SUMS
```

This initial check must report four `OK` entries: the three downloaded archives
and the manifest. It skips the two model archives, which are downloaded next.
Continue after all four checks report `OK`. Extract the source package first:

```bash
cd "$SPECTER_WORK"
tar -xzf downloads/specter-ae-v1.tar.gz
cd "$SPECTER_WORK/specter"
```

Download and reconstruct the two checkpoint archives with:

```bash
python3 -B initialization/download_models.py \
  --manifest initialization/model-assets.json \
  --output "$SPECTER_WORK/downloads" \
  --source-root "$SPECTER_WORK/specter"
```

Rerun the same command to resume interrupted parts. Each part is checked before
use, and each reconstructed archive is checked before receiving its final
filename. Downloads remain outside the source tree. Verified parts are retained
under `downloads/.specter-parts/`; parts plus reconstructed model archives use
about 68 GB before extraction. `--only target` or `--only draft` selects one model.

After both model archives have been verified, check all files together:

```bash
cd "$SPECTER_WORK/downloads"
sha256sum -c SHA256SUMS
```

Continue after all six checks report `OK`, then extract the four asset archives:

```bash
cd "$SPECTER_WORK"
for file in specter-ae-wheels.tar.gz specter-ae-datasets.tar.gz \
  specter-ae-dsv2-target.tar.gz specter-ae-dsv2-draft.tar.gz; do
  tar -xzf "downloads/$file" || break
done
unzip -n -P deserted-untie-orchid \
  specter-assets/datasets/gpqa/gpqa_main.csv.zip \
  -d specter-assets/datasets/gpqa/
cp -n specter/config.example.toml specter.local.toml
export SPECTER_CONFIG="$SPECTER_WORK/specter.local.toml"
cd "$SPECTER_WORK/specter"
```

GPQA uses the [upstream password-protected distribution convention](https://github.com/idavidrein/gpqa).
Keep its canary and accompanying notices. Dataset/model licenses and source
attribution are included with the asset archives. The target stores the
prepared source tensors and is loaded as FP16 by Specter; the draft quantizes
routed experts to GPTQ INT4, group size 128. The supplied checkpoints already
have the required layout and quantization. They cover all three selected
experiments below; Qwen/Phi checkpoints for optional experiments are not bundled.

The extracted layout is:

```text
$SPECTER_WORK/
  specter/                 # Source extracted from Zenodo
  specter-assets/
    wheels/                # Four validated wheels
    models/                # DeepSeek target and INT4 draft
    datasets/              # Five prepared datasets
  specter.local.toml       # External configuration copied from the source package
```

The copied configuration points to these actual model and dataset directories;
no example paths need replacing for the supplied assets. It also places `env/`,
`cache/`, and `results/` under `$SPECTER_WORK`. To use existing assets, edit their
paths in this external configuration. Relative paths resolve from the
configuration file's directory, not from the shell's current directory.

### Install the environment

Use the four builds supplied in `specter-ae-wheels.tar.gz`:

| Package | Bundled version |
| --- | --- |
| GPTQModel | `4.0.0.dev0` |
| Transformers | `4.53.3` |
| FlashAttention | `2.8.1` |
| vLLM | `0.9.2` |

These are the validated builds, including the custom GPTQModel adaptations.
Upstream wheels with matching version numbers are not interchangeable.
The installer checks their exact SHA-256 hashes against the bundled
`environment/wheels.sha256`; the default wheelhouse is `specter-assets/wheels/`.

Install the validated uv version in your user directory; no `sudo` is needed:

```bash
set -o pipefail
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | \
  env UV_INSTALL_DIR="$HOME/.local/bin" sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

export UV_CACHE_DIR="$SPECTER_WORK/cache/uv"
export UV_PYTHON_INSTALL_DIR="$SPECTER_WORK/cache/python"
export TMPDIR="$SPECTER_WORK/cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TMPDIR" "$SPECTER_WORK/logs"

uv run --python 3.11.13 --no-project environment/install.py \
  --config "$SPECTER_CONFIG" --check
```

The cache variables above place package downloads, managed Python, and temporary
build files on the chosen work filesystem. Keep them exported before the first
`uv run` to avoid filling the home filesystem or `/tmp` during installation.

The check prints the resolved environment/cache paths, verifies the four wheel
checksums, and checks available disk space. It does not install packages or
validate GPU execution. Confirm that the paths are on the intended filesystem,
then install:

```bash
uv run --python 3.11.13 --no-project environment/install.py \
  --config "$SPECTER_CONFIG" 2>&1 | tee "$SPECTER_WORK/logs/install.log"
source "$SPECTER_WORK/env/bin/activate"
```

Successful installation ends with `Specter environment is ready.` The installer
uses 194 pinned runtime dependencies; the 190 packages outside the wheel bundle
require an online package index. If you changed `environment.venv`, activate
that directory instead of `env/`.

FlashInfer is pinned to `flashinfer-python==0.2.11.post3`. The installer also
passes `environment/build-constraints.txt` to uv for its isolated build:

```text
torch==2.7.0
setuptools==78.1.1
wheel==0.45.1
packaging==25.0
ninja==1.11.1.4
numpy==2.2.6
```

These constraints keep the build on the validated dependency versions instead
of resolving a newer Torch/CUDA stack independently of the runtime lock.

### Validate resources, build kernels, and test

Run the resource check before loading any model:

```bash
python -B -m configuration.check --config "$SPECTER_CONFIG"
```

It checks the selected checkpoint directories and referenced weight files, and
reads five nonempty inputs from each dataset. Missing paths and invalid data
are reported with their dataset name and resolved path. The main TPOT entry
also validates its inputs before model initialization. `--dry-run` on experiment
entries only prints the execution plan; it is not a resource check.

Set `CUDA_HOME` to your installed toolkit location and select an A100:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SPECTER_BUILD_VERBOSE=1
python -B -m streamlined_execution_engine.kernels.build \
  2>&1 | tee "$SPECTER_WORK/logs/build.log"
python -B -m pytest -p no:cacheprovider tests -q \
  2>&1 | tee "$SPECTER_WORK/logs/tests.log"
```

Build success prints the local `specter_cuda.so` path. Continue after the tests
pass. Build/test entry points apply the cache locations from `SPECTER_CONFIG`
before importing runtime libraries. Keep that variable exported when starting
an experiment, or pass `--config` explicitly.

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
