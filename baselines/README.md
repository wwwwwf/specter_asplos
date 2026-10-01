# Baseline comparisons

Run the DeepSeek-V2-Lite comparison from the source root in the installed
Specter environment. The existing DeepSeek checkpoints, five prepared
datasets, and external TOML configuration are reused.

```bash
python -B -m benchmarks.run_ae --experiment speedup --memory high \
  --config ../specter.local.toml --output ae_speedup_high
python -B -m benchmarks.run_ae --experiment speedup --memory low \
  --config ../specter.local.toml --output ae_speedup_low
```

Use `--memory both --output ae_speedup_both` for both configurations in sequence.
Add `--dry-run` to preview the commands.

| Setting | Specter residents / K | SpecMoEOff residents / K | Mixtral-Offloading residents |
| --- | ---: | ---: | ---: |
| High | 16 / 16 | 12 / 3 | 32 |
| Low | 4 / 16 | 2 / 3 | 20 |

Resident counts are per MoE layer. Specter and SpecMoEOff each use 32 global
buffers; Mixtral-Offloading uses 4. These fixed presets report actual GPU
memory for each method.

Each method runs the same first five nonempty inputs from GK, WT, HE, GP,
and C4, with three repeats, greedy decoding, a 16-token prefix cap, and
128 output tokens: 75 measurements per method and setting. All measurements
are included in the summaries.

## Implementations

- **Mixtral-Offloading:** the upstream demand-swap expert cache with the
  author's DeepSeek FP16 adapter and unfused expert dispatch. See
  [source reference and license](mixtral_offloading/PROVENANCE.md).
- **SpecMoEOff:** the author's retained Transformers implementation, with
  an INT4 draft, an FP16 target, route-frequency prefetching, and independent
  weights and KV caches. See [source reference](specmoeoff/PROVENANCE.md).

## Outputs

For one setting, results are under `<output>/speedup/<high|low>/<method>/`.
`comparison.json` and `comparison.csv` are in `<output>/speedup/`. With
`--memory both`, the comparison roots are `<output>/speedup_high/` and
`<output>/speedup_low/`. All paths resolve under the configured output root,
outside the source tree. Existing outputs are not overwritten.

`[result] current/75` shows progress. Raw records include generated tokens,
timing, and memory. Comparisons check matching inputs, seeds, workload,
configuration, device, and timing definition, and report output agreement.

TPOT is `(last_commit_ms - first_commit_ms) / (N - 1)` for every method.
TTFT and end-to-end latency are reported separately. Speedup is the baseline
mean TPOT divided by Specter's mean TPOT.

Add `--require-greedy-match` to require exact checked output agreement.
To reuse a matching Specter run, add
`--reference-root /path/to/comparison-root`, where the root contains
`<high|low>/specter/`. Both baselines run and are compared with that reference.
The reference must match the workload, resident count, configuration,
device selection, and CPU placement. For `--memory both` results, reuse each
tier's `speedup_high/` or `speedup_low/` comparison root separately.

## Individual methods

For example, run only SpecMoEOff low against a matching Specter reference:

```bash
python -B -m baselines.run --config ../specter.local.toml \
  --methods specmoeoff --memory low --memory-policy fixed \
  --specmoeoff-resident-per-layer 2 \
  --reference-root /path/to/comparison-root --output ds_low_specmoeoff
```

For Mixtral-Offloading, use `--methods mixtral-offloading --memory low
--mixtral-resident-per-layer 20`.

Run baseline CPU regression tests with:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m pytest -p no:cacheprovider -q \
  tests/test_speedup_entry.py tests/test_baseline_entry.py \
  tests/test_baseline_memory.py tests/test_gpu_memory.py baselines
```
