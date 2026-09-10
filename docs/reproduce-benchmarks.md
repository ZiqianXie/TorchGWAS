# Benchmark Reproduction

## Included summaries

Generated benchmark tables are stored in `results/benchmarks/`.

## PLINK concordance from an external baseline directory

The command below expects an existing local baseline output directory and is not required for the toy workflow.

```bash
python benchmarks/summarize_plink_accuracy.py \
  --fast-gwas-root /path/to/fast_GWAS \
  --trait-indices 0,17,63,127 \
  --output results/benchmarks/accuracy_plink.tsv
```

## Historical runtime points from the original notebook

```bash
python benchmarks/materialize_runtime_table.py \
  --output results/benchmarks/runtime_scaling.tsv
```

## Native BED reader scaling

This benchmark creates a temporary variant-major BED file, reads every variant, and reports wall time, process CPU time, throughput, and peak RSS for each worker count:

```bash
PYTHONPATH=src python benchmarks/benchmark_bed_reader.py \
  --n-samples 35365 \
  --n-variants 12000 \
  --chunk-size 1000 \
  --workers 1,2,4,8
```

The A100 lab-host measurements and the corresponding public-main `pandas-plink` baseline are recorded in `results/benchmarks/bed_reader_parallel_remote_20260910.tsv`.

### Native BED GPU pipeline

The end-to-end benchmark keeps PLINK's packed two-bit calls through positional
reads and H2D transfer, then decodes and runs exact covariate-adjusted OLS on
the selected GPU. Run it separately for every candidate GPU because devices of
the same model can have materially different host-to-device bandwidth through
their PCIe switch and NUMA placement:

```bash
PYTHONPATH=src python benchmarks/benchmark_bed_gpu_pipeline.py \
  --device cuda:0 \
  --n-samples 22250 \
  --n-variants 120000 \
  --n-traits 128 \
  --covariate-rank 27 \
  --chunk-size 5000 \
  --workers 24 \
  --skip-p-values
```

`--skip-p-values` measures the scan core used by filtered streaming output.
Without it, the benchmark also measures exact p-values for every marker-trait
pair. Report both first-use time and the median repeated time. The first call
includes TorchInductor compilation; every call includes pinned-memory
registration, and the rings are then reused for all chunks in that scan. Use OS
CPU/NUMA affinity controls when comparing GPUs, and do not add per-GPU
bandwidths unless the links have been shown to operate concurrently without
sharing a bottleneck.

For an existing BED with matched phenotype and covariate arrays, measure the
packed read boundary and the steady scan separately. Add `--retain-t-array`
when the production contract returns the complete marker-by-trait t-statistic
matrix in host NumPy memory, or `--dump-t-dir` to pipeline a complete `.npy`
dump to storage:

```bash
PYTHONPATH=src:benchmarks python benchmarks/benchmark_bed_gpu_real.py \
  --bed /path/to/study.bed \
  --phenotype /path/to/phenotypes.npy \
  --covariates /path/to/covariates.npy \
  --device cuda:0 \
  --chunk-size 5000 \
  --workers 4 \
  --repeats 3 \
  --cold-scan-repeats \
  --skip-read-probes \
  --dump-t-dir /local/output/t_statistics \
  --dump-writer-depth 4
```

With `--cold-scan-repeats`, the script requests per-file client page-cache
eviction before every measured repeat when `posix_fadvise` is available. Use
those cold repeats for manuscript-facing storage claims. First-use compilation
is reported separately; hot-cache measurements are diagnostic only and should
not replace cold-input results. The NPY writer uses a bounded staging ring and
one ordered writer thread; the timer ends only after `flush` and `fsync`, so the
reported value includes durable output rather than page-cache admission alone.

## Calibrated stage runtime predictor

The predictor separates compressed disk read, CPU Zstandard decode, host-to-device transfer, and GPU compute. The hardware model uses measured FP32 throughput, CPU frequency, decode thread count, and H2D bandwidth; `--pipeline-contention-factor` comes from a short joint-stage calibration and captures NUMA/PCIe contention.

The following reference calibration reproduces the measured A100-host ceiling (about 6.7 seconds for the scan itself):

```bash
PYTHONPATH=src python benchmarks/predict_runtime.py \
  --model hardware \
  --n-variants 8086101 \
  --n-samples 22250 \
  --n-traits 128 \
  --covariate-rank 27 \
  --gpu-fp32-tflops 19.1 \
  --gpu-count 2 \
  --cpu-frequency-ghz 2.3 \
  --decode-threads 40 \
  --disk-gbps 100 \
  --h2d-gbps 42.5 \
  --pipeline-contention-factor 1.584
```

Use measured sustained values for a new system. Peak vendor specifications alone do not capture filesystem cache state, PCIe topology, NUMA placement, or decode/H2D contention.
For native packed BED input, pass `--h2d-bytes-per-value 0.25`; the GPU receives
the on-disk two-bit calls and expands them after transfer. Measure
`--h2d-gbps` on the exact selected GPU rather than reusing a value from another
GPU of the same model.

## Toy multivariate case summary

This benchmark is kept as an exploratory extension rather than the main reported workflow.

```bash
python benchmarks/run_multivariate_case.py \
  --output results/benchmarks/multivariate_case.tsv
```

## Simulated CPU/GPU memory profile for linear GWAS

This benchmark runs linear TorchGWAS on simulated matrices and records:

- peak CPU resident set size for each isolated subprocess
- peak GPU allocated and reserved memory reported by PyTorch
- runtime for each configuration

```bash
python benchmarks/profile_linear_memory.py \
  --shapes 1024x4096x8 2048x8192x8 \
  --devices cpu cuda \
  --chunk-size 1024 \
  --output results/benchmarks/linear_memory_profile.tsv
```

## Large-scale streaming profile at 20k samples / 1M SNPs / 2k traits

For a more realistic large-scale memory benchmark, use the streaming profile script.
This benchmark measures peak steady-state memory on repeated chunks and extrapolates runtime to the full marker count without materializing the complete result table.

```bash
python benchmarks/profile_linear_streaming_large.py \
  --n-samples 20000 \
  --n-markers-total 1000000 \
  --n-traits 2000 \
  --chunk-size 4096 \
  --devices cpu cuda \
  --output results/benchmarks/linear_streaming_profile_large.tsv
```
