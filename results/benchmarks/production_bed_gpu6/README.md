# Production-scale direct-BED benchmark

This bundle records the direct PLINK BED experiment used in the 2026-09-10
implementation/manuscript audit. The fixture has 22,250 samples, 8,086,101
variants, and 128 phenotypes. Its BED payload is 44,982,979,866 bytes.

## Timing boundary

Measured cold scans include positional BED reads, packed H2D, GPU unpacking,
covariate-adjusted beta/t computation, fused missing/invariant checks, and D2H
into NumPy chunk arrays. Exact Student-t tail evaluation and filesystem output
are excluded. Each measured cold repeat calls `POSIX_FADV_DONTNEED` for the BED
before timing. Hot-cache results are not used for the headline value.

## Results

| Experiment | Result |
|---|---:|
| Cold read, 1 worker | 81.13 s / 0.554 GB/s |
| Cold read, 4 workers | 27.45 s / 1.638 GB/s |
| Cold read, 8 workers | 29.27 s / 1.537 GB/s |
| Cold read, 12 workers | 30.60 s / 1.470 GB/s |
| Cold read, 24 workers | 29.67 s / 1.516 GB/s |
| Compiled cold scans, 6 repeats | 25.97-36.32 s; median 29.81 s |
| Eager cold scan | 36.16 s |

All scan repeats produced checksum `13089616136.36541` (the eager result
differs only in the last printed floating-point digits). No missing or invariant
variants were observed. The independent BED-versus-zstd check covered seven
variants and 155,750 calls with zero mismatches.

## Interpretation

Four readers saturate this node-local disk. The compiled scan's median packed
throughput is close to the independently measured cold-read ceiling, so GPU
work is substantially hidden behind storage. Eager mode remains below one
minute without a compilation-cache dependency and is appropriate for a
one-off scan; compilation is useful for repeated scans.

## Files

- `read_scaling_cold.json`: one page-cache-evicted read per worker count.
- `compiled_cold_run1.json`, `compiled_cold_run2.json`: two independent
  compiled invocations, three cold measured repeats each.
- `eager_cold.json`: eager-mode invocation with one cold measured repeat.
- `bed_vs_zstd_validation.json`: independent hard-call comparison.
- `fixture_conversion.json`: provenance for the BED fixture construction.
- `metadata_cache_timing.json`: one-time cache build and cached reopen timing.

The large BED and metadata cache are intentionally not committed.
