# Data Format

## Genotype

- supported inputs:
  `npy`
  `PLINK bed/bim/fam`
  `BGEN` with embedded sample IDs or `BGEN + sample`
- internal shape after loading: `(n_samples, n_markers)`
- encoding: numeric dosage or hard-call counts

PLINK BED is read directly in variant-major order. The effect allele is BIM A2.
For very large BIM files, `--genotype-cache-dir` stores validated fixed-width
NumPy metadata arrays and reopens them read-only with memory mapping. The cache
key includes the resolved BIM path, size, modification time, and schema; the
text BIM remains authoritative.
When `--sample-ids` is supplied, FAM IIDs are subset and reordered to match that
vector without rewriting the BED. On the packed CUDA path, variants with a
missing two-bit call in any selected sample and variants with zero residual
variance are omitted from the result table; exclusion counts are written to
`qc.json`. A hard-called BED can still contain PLINK's missing (`01`) state, so
the strict any-missing policy may require users to prepare a complete-data BED
before analysis.

BGEN is converted once into TorchGWAS's variant-major zstd dosage-store representation:

- expected dosage of BGEN allele 2: `P(A1/A2) + 2P(A2/A2)`
- quantized byte: `round(clip(dosage, 0, 2) × 127.5)`
- level-15 Zstandard compression
- one independent frame per 2,500 retained variants (`sub=1`)
- multiallelic, phased, non-diploid, incomplete, and invalid-probability variants skipped during conversion

Skipping an unsupported site does not abort the conversion. Cohort-level MAF, HLA/MHC, chromosome, marker-ID, and allele-alphabet filters are left to the user.

The `.complete.json` sidecar records the dosage scale and exclusion counts. Values are dequantized when beta and standard error are reported; scale-invariant test statistics can operate directly on the byte codes.

## Phenotype

- `.npy`, `.csv`, `.tsv`, or `.txt`
- shape: `(n_samples, n_traits)`
- quantitative traits only in v0.1
- tabular mode should include `IID`; `FID` is allowed and ignored for alignment

## Covariates

- `.npy`, `.csv`, `.tsv`, or `.txt`
- shape: `(n_samples, n_covariates)`
- columns with zero variance are dropped and reported in `qc.json`
- user should provide raw covariates, not a precomputed `covarQ`

## Alignment

When genotype is loaded from PLINK/BGEN, TorchGWAS uses genotype sample order as the source of truth by default.
Phenotype and covariate tables are reordered to that sample order via `IID`.
An explicit PLINK `--sample-ids` vector instead defines the analysis order and
must contain unique IIDs present in the FAM file.

## Outputs

Linear GWAS writes:

- `results.tsv.gz`
- `run.json`
- `qc.json`

The experimental multivariate mode also writes:

- `phenotype_correlation.json`
