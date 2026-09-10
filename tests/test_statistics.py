from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from torchgwas.api import run_linear_gwas
from torchgwas.bed import PlinkBedGenotype, resolve_plink_triplet
from torchgwas.linear import _unpack_plink_a2_float, linear_scan, linear_scan_streaming
from torchgwas.preprocess import residualize_and_standardize


def _write_bed(prefix: Path, dosage_a2: np.ndarray) -> Path:
    bed, bim, fam = resolve_plink_triplet(prefix)
    n_samples, n_markers = dosage_a2.shape
    fam.write_text("".join(f"F{i} I{i} 0 0 0 -9\n" for i in range(n_samples)))
    bim.write_text("".join(f"1 rs{i} 0 {i + 1} A G\n" for i in range(n_markers)))
    dosage_to_code = {2: 0b00, 1: 0b10, 0: 0b11}
    payload = bytearray(b"\x6c\x1b\x01")
    for marker in range(n_markers):
        for sample_start in range(0, n_samples, 4):
            byte = 0
            for offset in range(4):
                sample = sample_start + offset
                code = (
                    0b01
                    if sample >= n_samples or np.isnan(dosage_a2[sample, marker])
                    else dosage_to_code[int(dosage_a2[sample, marker])]
                )
                byte |= code << (2 * offset)
            payload.append(byte)
    bed.write_bytes(payload)
    return bed


class ExactLinearStatisticsTestCase(unittest.TestCase):
    def test_packed_plink_decoder_preserves_a2_dosage_and_missing_calls(self):
        # PLINK pairs are least-significant first: 00, 10, 11, 01.
        packed = torch.tensor([[0b01_11_10_00]], dtype=torch.uint8)
        observed = _unpack_plink_a2_float(packed, n_samples=4).numpy()
        expected = np.asarray([[2.0, 1.0, 0.0, np.nan]], dtype=np.float32)
        np.testing.assert_allclose(observed, expected, equal_nan=True)

    def test_packed_plink_decoder_trims_byte_padding(self):
        packed = torch.tensor([[0b11_11_10_00]], dtype=torch.uint8)
        observed = _unpack_plink_a2_float(packed, n_samples=2).numpy()
        np.testing.assert_array_equal(observed, np.asarray([[2.0, 1.0]], dtype=np.float32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the packed BED scan")
    def test_packed_bed_cuda_scan_matches_float64_reference(self):
        rng = np.random.default_rng(20260911)
        n_samples = 65
        genotype = rng.integers(0, 3, size=(n_samples, 13), dtype=np.uint8)
        # Make every marker nonconstant in this small randomized fixture.
        genotype[:3, :] = np.arange(3, dtype=np.uint8)[:, None]
        covariate = rng.normal(size=n_samples)
        covariates = np.column_stack((covariate, 2.0 * covariate))
        phenotype = np.column_stack(
            (
                0.3 * genotype[:, 2] + covariate + rng.normal(size=n_samples),
                -0.2 * genotype[:, 9] - covariate + rng.normal(size=n_samples),
            )
        )
        beta_ref, t_ref, p_ref, _ = linear_scan(
            genotype.astype(np.float64),
            phenotype,
            covariates,
            chunk_size=5,
            device="cpu",
            compute_dtype="float64",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bed = _write_bed(Path(tmpdir) / "fixture", genotype)
            packed = PlinkBedGenotype(bed, reader_workers=2, prefetch_chunks=2)
            beta, t_stat, p_value, _ = linear_scan_streaming(
                packed,
                phenotype,
                covariates,
                chunk_size=5,
                device="cuda:0",
                compute_dtype="float32",
                reader_workers=2,
            )
            output_dir = Path(tmpdir) / "topk"
            streamed = run_linear_gwas(
                genotype=bed,
                phenotype=phenotype,
                covariates=covariates,
                genotype_format="plink",
                chunk_size=5,
                device="cuda:0",
                compute_dtype="float32",
                reader_workers=2,
                topk_per_trait=2,
                output_dir=output_dir,
            )
            with gzip.open(output_dir / "results.tsv.gz", "rt", newline="") as handle:
                top_rows = list(csv.DictReader(handle, delimiter="\t"))
        np.testing.assert_allclose(beta, beta_ref, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(t_stat, t_ref, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(p_value, p_ref, rtol=2e-4, atol=1e-7)
        self.assertEqual(streamed.qc_summary["genotype_qc_mode"], "fused_gpu_scan")
        self.assertEqual(len(top_rows), 4)
        for trait_index in range(2):
            observed = {
                row["marker_id"]
                for row in top_rows
                if row["trait"] == f"trait_{trait_index}"
            }
            expected = {
                f"rs{index}"
                for index in np.argsort(np.abs(t_ref[:, trait_index]))[-2:]
            }
            self.assertEqual(observed, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the packed BED scan")
    def test_packed_bed_cuda_scan_rejects_missing_calls(self):
        rng = np.random.default_rng(20260912)
        genotype = rng.integers(0, 3, size=(65, 13)).astype(np.float32)
        genotype[:3, :] = np.arange(3, dtype=np.float32)[:, None]
        genotype[7, 3] = np.nan
        phenotype = rng.normal(size=(65, 2))
        covariates = rng.normal(size=(65, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            bed = _write_bed(Path(tmpdir) / "missing", genotype)
            packed = PlinkBedGenotype(bed, reader_workers=2, prefetch_chunks=2)
            with self.assertRaisesRegex(ValueError, "marker 3"):
                linear_scan_streaming(
                    packed,
                    phenotype,
                    covariates,
                    chunk_size=5,
                    device="cuda:0",
                    compute_dtype="float32",
                    reader_workers=2,
                )

    def test_matches_explicit_ols_with_rank_deficient_covariates(self):
        rng = np.random.default_rng(20260910)
        n_samples = 96
        genotype = rng.integers(0, 3, size=(n_samples, 7)).astype(np.float64)
        covariate = rng.normal(size=n_samples)
        covariates = np.column_stack([covariate, 2.0 * covariate])
        phenotype = np.column_stack(
            [
                0.4 * genotype[:, 1] + 0.8 * covariate + rng.normal(size=n_samples),
                -0.2 * genotype[:, 4] - 0.5 * covariate + rng.normal(size=n_samples),
            ]
        )

        beta, t_stat, p_value, q_matrix = linear_scan(
            genotype,
            phenotype,
            covariates,
            chunk_size=3,
            device="cpu",
            compute_dtype="float64",
        )
        phenotype_processed, q_reference = residualize_and_standardize(phenotype, covariates)
        self.assertIsNotNone(q_matrix)
        self.assertEqual(q_matrix.shape[1], 1)
        np.testing.assert_allclose(np.abs(q_matrix), np.abs(q_reference), atol=1e-12)

        beta_reference = np.empty_like(beta)
        t_reference = np.empty_like(t_stat)
        p_reference = np.empty_like(p_value)
        for marker in range(genotype.shape[1]):
            design = np.column_stack([np.ones(n_samples), q_reference, genotype[:, marker]])
            coefficients, _, rank, _ = np.linalg.lstsq(design, phenotype_processed, rcond=None)
            residual = phenotype_processed - design @ coefficients
            df = n_samples - rank
            sigma2 = np.sum(residual * residual, axis=0) / df
            covariance_last = np.linalg.pinv(design.T @ design)[-1, -1]
            standard_error = np.sqrt(sigma2 * covariance_last)
            beta_reference[marker] = coefficients[-1]
            t_reference[marker] = coefficients[-1] / standard_error
            p_reference[marker] = 2.0 * stats.t.sf(np.abs(t_reference[marker]), df=df)

        np.testing.assert_allclose(beta, beta_reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(t_stat, t_reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(p_value, p_reference, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
