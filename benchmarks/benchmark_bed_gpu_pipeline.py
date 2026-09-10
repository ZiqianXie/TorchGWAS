from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from torchgwas.bed import PlinkBedGenotype, resolve_plink_triplet
from torchgwas.linear import linear_scan_streaming_chunks


def write_valid_synthetic_bed(
    prefix: Path,
    n_samples: int,
    n_variants: int,
    seed: int,
) -> Path:
    """Write variant-major BED calls without missing or invariant variants."""

    bed, bim, fam = resolve_plink_triplet(prefix)
    fam.write_text("".join(f"F{i} I{i} 0 0 0 -9\n" for i in range(n_samples)))
    bim.write_text("".join(f"1 rs{i} 0 {i + 1} A G\n" for i in range(n_variants)))
    valid_bytes = np.asarray(
        [
            a | (b << 2) | (c << 4) | (d << 6)
            for a in (0, 2, 3)
            for b in (0, 2, 3)
            for c in (0, 2, 3)
            for d in (0, 2, 3)
        ],
        dtype=np.uint8,
    )
    bytes_per_variant = (n_samples + 3) // 4
    rng = np.random.default_rng(seed)
    with bed.open("wb", buffering=8 << 20) as handle:
        handle.write(b"\x6c\x1b\x01")
        for start in range(0, n_variants, 512):
            count = min(512, n_variants - start)
            handle.write(rng.choice(valid_bytes, size=(count, bytes_per_variant)).tobytes())
    return bed


def measure_h2d_gbps(
    *,
    device: torch.device,
    chunk_size: int,
    bytes_per_variant: int,
    repetitions: int = 40,
) -> float:
    """Measure the selected GPU's H2D link with the exact packed chunk shape."""

    host = torch.zeros(
        (chunk_size, bytes_per_variant),
        dtype=torch.uint8,
        pin_memory=True,
    )
    target = torch.empty_like(host, device=device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        target.copy_(host, non_blocking=True)
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(repetitions):
            target.copy_(host, non_blocking=True)
        end.record(stream)
    end.synchronize()
    seconds = start.elapsed_time(end) / 1000.0
    return host.numel() * host.element_size() * repetitions / seconds / 1e9


def measure_packed_reads(genotype: PlinkBedGenotype, chunk_size: int, workers: int) -> dict:
    setup_started = time.perf_counter()
    loader = genotype.iter_packed_chunks(
        chunk_size=chunk_size,
        reader_workers=workers,
        depth=max(3, workers + 2),
    )
    setup_seconds = time.perf_counter() - setup_started
    checksum = 0
    started = time.perf_counter()
    try:
        for buffer_index, packed, start, end in loader:
            checksum ^= int(packed[0, 0]) + start + end
            loader.release(buffer_index)
    finally:
        loader.close()
    wall_seconds = time.perf_counter() - started
    payload_bytes = genotype.shape[1] * genotype._bytes_per_variant
    return {
        "read_wall_seconds": wall_seconds,
        "read_pinned_setup_seconds": setup_seconds,
        "read_packed_gbps": payload_bytes / wall_seconds / 1e9,
        "read_checksum": checksum,
    }


def measure_scan(
    genotype: PlinkBedGenotype,
    phenotype: np.ndarray,
    covariates: np.ndarray,
    *,
    chunk_size: int,
    workers: int,
    device: torch.device,
    compute_p_values: bool,
    retain_t_array: bool = False,
) -> dict:
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    chunks, _ = linear_scan_streaming_chunks(
        genotype,
        phenotype,
        covariates,
        chunk_size=chunk_size,
        device=str(device),
        compute_dtype="float32",
        reader_workers=workers,
        compute_p_values=compute_p_values,
    )
    checksum = 0.0
    first_result_seconds = None
    t_output = (
        np.empty((genotype.shape[1], phenotype.shape[1]), dtype=np.float32)
        if retain_t_array
        else None
    )
    for start, end, beta, t_stat, p_value in chunks:
        if first_result_seconds is None:
            first_result_seconds = time.perf_counter() - started
        checksum += float(beta[0, 0] + t_stat[-1, -1])
        if t_output is not None:
            t_output[start:end] = t_stat
        if p_value is not None:
            checksum += float(p_value[0, -1])
        checksum += start + end
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    return {
        "scan_wall_seconds": wall_seconds,
        "first_result_seconds": first_result_seconds,
        "scan_variants_per_second": genotype.shape[1] / wall_seconds,
        "scan_packed_gbps": (
            genotype.shape[1] * genotype._bytes_per_variant / wall_seconds / 1e9
        ),
        "scan_decoded_gbps": (
            genotype.shape[0] * genotype.shape[1] * 4 / wall_seconds / 1e9
        ),
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "scan_checksum": checksum,
        "retained_t_numpy_bytes": 0 if t_output is None else int(t_output.nbytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark native packed BED reads, selected-GPU H2D, and exact linear scan"
    )
    parser.add_argument("--n-samples", type=int, default=22_250)
    parser.add_argument("--n-variants", type=int, default=120_000)
    parser.add_argument("--n-traits", type=int, default=128)
    parser.add_argument("--covariate-rank", type=int, default=27)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-p-values", action="store_true")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    phenotype = rng.normal(size=(args.n_samples, args.n_traits))
    covariates = rng.normal(size=(args.n_samples, args.covariate_rank))

    with tempfile.TemporaryDirectory(prefix="torchgwas-packed-bed-") as tmpdir:
        bed = write_valid_synthetic_bed(
            Path(tmpdir) / "synthetic",
            args.n_samples,
            args.n_variants,
            args.seed,
        )
        genotype = PlinkBedGenotype(
            bed,
            reader_workers=args.workers,
            prefetch_chunks=max(4, args.workers),
        )
        # Populate the filesystem cache before measuring steady-state scheduling.
        with bed.open("rb") as handle:
            while handle.read(8 << 20):
                pass

        result = {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "n_samples": args.n_samples,
            "n_variants": args.n_variants,
            "n_traits": args.n_traits,
            "covariate_rank": args.covariate_rank,
            "chunk_size": args.chunk_size,
            "workers": args.workers,
            "h2d_gbps": measure_h2d_gbps(
                device=device,
                chunk_size=args.chunk_size,
                bytes_per_variant=genotype._bytes_per_variant,
            ),
        }
        result.update(measure_packed_reads(genotype, args.chunk_size, args.workers))

        # First call includes TorchInductor setup; report it separately. Later
        # calls measure the stable allocator/event/buffer-reuse regime.
        first = measure_scan(
            genotype,
            phenotype,
            covariates,
            chunk_size=args.chunk_size,
            workers=args.workers,
            device=device,
            compute_p_values=not args.skip_p_values,
        )
        repeats = [
            measure_scan(
                genotype,
                phenotype,
                covariates,
                chunk_size=args.chunk_size,
                workers=args.workers,
                device=device,
                compute_p_values=not args.skip_p_values,
            )
            for _ in range(args.repeats)
        ]
        result["first_scan"] = first
        result["steady_scan_median"] = {
            key: statistics.median(run[key] for run in repeats)
            for key in repeats[0]
        }
        result["steady_scan_repeats"] = repeats
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
