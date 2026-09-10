from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch

from benchmark_bed_gpu_pipeline import measure_h2d_gbps, measure_packed_reads, measure_scan
from torchgwas.bed import PlinkBedGenotype
import torchgwas.linear as linear_module


def evict_local_file_pages(path: Path) -> bool:
    """Ask the client kernel to discard clean cached pages for one BED file."""

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a real packed BED path without materializing decoded genotypes"
    )
    parser.add_argument("--bed", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path, required=True)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metadata-cache-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compute-p-values",
        action="store_true",
        help="Include exact two-sided t-test p-values in the timed scan",
    )
    parser.add_argument(
        "--cold-scan-repeats",
        action="store_true",
        help="Request BED page-cache eviction immediately before each measured repeat",
    )
    parser.add_argument(
        "--skip-read-probes",
        action="store_true",
        help="Skip the separate cold/warm packed-read passes before the scan",
    )
    parser.add_argument(
        "--retain-t-array",
        action="store_true",
        help="Preallocate and fill the complete marker-by-trait NumPy t-statistic array",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Use eager PyTorch kernels instead of torch.compile",
    )
    parser.add_argument("--dump-t-dir", type=Path)
    parser.add_argument("--dump-writer-depth", type=int, default=4)
    parser.add_argument(
        "--skip-first-scan",
        action="store_true",
        help="Skip the separate first-use scan (useful for an eager cold-only run)",
    )
    parser.add_argument(
        "--discard-dumps",
        action="store_true",
        help="Validate and delete each NPY dump after its timed fsync completes",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.discard_dumps and args.dump_t_dir is None:
        raise ValueError("--discard-dumps requires --dump-t-dir")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.eager:
        linear_module._COMPILED_PACKED_BED_STATISTICS = None
    phenotype = np.load(args.phenotype, allow_pickle=False)
    covariates = np.load(args.covariates, allow_pickle=False)
    genotype = PlinkBedGenotype(
        args.bed,
        reader_workers=args.workers,
        prefetch_chunks=max(4, args.workers),
        metadata_cache_dir=args.metadata_cache_dir,
    )
    if phenotype.shape[0] != genotype.shape[0] or covariates.shape[0] != genotype.shape[0]:
        raise ValueError(
            "row-count mismatch: "
            f"BED={genotype.shape[0]}, phenotype={phenotype.shape[0]}, "
            f"covariates={covariates.shape[0]}"
        )

    evicted = False
    cold_read = None
    warm_read = None
    if not args.skip_read_probes:
        evicted = evict_local_file_pages(args.bed)
        cold_read = measure_packed_reads(genotype, args.chunk_size, args.workers)
        warm_read = measure_packed_reads(genotype, args.chunk_size, args.workers)
    first_dump_path = None
    if args.dump_t_dir is not None:
        first_dump_path = args.dump_t_dir / "first_scan.npy"
    first_scan = None
    if not args.skip_first_scan:
        first_scan = measure_scan(
            genotype,
            phenotype,
            covariates,
            chunk_size=args.chunk_size,
            workers=args.workers,
            device=device,
            compute_p_values=args.compute_p_values,
            retain_t_array=args.retain_t_array,
            dump_t_path=first_dump_path,
            dump_writer_depth=args.dump_writer_depth,
        )
        if args.discard_dumps and first_dump_path is not None:
            np.load(first_dump_path, mmap_mode="r", allow_pickle=False)
            first_dump_path.unlink()
            first_scan["dump_discarded_after_measurement"] = True
    repeats = []
    repeat_evictions = []
    for repeat_index in range(args.repeats):
        repeat_evictions.append(
            evict_local_file_pages(args.bed) if args.cold_scan_repeats else False
        )
        dump_path = (
            None
            if args.dump_t_dir is None
            else args.dump_t_dir / f"repeat_{repeat_index + 1}.npy"
        )
        measurement = measure_scan(
            genotype,
            phenotype,
            covariates,
            chunk_size=args.chunk_size,
            workers=args.workers,
            device=device,
            compute_p_values=args.compute_p_values,
            retain_t_array=args.retain_t_array,
            dump_t_path=dump_path,
            dump_writer_depth=args.dump_writer_depth,
        )
        if args.discard_dumps and dump_path is not None:
            dumped = np.load(dump_path, mmap_mode="r", allow_pickle=False)
            if dumped.shape != (genotype.shape[1], phenotype.shape[1]):
                raise RuntimeError(f"unexpected dumped t-statistic shape: {dumped.shape}")
            del dumped
            dump_path.unlink()
            measurement["dump_discarded_after_measurement"] = True
        repeats.append(measurement)
    result = {
        "bed": str(args.bed),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "n_samples": genotype.shape[0],
        "n_variants": genotype.shape[1],
        "n_traits": phenotype.shape[1],
        "covariate_columns": covariates.shape[1],
        "chunk_size": args.chunk_size,
        "workers": args.workers,
        "metadata_cache_path": (
            None
            if genotype.metadata_cache_path is None
            else str(genotype.metadata_cache_path)
        ),
        "compute_p_values": args.compute_p_values,
        "execution_mode": "eager" if args.eager else "torch.compile",
        "retain_t_array": args.retain_t_array,
        "dump_t_dir": None if args.dump_t_dir is None else str(args.dump_t_dir),
        "dump_writer_depth": args.dump_writer_depth,
        "skip_first_scan": args.skip_first_scan,
        "cold_scan_repeats": args.cold_scan_repeats,
        "repeat_cache_eviction_requested": repeat_evictions,
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "client_cache_eviction_requested": evicted,
        "h2d_gbps": measure_h2d_gbps(
            device=device,
            chunk_size=args.chunk_size,
            bytes_per_variant=genotype._bytes_per_variant,
        ),
        "cold_read": cold_read,
        "warm_read": warm_read,
        "first_scan": first_scan,
        "steady_scan_median": {
            key: statistics.median(run[key] for run in repeats) for key in repeats[0]
        },
        "steady_scan_repeats": repeats,
        "scan_exclusion_counts": getattr(genotype, "_last_scan_exclusion_counts", None),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
