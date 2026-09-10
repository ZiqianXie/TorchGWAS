from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_bed_gpu_pipeline import measure_packed_reads
from torchgwas.bed import PlinkBedGenotype


def evict_file_pages(path: Path) -> bool:
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
        description="Measure cold packed-BED bandwidth versus reader worker count"
    )
    parser.add_argument("--bed", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--workers", default="1,4,8,12,24")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--metadata-cache-dir", type=Path)
    args = parser.parse_args()

    worker_counts = [int(value) for value in args.workers.split(",")]
    genotype = PlinkBedGenotype(
        args.bed,
        reader_workers=max(worker_counts),
        prefetch_chunks=max(worker_counts) + 2,
        metadata_cache_dir=args.metadata_cache_dir,
    )
    rows = []
    for workers in worker_counts:
        for repeat in range(1, args.repeats + 1):
            evicted = evict_file_pages(args.bed)
            measurement = measure_packed_reads(
                genotype,
                chunk_size=args.chunk_size,
                workers=workers,
            )
            rows.append(
                {
                    "workers": workers,
                    "repeat": repeat,
                    "cache_eviction_requested": evicted,
                    **measurement,
                }
            )
    print(
        json.dumps(
            {
                "bed": str(args.bed),
                "n_samples": genotype.shape[0],
                "n_variants": genotype.shape[1],
                "bed_bytes": args.bed.stat().st_size,
                "chunk_size": args.chunk_size,
                "measurements": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
