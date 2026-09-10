from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import zstandard as zstd


def decode_bed_row(raw: bytes, n_samples: int) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=np.uint8)
    calls = np.stack(
        (
            packed & 3,
            (packed >> 2) & 3,
            (packed >> 4) & 3,
            (packed >> 6) & 3,
        ),
        axis=1,
    ).reshape(-1)[:n_samples]
    dosage = np.asarray([2, -1, 1, 0], dtype=np.int8)[calls]
    return dosage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare selected hard-called BED rows with their source zstd dosages"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--bed", type=Path, required=True)
    parser.add_argument("--variant-indices", default="0,1,2499,2500,4043050,8086099,8086100")
    args = parser.parse_args()

    index = np.load(f"{args.store}.idx.npz", allow_pickle=False)
    n_variants = int(index["nsnp"])
    n_samples = int(index["nsamp"])
    chunk_size = int(index["chunk"])
    subframes = int(index["sub"])
    offsets = index["offs"]
    sizes = index["sizes"]
    if subframes != 1:
        raise ValueError("this validator currently requires one zstd frame per chunk")
    bytes_per_variant = (n_samples + 3) // 4
    expected_size = 3 + n_variants * bytes_per_variant
    if args.bed.stat().st_size != expected_size:
        raise ValueError(
            f"BED size is {args.bed.stat().st_size}; expected {expected_size}"
        )

    selected = [int(value) for value in args.variant_indices.split(",")]
    if any(value < 0 or value >= n_variants for value in selected):
        raise ValueError("variant index is outside the zstd store")
    zfd = os.open(f"{args.store}.zst", os.O_RDONLY)
    bfd = os.open(args.bed, os.O_RDONLY)
    decoder = zstd.ZstdDecompressor()
    checked_samples = 0
    try:
        for variant_index in selected:
            chunk_index = variant_index // chunk_size
            chunk_start = chunk_index * chunk_size
            rows = min(chunk_size, n_variants - chunk_start)
            compressed = os.pread(
                zfd,
                int(sizes[chunk_index]),
                int(offsets[chunk_index]),
            )
            decoded = decoder.decompress(
                compressed,
                max_output_size=rows * n_samples,
            )
            dosage_u8 = np.frombuffer(decoded, dtype=np.uint8).reshape(rows, n_samples)[
                variant_index - chunk_start
            ]
            expected = np.where(dosage_u8 < 64, 0, np.where(dosage_u8 < 192, 1, 2))
            bed_raw = os.pread(
                bfd,
                bytes_per_variant,
                3 + variant_index * bytes_per_variant,
            )
            observed = decode_bed_row(bed_raw, n_samples)
            if not np.array_equal(observed, expected):
                mismatches = int(np.count_nonzero(observed != expected))
                raise AssertionError(
                    f"variant {variant_index} has {mismatches}/{n_samples} hard-call mismatches"
                )
            checked_samples += n_samples
    finally:
        os.close(zfd)
        os.close(bfd)

    result = {
        "bed": str(args.bed),
        "source_store": str(args.store),
        "n_samples": n_samples,
        "n_variants": n_variants,
        "bed_bytes": expected_size,
        "variant_indices_checked": selected,
        "hard_calls_checked": checked_samples,
        "hard_call_mismatches": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
