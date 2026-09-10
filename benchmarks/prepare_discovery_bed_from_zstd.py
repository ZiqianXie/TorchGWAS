from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def write_fam(path: Path, sample_ids: np.ndarray) -> None:
    with path.open("w") as handle:
        for value in sample_ids:
            sample_id = str(value)
            handle.write(f"{sample_id}\t{sample_id}\t0\t0\t0\t-9\n")


def write_bim(path: Path, prep_path: Path, block_size: int = 100_000) -> int:
    prep = np.load(prep_path, allow_pickle=False)
    required = {"chrom", "pos", "snp", "a1", "a2"}
    missing = required.difference(prep.files)
    if missing:
        raise ValueError(f"{prep_path} is missing arrays: {sorted(missing)}")
    arrays = {key: prep[key] for key in required}
    prep.close()
    n_variants = int(arrays["pos"].shape[0])
    if any(arrays[key].shape[0] != n_variants for key in required):
        raise ValueError(f"variant metadata arrays in {prep_path} have inconsistent lengths")
    # The project prep file names BGEN allele 2 as a1 (alternative) and allele
    # 1 as a2 (first).  TorchGWAS reports BED dosage for BIM A2, so placing the
    # BGEN alternative allele in BIM column 6 preserves the dosage orientation.
    with path.open("w", buffering=8 << 20) as handle:
        for start in range(0, n_variants, block_size):
            end = min(n_variants, start + block_size)
            frame = pd.DataFrame(
                {
                    "chrom": arrays["chrom"][start:end],
                    "snp": arrays["snp"][start:end],
                    "cm": 0,
                    "pos": arrays["pos"][start:end],
                    "a1": arrays["a2"][start:end],
                    "a2": arrays["a1"][start:end],
                }
            )
            frame.to_csv(handle, sep="\t", header=False, index=False)
    return n_variants


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hard-call the established TorchGWAS zstd dosage store into PLINK BED"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True, help="Prefix before .zst/.idx.npz")
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--sample-order", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=40)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reuse-metadata",
        action="store_true",
        help="Reuse an existing complete BIM/FAM pair and regenerate only BED",
    )
    parser.add_argument(
        "--resume-bed",
        action="store_true",
        help="Resume an existing BED whose payload ends on a zstd chunk boundary",
    )
    parser.add_argument("--metadata-cache-dir", type=Path)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Create the binary BIM metadata cache for an existing complete BED triplet",
    )
    args = parser.parse_args()

    index = np.load(f"{args.store}.idx.npz", allow_pickle=False)
    n_variants = int(index["nsnp"])
    n_samples = int(index["nsamp"])
    sample_ids = np.load(args.sample_order, allow_pickle=False)
    if sample_ids.shape != (n_samples,):
        raise ValueError(
            f"sample order has shape {sample_ids.shape}; zstd store has {n_samples} samples"
        )

    output = args.output_prefix
    output.parent.mkdir(parents=True, exist_ok=True)
    bed_path = output.with_suffix(".bed")
    bim_path = output.with_suffix(".bim")
    fam_path = output.with_suffix(".fam")
    expected_bed_bytes = 3 + n_variants * ((n_samples + 3) // 4)
    if args.cache_only:
        if args.metadata_cache_dir is None:
            raise ValueError("--cache-only requires --metadata-cache-dir")
        for path in (bed_path, bim_path, fam_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if bed_path.stat().st_size != expected_bed_bytes:
            raise ValueError(
                f"BED size mismatch: expected {expected_bed_bytes}, got {bed_path.stat().st_size}"
            )
        from torchgwas.bed import write_plink_bim_cache

        prep = np.load(args.prep, allow_pickle=False)
        cache_started = time.perf_counter()
        cache_path = write_plink_bim_cache(
            args.metadata_cache_dir,
            bim_path,
            chromosomes=prep["chrom"],
            marker_ids=prep["snp"],
            positions=prep["pos"],
            other_alleles=prep["a2"],
            effect_alleles=prep["a1"],
        )
        prep.close()
        print(
            json.dumps(
                {
                    "metadata_cache_path": str(cache_path),
                    "n_variants": n_variants,
                    "cache_seconds": time.perf_counter() - cache_started,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if bed_path.exists() and not args.resume_bed:
        raise FileExistsError(f"refusing to overwrite {bed_path}")
    if args.reuse_metadata:
        for path in (bim_path, fam_path):
            if not path.is_file():
                raise FileNotFoundError(path)
    else:
        for path in (bim_path, fam_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")

    started = time.perf_counter()
    if not args.reuse_metadata:
        write_fam(fam_path, sample_ids)
        metadata_variants = write_bim(bim_path, args.prep)
        if metadata_variants != n_variants:
            raise ValueError(
                f"prep has {metadata_variants} variants; zstd store has {n_variants}"
            )
    metadata_seconds = time.perf_counter() - started

    # uint8 stores round(expected allele-2 dosage * 127.5).  Nearest-integer
    # hard calls therefore map 0..63 -> 0, 64..191 -> 1, and 192..255 -> 2.
    # PLINK variant-major codes are 11=A1/A1, 10=A1/A2, 00=A2/A2.
    import torch

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    bytes_per_variant = (n_samples + 3) // 4
    padded_samples = bytes_per_variant * 4
    chunk_size = int(index["chunk"])
    subframes = int(index["sub"])
    if subframes != 1:
        raise ValueError("BED preparation currently requires one zstd frame per chunk")
    offsets = index["offs"]
    sizes = index["sizes"]
    dosage_device = torch.empty(
        (chunk_size, padded_samples), dtype=torch.uint8, device=device
    )
    packed_host = torch.empty(
        (chunk_size, bytes_per_variant), dtype=torch.uint8, pin_memory=True
    )

    def pack_dosage(dosage):
        calls = ((dosage < 192).to(torch.uint8) << 1) | (dosage < 64).to(torch.uint8)
        calls = calls.reshape(chunk_size, bytes_per_variant, 4)
        return (
            calls[:, :, 0]
            | (calls[:, :, 1] << 2)
            | (calls[:, :, 2] << 4)
            | (calls[:, :, 3] << 6)
        )

    dosage_device.fill_(255)
    packed_device = pack_dosage(dosage_device)
    packed_device.sum().item()
    if args.resume_bed:
        current_bytes = bed_path.stat().st_size
        if current_bytes < 3 or (current_bytes - 3) % bytes_per_variant:
            raise ValueError(f"BED resume point is not variant-aligned: {current_bytes} bytes")
        emitted = (current_bytes - 3) // bytes_per_variant
        if emitted != n_variants and emitted % chunk_size:
            raise ValueError(
                f"BED resume point {emitted} is not on a {chunk_size}-variant chunk boundary"
            )
        file_mode = "ab"
    else:
        emitted = 0
        file_mode = "wb"
    resume_start = int(emitted)
    encode_started = time.perf_counter()
    import zstandard as zstd

    decoder = zstd.ZstdDecompressor()
    source_fd = os.open(f"{args.store}.zst", os.O_RDONLY)
    try:
        with bed_path.open(file_mode, buffering=32 << 20) as handle:
            if not args.resume_bed:
                handle.write(b"\x6c\x1b\x01")
            for chunk_index in range(emitted // chunk_size, len(offsets)):
                start = chunk_index * chunk_size
                count = min(chunk_size, n_variants - start)
                compressed = os.pread(
                    source_fd,
                    int(sizes[chunk_index]),
                    int(offsets[chunk_index]),
                )
                decoded = decoder.decompress(
                    compressed,
                    max_output_size=count * n_samples,
                )
                dosage = np.frombuffer(decoded, dtype=np.uint8).reshape(count, n_samples)
                dosage_t = torch.from_numpy(dosage)
                dosage_device[:count, :n_samples].copy_(dosage_t, non_blocking=True)
                if padded_samples != n_samples:
                    dosage_device[:count, n_samples:].fill_(255)
                packed_device = pack_dosage(dosage_device)
                packed_host[:count].copy_(packed_device[:count], non_blocking=True)
                torch.cuda.current_stream(device).synchronize()
                handle.write(packed_host[:count].numpy().tobytes(order="C"))
                emitted += count
    finally:
        os.close(source_fd)
    encode_seconds = time.perf_counter() - encode_started
    if emitted != n_variants or bed_path.stat().st_size != expected_bed_bytes:
        raise RuntimeError(
            f"incomplete BED: emitted={emitted}/{n_variants}, "
            f"bytes={bed_path.stat().st_size}/{expected_bed_bytes}"
        )

    metadata_cache_path = None
    if args.metadata_cache_dir is not None:
        from torchgwas.bed import write_plink_bim_cache

        prep = np.load(args.prep, allow_pickle=False)
        metadata_cache_path = write_plink_bim_cache(
            args.metadata_cache_dir,
            bim_path,
            chromosomes=prep["chrom"],
            marker_ids=prep["snp"],
            positions=prep["pos"],
            other_alleles=prep["a2"],
            effect_alleles=prep["a1"],
        )
        prep.close()

    result = {
        "source_store": str(args.store),
        "source_project_root": str(args.project_root),
        "source_prep": str(args.prep),
        "source_sample_order": str(args.sample_order),
        "metadata_cache_path": (
            None if metadata_cache_path is None else str(metadata_cache_path)
        ),
        "output_prefix": str(output),
        "n_samples": n_samples,
        "n_variants": n_variants,
        "bed_bytes": bed_path.stat().st_size,
        "packing_device": str(device),
        "resumed_at_variant": resume_start,
        "metadata_seconds": metadata_seconds,
        "bed_encode_seconds": encode_seconds,
        "total_seconds": time.perf_counter() - started,
        "hard_call_rule": "dosage_u8 0..63 -> 0; 64..191 -> 1; 192..255 -> 2",
        "bim_a2": "BGEN allele 2 (alternative allele; prep a1)",
    }
    output.with_suffix(".conversion.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
