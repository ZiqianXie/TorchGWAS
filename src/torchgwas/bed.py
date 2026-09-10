from __future__ import annotations

import os
import queue
import hashlib
import json
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .streaming import OrderedChunkLoader


_BED_MAGIC = b"\x6c\x1b\x01"
_BIM_CACHE_SCHEMA = 1


def _bim_cache_path(cache_dir: str | Path, bim_path: Path) -> Path:
    stat = bim_path.stat()
    identity = json.dumps(
        {
            "schema": _BIM_CACHE_SCHEMA,
            "path": str(bim_path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{bim_path.stem}_{digest}.bim.cache"


def write_plink_bim_cache(
    cache_dir: str | Path,
    bim_path: str | Path,
    *,
    chromosomes,
    marker_ids,
    positions,
    other_alleles,
    effect_alleles,
) -> Path:
    """Write a validated binary BIM cache for repeated large BED analyses."""

    bim_path = Path(bim_path)
    cache_path = _bim_cache_path(cache_dir, bim_path)
    arrays = {
        "chromosomes": np.asarray(chromosomes, dtype=str),
        "marker_ids": np.asarray(marker_ids, dtype=str),
        "positions": np.asarray(positions, dtype=np.int64),
        "other_alleles": np.asarray(other_alleles, dtype=str),
        "effect_alleles": np.asarray(effect_alleles, dtype=str),
    }
    lengths = {value.shape[0] for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("BIM cache arrays have inconsistent lengths")
    if cache_path.is_dir():
        return cache_path
    stat = bim_path.stat()
    directory = cache_path.parent
    with tempfile.TemporaryDirectory(
        prefix=f".{cache_path.name}.",
        dir=directory,
    ) as temporary_name:
        temporary = Path(temporary_name)
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": _BIM_CACHE_SCHEMA,
                    "bim_path": str(bim_path.resolve()),
                    "bim_size": int(stat.st_size),
                    "bim_mtime_ns": int(stat.st_mtime_ns),
                    "n_variants": next(iter(lengths)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.replace(cache_path)
    return cache_path


def _load_plink_bim_cache(cache_dir: str | Path, bim_path: Path):
    cache_path = _bim_cache_path(cache_dir, bim_path)
    if not cache_path.is_dir():
        return None, cache_path
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.is_file():
        return None, cache_path
    manifest = json.loads(manifest_path.read_text())
    stat = bim_path.stat()
    valid = (
        int(manifest.get("schema", -1)) == _BIM_CACHE_SCHEMA
        and str(manifest.get("bim_path")) == str(bim_path.resolve())
        and int(manifest.get("bim_size", -1)) == stat.st_size
        and int(manifest.get("bim_mtime_ns", -1)) == stat.st_mtime_ns
    )
    if not valid:
        return None, cache_path
    arrays = tuple(
        np.load(cache_path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in (
            "chromosomes",
            "marker_ids",
            "positions",
            "other_alleles",
            "effect_alleles",
        )
    )
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError(f"BIM cache arrays have inconsistent lengths: {cache_path}")
    if arrays[0].shape[0] != int(manifest["n_variants"]):
        raise ValueError(f"BIM cache length does not match manifest: {cache_path}")
    return arrays, cache_path


def _make_a2_dosage_lut() -> np.ndarray:
    """Decode PLINK 1 two-bit calls as BIM-column-6 (A2) dosage."""

    # PLINK codes, least-significant pair first:
    # 00=A2/A2, 01=missing, 10=A1/A2, 11=A1/A1.
    code_to_a2 = np.asarray([2.0, np.nan, 1.0, 0.0], dtype=np.float32)
    lut = np.empty((256, 4), dtype=np.float32)
    for byte in range(256):
        for sample_offset in range(4):
            lut[byte, sample_offset] = code_to_a2[(byte >> (2 * sample_offset)) & 0b11]
    return lut


_A2_DOSAGE_LUT = _make_a2_dosage_lut()


def resolve_plink_triplet(
    genotype_path: str | Path,
    bim: str | Path | None = None,
    fam: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    path = Path(genotype_path)
    if path.suffix.lower() in {".bed", ".bim", ".fam"}:
        prefix = Path(str(path)[: -len(path.suffix)])
    else:
        prefix = path
    bed_path = path if path.suffix.lower() == ".bed" else Path(f"{prefix}.bed")
    bim_path = Path(bim) if bim is not None else Path(f"{prefix}.bim")
    fam_path = Path(fam) if fam is not None else Path(f"{prefix}.fam")
    return bed_path, bim_path, fam_path


class PlinkBedGenotype:
    """Bounded-memory, variant-major PLINK BED reader.

    The on-disk format is already variant-major, so each worker receives one
    contiguous variant range and performs one positional read. Decoded chunks
    are returned in the package's public sample-by-variant orientation.
    """

    ndim = 2
    preferred_gpu_chunk_size = 5000

    def __init__(
        self,
        genotype_path: str | Path,
        bim: str | Path | None = None,
        fam: str | Path | None = None,
        reader_workers: int = 4,
        prefetch_chunks: int = 4,
        metadata_cache_dir: str | Path | None = None,
    ) -> None:
        self.bed_path, self.bim_path, self.fam_path = resolve_plink_triplet(genotype_path, bim=bim, fam=fam)
        for path in (self.bed_path, self.bim_path, self.fam_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        fam_table = pd.read_csv(
            self.fam_path,
            sep=r"\s+",
            header=None,
            dtype=str,
            usecols=[0, 1],
            memory_map=True,
        )
        cached_bim = None
        self.metadata_cache_path = None
        if metadata_cache_dir is not None:
            cached_bim, self.metadata_cache_path = _load_plink_bim_cache(
                metadata_cache_dir,
                self.bim_path,
            )
        bim_table = None
        if cached_bim is None:
            bim_table = pd.read_csv(
                self.bim_path,
                sep=r"\s+",
                header=None,
                usecols=[0, 1, 3, 4, 5],
                dtype={0: str, 1: str, 3: np.int64, 4: str, 5: str},
                memory_map=True,
            )
        if fam_table.shape[1] != 2:
            raise ValueError(f"invalid FAM file (expected at least 2 columns): {self.fam_path}")
        if bim_table is not None and bim_table.shape[1] != 5:
            raise ValueError(f"invalid BIM file (expected at least 6 columns): {self.bim_path}")

        self._stored_family_ids = fam_table.iloc[:, 0].to_numpy(dtype=object)
        self._stored_sample_ids = fam_table.iloc[:, 1].to_numpy(dtype=object)
        self.family_ids = self._stored_family_ids
        self.sample_ids = self._stored_sample_ids
        if cached_bim is None:
            self.chromosomes = bim_table.iloc[:, 0].to_numpy(dtype=object)
            self.marker_ids = bim_table.iloc[:, 1].to_numpy(dtype=object)
            self.positions = bim_table[3].to_numpy(dtype=np.int64)
            self.other_alleles = bim_table[4].to_numpy(dtype=object)  # A1
            self.effect_alleles = bim_table[5].to_numpy(dtype=object)  # A2 dosage
            if metadata_cache_dir is not None:
                self.metadata_cache_path = write_plink_bim_cache(
                    metadata_cache_dir,
                    self.bim_path,
                    chromosomes=self.chromosomes,
                    marker_ids=self.marker_ids,
                    positions=self.positions,
                    other_alleles=self.other_alleles,
                    effect_alleles=self.effect_alleles,
                )
        else:
            (
                self.chromosomes,
                self.marker_ids,
                self.positions,
                self.other_alleles,
                self.effect_alleles,
            ) = cached_bim
        self.reader_workers = int(reader_workers)
        self.prefetch_chunks = int(prefetch_chunks)
        if self.reader_workers <= 0 or self.prefetch_chunks <= 0:
            raise ValueError("reader_workers and prefetch_chunks must be positive")

        self._stored_n_samples = int(self._stored_sample_ids.size)
        self._sample_indices: np.ndarray | None = None
        self._n_samples = self._stored_n_samples
        self._n_markers = int(self.marker_ids.size)
        self._bytes_per_variant = (self._stored_n_samples + 3) // 4
        expected_size = 3 + self._n_markers * self._bytes_per_variant
        actual_size = self.bed_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"BED size mismatch for {self.bed_path}: expected {expected_size} bytes "
                f"for {self._n_samples} samples and {self._n_markers} variants, got {actual_size}"
            )
        with self.bed_path.open("rb") as handle:
            magic = handle.read(3)
        if magic != _BED_MAGIC:
            raise ValueError(
                f"unsupported BED header {magic!r}; TorchGWAS requires PLINK 1 variant-major BED"
            )
        self._fd = os.open(self.bed_path, os.O_RDONLY)

    def __del__(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._n_samples, self._n_markers

    @property
    def genotype(self) -> "PlinkBedGenotype":
        return self

    @property
    def variant_metadata(self) -> dict[str, np.ndarray]:
        return {
            "chromosome": self.chromosomes,
            "position": self.positions,
            "effect_allele": self.effect_alleles,
            "other_allele": self.other_alleles,
        }

    def select_samples(self, sample_ids) -> "PlinkBedGenotype":
        """Restrict and reorder samples by FAM IID without rewriting the BED."""

        requested = np.asarray([str(value) for value in sample_ids], dtype=object)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("sample_ids must be a non-empty one-dimensional sequence")
        if np.unique(requested).size != requested.size:
            raise ValueError("requested sample_ids contain duplicate IID values")
        stored = np.asarray([str(value) for value in self._stored_sample_ids], dtype=object)
        if np.unique(stored).size != stored.size:
            raise ValueError("BED FAM contains duplicate IID values; select samples by FID+IID")
        lookup = {sample_id: index for index, sample_id in enumerate(stored.tolist())}
        missing = [sample_id for sample_id in requested.tolist() if sample_id not in lookup]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"{len(missing)} requested sample IDs are absent from {self.fam_path} ({preview})"
            )
        indices = np.asarray([lookup[sample_id] for sample_id in requested], dtype=np.int64)
        self._sample_indices = indices
        self.sample_ids = stored[indices]
        self.family_ids = self._stored_family_ids[indices]
        self._n_samples = int(indices.size)
        return self

    def _read_exact(self, offset: int, length: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while received < length:
            block = os.pread(self._fd, length - received, offset + received)
            if not block:
                raise OSError(f"unexpected EOF in {self.bed_path} at byte {offset + received}")
            chunks.append(block)
            received += len(block)
        return b"".join(chunks)

    def read_chunk(self, start: int, end: int, dtype: np.dtype = np.float32) -> np.ndarray:
        if not (0 <= start <= end <= self._n_markers):
            raise IndexError(f"invalid marker range [{start}, {end}) for {self._n_markers} variants")
        count = end - start
        raw = self._read_exact(3 + start * self._bytes_per_variant, count * self._bytes_per_variant)
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(count, self._bytes_per_variant)
        decoded = _A2_DOSAGE_LUT[packed].reshape(count, self._bytes_per_variant * 4)[
            :, : self._stored_n_samples
        ]
        if self._sample_indices is not None:
            decoded = decoded[:, self._sample_indices]
        return np.asarray(decoded.T, dtype=dtype, order="C")

    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("PLINK BED access requires genotype[samples, variants]")
        sample_key, marker_key = key
        if not isinstance(marker_key, slice):
            raise IndexError("PLINK BED marker access must be a contiguous slice")
        start, end, step = marker_key.indices(self._n_markers)
        if step != 1:
            raise IndexError("PLINK BED marker slices must have step=1")
        chunk = self.read_chunk(start, end, np.float32)
        return chunk[sample_key]

    def iter_chunks(
        self,
        chunk_size: int,
        dtype: np.dtype = np.float64,
        prefetch_chunks: int | None = None,
        reader_workers: int | None = None,
    ) -> OrderedChunkLoader:
        return OrderedChunkLoader(
            self._n_markers,
            self.read_chunk,
            chunk_size=chunk_size,
            dtype=dtype,
            prefetch_chunks=self.prefetch_chunks if prefetch_chunks is None else prefetch_chunks,
            reader_workers=self.reader_workers if reader_workers is None else reader_workers,
        )

    def iter_packed_chunks(
        self,
        chunk_size: int | None = None,
        reader_workers: int | None = None,
        depth: int | None = None,
    ) -> "PinnedPackedBedLoader":
        workers = self.reader_workers if reader_workers is None else int(reader_workers)
        chunk = self.preferred_gpu_chunk_size if chunk_size is None else int(chunk_size)
        ring_depth = max(3, workers + 2) if depth is None else int(depth)
        return PinnedPackedBedLoader(self, chunk, workers, ring_depth)


class PinnedPackedBedLoader:
    """Ordered BED reads into pinned buffers without CPU genotype decoding."""

    def __init__(
        self,
        genotype: PlinkBedGenotype,
        chunk_size: int,
        reader_workers: int,
        depth: int,
    ) -> None:
        import torch

        if chunk_size <= 0 or reader_workers <= 0 or depth <= 0:
            raise ValueError("chunk_size, reader_workers, and depth must be positive")
        self.genotype = genotype
        self.chunk_size = int(chunk_size)
        self.reader_workers = int(reader_workers)
        self.depth = int(depth)
        self.buffers = [
            torch.empty(
                (self.chunk_size, genotype._bytes_per_variant),
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(self.depth)
        ]
        self.free: queue.Queue[int] = queue.Queue()
        for index in range(self.depth):
            self.free.put(index)
        self._pool = ThreadPoolExecutor(
            max_workers=self.reader_workers,
            thread_name_prefix="torchgwas-bed-packed",
        )

    def _fill(self, buffer_index: int, start: int, end: int):
        count = end - start
        byte_count = count * self.genotype._bytes_per_variant
        target = memoryview(self.buffers[buffer_index].numpy()).cast("B")[:byte_count]
        offset = 3 + start * self.genotype._bytes_per_variant
        received = 0
        while received < byte_count:
            amount = os.preadv(
                self.genotype._fd,
                [target[received:]],
                offset + received,
            )
            if amount == 0:
                raise OSError(
                    f"unexpected EOF in {self.genotype.bed_path} at byte {offset + received}"
                )
            received += amount
        return buffer_index, start, end

    def __iter__(self):
        bounds = [
            (start, min(self.genotype._n_markers, start + self.chunk_size))
            for start in range(0, self.genotype._n_markers, self.chunk_size)
        ]
        pending: dict[int, Future] = {}
        submit_index = 0

        def submit_one() -> None:
            nonlocal submit_index
            buffer_index = self.free.get()
            start, end = bounds[submit_index]
            pending[submit_index] = self._pool.submit(
                self._fill,
                buffer_index,
                start,
                end,
            )
            submit_index += 1

        while submit_index < min(len(bounds), self.depth):
            submit_one()
        try:
            for output_index in range(len(bounds)):
                buffer_index, start, end = pending.pop(output_index).result()
                yield buffer_index, self.buffers[buffer_index], start, end
                if submit_index < len(bounds):
                    submit_one()
        finally:
            for future in pending.values():
                future.cancel()

    def release(self, buffer_index: int) -> None:
        self.free.put(buffer_index)

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
