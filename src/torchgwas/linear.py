from __future__ import annotations

import math
from collections.abc import Iterator
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
from scipy import special

from .kernels import linear_chunk_kernel
from .preprocess import residualize_and_standardize
from .streaming import ChunkedGenotype
from .utils import choose_device, chunk_bounds


def _resolve_compute_dtypes(compute_dtype: str) -> tuple[np.dtype, torch.dtype]:
    if compute_dtype == "float32":
        return np.float32, torch.float32
    if compute_dtype == "float64":
        return np.float64, torch.float64
    raise ValueError(f"unsupported compute dtype: {compute_dtype}")


def _two_sided_t_pvalue(t_stat: np.ndarray, df: int) -> np.ndarray:
    return 2.0 * special.stdtr(df, -np.abs(t_stat))


def _unpack_plink_a2_float(packed: torch.Tensor, n_samples: int) -> torch.Tensor:
    calls = torch.stack(
        (
            packed & 3,
            (packed >> 2) & 3,
            (packed >> 4) & 3,
            (packed >> 6) & 3,
        ),
        dim=2,
    ).reshape(packed.shape[0], -1)[:, :n_samples]
    dosage = (2 - ((calls + 1) >> 1)).to(torch.float32)
    return torch.where(calls == 1, torch.nan, dosage)


def _packed_bed_statistics(
    packed: torch.Tensor,
    design: torch.Tensor,
    phenotype_ss: torch.Tensor,
    n_samples: int,
    n_traits: int,
    df: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    genotype = _unpack_plink_a2_float(packed, n_samples)
    products = genotype @ design
    gy = products[:, :n_traits]
    gc = products[:, n_traits:]
    residual_ss = torch.sum(genotype * genotype, dim=1) - torch.sum(gc * gc, dim=1)
    valid = residual_ss > 1e-12
    status = torch.where(
        torch.isfinite(residual_ss),
        torch.where(
            valid,
            torch.zeros_like(residual_ss, dtype=torch.uint8),
            torch.full_like(residual_ss, 2, dtype=torch.uint8),
        ),
        torch.ones_like(residual_ss, dtype=torch.uint8),
    )
    safe_ss = torch.clamp(residual_ss, min=1e-12)
    beta = gy / safe_ss[:, None]
    explained_ss = gy * gy / safe_ss[:, None]
    residual_y_ss = torch.clamp(phenotype_ss[None, :] - explained_ss, min=1e-12)
    standard_error = torch.sqrt(residual_y_ss / float(df) / safe_ss[:, None])
    t_stat = beta / standard_error
    beta = torch.where(valid[:, None], beta, torch.zeros_like(beta))
    t_stat = torch.where(valid[:, None], t_stat, torch.zeros_like(t_stat))
    return beta, t_stat, status


_COMPILED_PACKED_BED_STATISTICS = None
if hasattr(torch, "compile"):
    try:
        _COMPILED_PACKED_BED_STATISTICS = torch.compile(
            _packed_bed_statistics,
            fullgraph=True,
            dynamic=False,
        )
    except Exception:
        pass


def _packed_bed_cuda_iterator(
    genotype: ChunkedGenotype,
    pheno_proc: np.ndarray,
    q_matrix: np.ndarray | None,
    chunk_size: int,
    torch_device: torch.device,
    reader_workers: int | None,
    compute_p_values: bool,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray | None]]:
    """Overlap packed BED reads, H2D, fused GPU decode, OLS, and result copies."""

    torch.cuda.set_device(torch_device)
    n_samples, n_traits = pheno_proc.shape
    covariate_rank = 0 if q_matrix is None else q_matrix.shape[1]
    df = n_samples - covariate_rank - 2
    workers = int(reader_workers or getattr(genotype, "reader_workers", 1))
    depth = max(3, workers + 2)
    bytes_per_variant = int(getattr(genotype, "_bytes_per_variant"))

    phenotype_t = torch.as_tensor(pheno_proc, dtype=torch.float32, device=torch_device)
    intercept_t = torch.full(
        (n_samples, 1),
        1.0 / math.sqrt(n_samples),
        dtype=torch.float32,
        device=torch_device,
    )
    if q_matrix is None:
        covariate_t = intercept_t
    else:
        q_t = torch.as_tensor(q_matrix, dtype=torch.float32, device=torch_device)
        covariate_t = torch.cat((intercept_t, q_t), dim=1)
    design_t = torch.cat((phenotype_t, covariate_t), dim=1)
    phenotype_ss_t = torch.sum(phenotype_t * phenotype_t, dim=0)

    packed_device = [
        torch.empty((chunk_size, bytes_per_variant), dtype=torch.uint8, device=torch_device)
        for _ in range(4)
    ]
    packed_device[0].zero_()
    compute_done = [torch.cuda.Event() for _ in packed_device]
    copy_done = [torch.cuda.Event() for _ in packed_device]
    copy_stream = torch.cuda.Stream(device=torch_device)
    result_stream = torch.cuda.Stream(device=torch_device)
    # Keep several result slots so CPU Student-t tails can run in parallel while
    # the next BED chunks are read and scanned on the GPU.
    result_depth = max(2, min(8, workers))
    beta_host = [
        torch.empty((chunk_size, n_traits), dtype=torch.float32, pin_memory=True)
        for _ in range(result_depth)
    ]
    t_host = [
        torch.empty((chunk_size, n_traits), dtype=torch.float32, pin_memory=True)
        for _ in range(result_depth)
    ]
    status_host = [
        torch.empty(chunk_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(result_depth)
    ]
    result_done = [torch.cuda.Event() for _ in range(result_depth)]

    statistics = _COMPILED_PACKED_BED_STATISTICS or _packed_bed_statistics
    try:
        warm_beta, warm_t, warm_status = statistics(
            packed_device[0],
            design_t,
            phenotype_ss_t,
            n_samples,
            n_traits,
            df,
        )
        (warm_beta.sum() + warm_t.sum() + warm_status.sum()).item()
    except Exception:
        statistics = _packed_bed_statistics
    torch.cuda.synchronize(torch_device)

    loader = genotype.iter_packed_chunks(
        chunk_size=chunk_size,
        reader_workers=workers,
        depth=depth,
    )
    pvalue_pool = ThreadPoolExecutor(
        max_workers=max(1, min(8, workers)),
        thread_name_prefix="torchgwas-pvalue",
    )
    pending: deque[Future] = deque()

    def finish_result(result_slot: int, start: int, end: int):
        result_done[result_slot].synchronize()
        count = end - start
        status_chunk = status_host[result_slot][:count].numpy()
        missing = np.flatnonzero(status_chunk == 1)
        if missing.size:
            marker = start + int(missing[0])
            raise ValueError(
                f"genotype contains missing/non-finite values at marker {marker}; "
                "v0.1 requires complete matrices"
            )
        invariant = np.flatnonzero(status_chunk == 2)
        if invariant.size:
            marker = start + int(invariant[0])
            raise ValueError(
                f"out-of-core genotype contains a zero-variance variant at marker {marker}; "
                "filter invariant variants before running TorchGWAS"
            )
        beta_chunk = beta_host[result_slot][:count].numpy()
        t_chunk = t_host[result_slot][:count].numpy()
        p_chunk = _two_sided_t_pvalue(t_chunk, df=df) if compute_p_values else None
        return start, end, beta_chunk, t_chunk, p_chunk

    try:
        for iteration, (buffer_index, host_packed, start, end) in enumerate(loader):
            if len(pending) >= result_depth:
                yield pending.popleft().result()

            count = end - start
            device_slot = iteration % len(packed_device)
            if iteration >= len(packed_device):
                copy_stream.wait_event(compute_done[device_slot])
            with torch.cuda.stream(copy_stream):
                packed_device[device_slot].copy_(host_packed, non_blocking=True)
                copy_done[device_slot].record(copy_stream)

            compute_stream = torch.cuda.current_stream(torch_device)
            compute_stream.wait_event(copy_done[device_slot])
            beta_t, t_t, status_t = statistics(
                packed_device[device_slot],
                design_t,
                phenotype_ss_t,
                n_samples,
                n_traits,
                df,
            )
            compute_done[device_slot].record(compute_stream)

            result_slot = iteration % result_depth
            with torch.cuda.stream(result_stream):
                result_stream.wait_event(compute_done[device_slot])
                beta_host[result_slot][:count].copy_(beta_t[:count], non_blocking=True)
                t_host[result_slot][:count].copy_(t_t[:count], non_blocking=True)
                status_host[result_slot][:count].copy_(status_t[:count], non_blocking=True)
                beta_t.record_stream(result_stream)
                t_t.record_stream(result_stream)
                status_t.record_stream(result_stream)
                result_done[result_slot].record(result_stream)

            # Free the packed host buffer as soon as DMA no longer reads it.
            copy_done[device_slot].synchronize()
            loader.release(buffer_index)
            pending.append(pvalue_pool.submit(finish_result, result_slot, start, end))

        while pending:
            yield pending.popleft().result()
    finally:
        pvalue_pool.shutdown(wait=True, cancel_futures=True)
        loader.close()


def linear_scan(
    genotype: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    chunk_size: int | None = None,
    device: str = "auto",
    compute_dtype: str = "float64",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    pheno_proc, q_matrix = residualize_and_standardize(phenotype, covariates)
    n_samples = pheno_proc.shape[0]
    n_markers = genotype.shape[1]
    n_traits = pheno_proc.shape[1]
    torch_device = choose_device(device)
    np_dtype, torch_dtype = _resolve_compute_dtypes(compute_dtype)
    pheno_t = torch.as_tensor(pheno_proc, dtype=torch_dtype, device=torch_device)
    q_t = None if q_matrix is None else torch.as_tensor(q_matrix, dtype=torch_dtype, device=torch_device)
    covariate_rank = 0 if q_matrix is None else q_matrix.shape[1]
    df = n_samples - covariate_rank - 2
    if df <= 0:
        raise ValueError(f"non-positive residual degrees of freedom: N={n_samples}, covariate_rank={covariate_rank}")

    beta = np.empty((n_markers, n_traits), dtype=np_dtype)
    t_stat = np.empty((n_markers, n_traits), dtype=np_dtype)
    p_value = np.empty((n_markers, n_traits), dtype=np.float64)

    for start, end in chunk_bounds(n_markers, chunk_size):
        geno_chunk = np.asarray(genotype[:, start:end], dtype=np_dtype)
        geno_t = torch.as_tensor(geno_chunk, dtype=torch_dtype, device=torch_device)
        beta_t, t_chunk_t = linear_chunk_kernel(geno_t, pheno_t, q_t, df)
        beta_chunk = beta_t.cpu().numpy()
        t_chunk = t_chunk_t.cpu().numpy()
        p_chunk = _two_sided_t_pvalue(t_chunk, df=df)
        beta[start:end] = beta_chunk
        t_stat[start:end] = t_chunk
        p_value[start:end] = p_chunk
    return beta, t_stat, p_value, q_matrix


def linear_scan_streaming(
    genotype: ChunkedGenotype,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    chunk_size: int | None = None,
    device: str = "auto",
    compute_dtype: str = "float32",
    reader_workers: int | None = None,
    prefetch_chunks: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    requested_device = choose_device(device)
    if (
        requested_device.type == "cuda"
        and compute_dtype == "float32"
        and hasattr(genotype, "iter_packed_chunks")
    ):
        chunk_iterator, q_matrix = linear_scan_streaming_chunks(
            genotype,
            phenotype,
            covariates,
            chunk_size=chunk_size,
            device=str(requested_device),
            compute_dtype=compute_dtype,
            reader_workers=reader_workers,
            prefetch_chunks=prefetch_chunks,
        )
        n_markers = genotype.shape[1]
        n_traits = phenotype.shape[1]
        beta = np.empty((n_markers, n_traits), dtype=np.float32)
        t_stat = np.empty((n_markers, n_traits), dtype=np.float32)
        p_value = np.empty((n_markers, n_traits), dtype=np.float64)
        for start, end, beta_chunk, t_chunk, p_chunk in chunk_iterator:
            beta[start:end] = beta_chunk
            t_stat[start:end] = t_chunk
            p_value[start:end] = p_chunk
        return beta, t_stat, p_value, q_matrix

    pheno_proc, q_matrix = residualize_and_standardize(phenotype, covariates)
    n_samples = pheno_proc.shape[0]
    n_markers = genotype.shape[1]
    n_traits = pheno_proc.shape[1]
    torch_device = requested_device
    chunk = (
        chunk_size
        or (
            getattr(genotype, "preferred_gpu_chunk_size", None)
            if torch_device.type == "cuda"
            else None
        )
        or getattr(genotype, "preferred_chunk_size", None)
        or min(n_markers, 4096)
        or 1
    )
    covariate_rank = 0 if q_matrix is None else q_matrix.shape[1]
    df = n_samples - covariate_rank - 2
    if df <= 0:
        raise ValueError(f"non-positive residual degrees of freedom: N={n_samples}, covariate_rank={covariate_rank}")

    np_dtype, torch_dtype = _resolve_compute_dtypes(compute_dtype)
    pheno_t = torch.as_tensor(pheno_proc, dtype=torch_dtype, device=torch_device)
    q_t = None if q_matrix is None else torch.as_tensor(q_matrix, dtype=torch_dtype, device=torch_device)

    beta = np.empty((n_markers, n_traits), dtype=np_dtype)
    t_stat = np.empty((n_markers, n_traits), dtype=np_dtype)
    p_value = np.empty((n_markers, n_traits), dtype=np.float64)

    for start, end, geno_chunk in genotype.iter_chunks(
        chunk_size=chunk,
        dtype=np_dtype,
        prefetch_chunks=prefetch_chunks,
        reader_workers=reader_workers,
    ):
        geno_t = torch.as_tensor(geno_chunk, dtype=torch_dtype, device=torch_device)
        beta_t, t_chunk_t = linear_chunk_kernel(geno_t, pheno_t, q_t, df)
        beta_chunk = beta_t.cpu().numpy()
        t_chunk = t_chunk_t.cpu().numpy()
        p_chunk = _two_sided_t_pvalue(t_chunk, df=df)
        beta[start:end] = beta_chunk
        t_stat[start:end] = t_chunk
        p_value[start:end] = p_chunk
    return beta, t_stat, p_value, q_matrix


def linear_scan_streaming_chunks(
    genotype: ChunkedGenotype,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    chunk_size: int | None = None,
    device: str = "auto",
    compute_dtype: str = "float32",
    reader_workers: int | None = None,
    prefetch_chunks: int | None = None,
    compute_p_values: bool = True,
) -> tuple[
    Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray | None]],
    np.ndarray | None,
]:
    pheno_proc, q_matrix = residualize_and_standardize(phenotype, covariates)
    n_samples = pheno_proc.shape[0]
    n_markers = genotype.shape[1]
    torch_device = choose_device(device)
    chunk = (
        chunk_size
        or (
            getattr(genotype, "preferred_gpu_chunk_size", None)
            if torch_device.type == "cuda"
            else None
        )
        or getattr(genotype, "preferred_chunk_size", None)
        or min(n_markers, 4096)
        or 1
    )
    covariate_rank = 0 if q_matrix is None else q_matrix.shape[1]
    df = n_samples - covariate_rank - 2
    if df <= 0:
        raise ValueError(f"non-positive residual degrees of freedom: N={n_samples}, covariate_rank={covariate_rank}")
    np_dtype, torch_dtype = _resolve_compute_dtypes(compute_dtype)

    if (
        torch_device.type == "cuda"
        and compute_dtype == "float32"
        and hasattr(genotype, "iter_packed_chunks")
    ):
        return (
            _packed_bed_cuda_iterator(
                genotype,
                pheno_proc,
                q_matrix,
                chunk,
                torch_device,
                reader_workers,
                compute_p_values,
            ),
            q_matrix,
        )

    pheno_t = torch.as_tensor(pheno_proc, dtype=torch_dtype, device=torch_device)
    q_t = None if q_matrix is None else torch.as_tensor(q_matrix, dtype=torch_dtype, device=torch_device)

    def _iterator() -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray | None]]:
        for start, end, geno_chunk in genotype.iter_chunks(
            chunk_size=chunk,
            dtype=np_dtype,
            prefetch_chunks=prefetch_chunks,
            reader_workers=reader_workers,
        ):
            geno_t = torch.as_tensor(geno_chunk, dtype=torch_dtype, device=torch_device)
            beta_t, t_chunk_t = linear_chunk_kernel(geno_t, pheno_t, q_t, df)
            beta_chunk = beta_t.cpu().numpy()
            t_chunk = t_chunk_t.cpu().numpy()
            p_chunk = _two_sided_t_pvalue(t_chunk, df=df) if compute_p_values else None
            yield start, end, beta_chunk, t_chunk, p_chunk

    return _iterator(), q_matrix
