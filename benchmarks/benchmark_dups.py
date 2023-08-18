import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cpuinfo
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar
from genutility.file import StdoutFile
from genutility.rich import Progress
from genutility.time import MeasureTime
from genutility.typing import SizedIterable
from rich.progress import Progress as RichProgress

from picturetool.ml_utils import faiss_duplicates_threshold_pairs, faiss_duplicates_topk_pairs
from picturetool.utils import (
    hamming_duplicates_chunk,
    l2squared_duplicates_chunk,
    npmp_duplicates_threshold_pairs,
    npmp_duplicates_topk_pairs,
    npmt_duplicates_threshold_pairs,
    npmt_duplicates_topk_pairs,
)
from picturetool.utils_annoy import annoy_duplicates_topk, annoy_from_array
from picturetool.utils_numba import numba_duplicates_threshold_pairs


def matmul_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(a.shape, b.shape)
    ab = np.broadcast_to(a, shape)
    bb = np.broadcast_to(b, shape)
    np.matmul(ab, bb)  # uses multi-threaded blas libary
    return a


funcs = {
    "l2-dups": l2squared_duplicates_chunk,
    "binary-dups": hamming_duplicates_chunk,
    "matmul": matmul_chunk,
}
funcs_kwargs = {
    "l2-dups": {"threshold": 1.0},
    "binary-dups": {"axis": -1, "hamming_threshold": 1},
    "matmul": {},
}

THRESHOLD_L2 = 1.0
THRESHOLD_HAMMING = 2
TOP_K = 5


def task_numba(task: str, arr: np.ndarray) -> np.ndarray:
    if task == "l2-dups":
        return numba_duplicates_threshold_pairs("l2-squared", arr, THRESHOLD_L2)
    else:
        raise ValueError(f"Invalid task: {task}")


def task_python(task: str, arr: np.ndarray, progress: Progress) -> np.ndarray:
    threshold = 1.0

    arr = arr.tolist()

    if task == "l2-dups":
        coords = []
        for i in progress.track(range(len(arr))):
            for j in range(i + 1, len(arr)):
                diffs = [a - b for a, b in zip(arr[i], arr[j])]
                norm = sum(i * i for i in diffs)
                if norm <= threshold:
                    coords.append([i, j])
        return np.array(coords)
    else:
        raise ValueError(f"Invalid task: {task}")


def task_faiss(task: str, arr: np.ndarray, chunksize: int, progress: Progress) -> np.ndarray:
    if task == "l2-dups":
        return faiss_duplicates_threshold_pairs("l2-squared", arr, THRESHOLD_L2, chunksize, progress)
    elif task == "l2-top-k":
        return faiss_duplicates_topk_pairs("l2-squared", arr, TOP_K, 1000, progress)
    elif task == "binary-dups":
        return faiss_duplicates_threshold_pairs("hamming", arr, THRESHOLD_HAMMING, chunksize, progress)
    else:
        raise ValueError(f"Invalid task: {task}")


def task_annoy(task: str, arr: np.ndarray, progress: Progress) -> np.ndarray:
    if task == "l2-dups":
        index = annoy_from_array(arr, "euclidean")
        return annoy_duplicates_topk(index, TOP_K, progress)
    elif task == "binary-dups":
        index = annoy_from_array(arr, "hamming")
        return annoy_duplicates_topk(index, TOP_K, progress)
    else:
        raise ValueError(f"Invalid task: {task}")


def task_numpy(task: str, arr: np.ndarray):
    a_arr = arr[None, :, :]
    b_arr = arr[:, None, :]
    return funcs[task](a_arr, b_arr, **funcs_kwargs[task])


def task_npmt(
    task: str, arr: np.ndarray, chunksize: int, limit: Optional[int], progress: Progress
) -> SizedIterable[np.ndarray]:
    if task == "l2-dups":
        return npmt_duplicates_threshold_pairs("l2-squared", arr, THRESHOLD_L2, chunksize, progress)
    elif task == "l2-top-k":
        return npmt_duplicates_topk_pairs("l2-squared", arr, TOP_K, chunksize, progress)
    elif task == "binary-dups":
        return npmt_duplicates_threshold_pairs("hamming", arr, THRESHOLD_HAMMING, chunksize, progress)
    else:
        raise ValueError(f"Invalid task: {task}")


def task_npmp(
    task: str, arr: np.ndarray, chunksize: int, limit: Optional[int], progress: Progress
) -> SizedIterable[np.ndarray]:
    if task == "l2-dups":
        return npmp_duplicates_threshold_pairs("l2-squared", arr, THRESHOLD_L2, chunksize, progress)
    elif task == "l2-top-k":
        return npmp_duplicates_topk_pairs("l2-squared", arr, TOP_K, chunksize, progress)
    elif task == "binary-dups":
        return npmp_duplicates_threshold_pairs("hamming", arr, THRESHOLD_HAMMING, chunksize, progress)
    else:
        raise ValueError(f"Invalid task: {task}")


def task_dask(task: str, np_arr: np.ndarray, chunksize: int):
    if task == "l2-dups":
        arr = da.from_array(np_arr, chunks=(chunksize, -1))
        a = da.broadcast_to(
            arr[None, :, :], (arr.shape[0], arr.shape[0], arr.shape[1]), chunks=(chunksize, chunksize, -1)
        )
        b = da.broadcast_to(
            arr[:, None, :], (arr.shape[0], arr.shape[0], arr.shape[1]), chunks=(chunksize, chunksize, -1)
        )
        m = da.sum(da.power(a - b, 2), axis=-1)
        return da.argwhere(m < 1.0).compute()
    else:
        raise ValueError(f"Invalid task: {task}")


def main(
    outpath: Optional[Path],
    engine: str,
    task: str,
    dims: Tuple[int, int],
    chunksize: int,
    seed: Optional[int],
    limit: Optional[int],
) -> None:
    rng = np.random.default_rng(seed)

    cpu_name = cpuinfo.get_cpu_info()["brand_raw"]

    if task == "binary-dups":
        np_arr = rng.integers(0, 256, size=dims, dtype=np.uint8)
    else:
        np_arr = rng.uniform(0, 1, size=dims).astype(np.float32)

    now = datetime.now().isoformat()
    prefix = f"{now} [{cpu_name}] {engine} {task} dims={dims} chunksize={chunksize} limit={limit}"

    try:
        with MeasureTime() as stopwatch, RichProgress() as progress:
            p = Progress(progress)
            if engine == "python":
                out = task_python(task, np_arr, p)
            elif engine == "numba":
                out = task_numba(task, np_arr)
            elif engine == "numpy":
                out = task_numpy(task, np_arr)
            elif engine == "npmt":
                out = task_npmt(task, np_arr, chunksize, limit, p)
            elif engine == "npmp":
                out = task_npmp(task, np_arr, chunksize, limit, p)
            elif engine == "dask":
                with ProgressBar():
                    out = task_dask(task, np_arr, chunksize)
            elif engine == "faiss":
                out = task_faiss(task, np_arr, chunksize, p)
            elif engine == "annoy":
                out = task_annoy(task, np_arr, p)
            else:
                raise ValueError(engine)

            delta = stopwatch.get()

    except MemoryError as e:
        with StdoutFile(outpath, "at") as fw:
            fw.write(f"{prefix}: MemoryError {e}\n")
            sys.exit(1)
    else:
        with StdoutFile(outpath, "at") as fw:
            fw.write(f"{prefix}: {out.shape} in {delta}\n")


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    CHUNKSIZE = 1000

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--engine", choices=("python", "numba", "numpy", "npmt", "npmp", "dask", "faiss", "annoy"), required=True
    )
    parser.add_argument("--dims", type=int, nargs=2, required=True)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threadpool-limit", type=int, default=None)
    parser.add_argument("--task", choices=("l2-dups", "binary-dups", "matmul"))
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default=None,
        help="Write results to file. Otherwise they are written to stdout.",
    )
    args = parser.parse_args()

    main(args.out, args.engine, args.task, args.dims, args.chunksize, args.seed, args.threadpool_limit)
