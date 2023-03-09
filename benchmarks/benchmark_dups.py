import sys
from typing import Optional, Tuple

import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar
from genutility.iter import progress
from genutility.time import PrintStatementTime
from genutility.typing import SizedIterable

from picturetool import npmp
from picturetool.ml_utils import (
    annoy_duplicates_top_k,
    annoy_from_array,
    faiss_duplicates_threshold,
    faiss_duplicates_top_k,
    faiss_from_array,
    faiss_to_pairs,
)
from picturetool.utils import hamming_duplicates_chunk, l2_duplicates_chunk


def matmul_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(a.shape, b.shape)
    ab = np.broadcast_to(a, shape)
    bb = np.broadcast_to(b, shape)
    np.matmul(ab, bb)  # uses multi-threaded blas libary
    return a


funcs = {
    "l2-dups": l2_duplicates_chunk,
    "binary-dups": hamming_duplicates_chunk,
    "matmul": matmul_chunk,
}
funcs_kwargs = {
    "l2-dups": {},
    "binary-dups": {"hamming_threshold": 1},
    "matmul": {},
}


def task_faiss(task: str, arr: np.ndarray, chunksize: int) -> np.ndarray:
    if task == "l2-dups":
        index = faiss_from_array(arr, "l2-squared")
        pairs, dists = faiss_to_pairs(
            faiss_duplicates_threshold(index, batchsize=chunksize, threshold=1.0, verbose=True)
        )
        return pairs
    elif task == "l2-top-k":
        index = faiss_from_array(arr, "l2-squared")
        pairs, dists = faiss_to_pairs(faiss_duplicates_top_k(index, batchsize=1000, top_k=5, verbose=True))
        return pairs
    elif task == "binary-dups":
        index = faiss_from_array(arr, "hamming")
        pairs, dists = faiss_to_pairs(
            faiss_duplicates_threshold(index, batchsize=chunksize, threshold=1.0, verbose=True)
        )
        return pairs
    else:
        raise ValueError(f"Invalid task: {task}")


def task_annoy(task: str, arr: np.ndarray) -> np.ndarray:
    top_k = 5
    if task == "l2-dups":
        index = annoy_from_array(arr, "euclidean")
        return annoy_duplicates_top_k(index, top_k)

    elif task == "binary-dups":
        index = annoy_from_array(arr, "hamming")
        return annoy_duplicates_top_k(index, top_k)
    else:
        raise ValueError(f"Invalid task: {task}")


def task_numpy(task: str, arr: np.ndarray):
    a_arr = arr[None, :, :]
    b_arr = arr[:, None, :]
    return funcs[task](a_arr, b_arr, **funcs_kwargs[task])


def task_npmt(task: str, arr: np.ndarray, chunksize: int, limit: Optional[int]) -> SizedIterable[np.ndarray]:
    if len(arr.shape) != 2:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    npmp.THREADPOOL_LIMIT = limit

    a_arr = arr[None, :, :]
    b_arr = arr[:, None, :]

    return npmp.ChunkedParallel(
        funcs[task], a_arr, b_arr, (chunksize, chunksize), backend="threading", pass_coords=False, **funcs_kwargs[task]
    )


def task_npmp(task: str, sharr: npmp.SharedNdarray, chunksize: int, limit: Optional[int]) -> SizedIterable[np.ndarray]:
    if len(sharr.shape) != 2:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    npmp.THREADPOOL_LIMIT = limit

    a_arr = sharr.reshape((1, sharr.shape[0], sharr.shape[1]))
    b_arr = sharr.reshape((sharr.shape[0], 1, sharr.shape[1]))

    return npmp.ChunkedParallel(
        funcs[task],
        a_arr,
        b_arr,
        (chunksize, chunksize),
        backend="multiprocessing",
        pass_coords=False,
        **funcs_kwargs[task],
    )


def task_dask(task: str, np_arr: np.ndarray, chunksize: int):
    if task == "l2-dups":
        arr = da.from_array(np_arr, chunks=(chunksize, -1))
        a = da.broadcast_to(
            arr[None, :, :], (arr.shape[0], arr.shape[0], arr.shape[1]), chunks=(chunksize, chunksize, -1)
        )
        b = da.broadcast_to(
            arr[:, None, :], (arr.shape[0], arr.shape[0], arr.shape[1]), chunks=(chunksize, chunksize, -1)
        )
        m = da.sqrt(da.sum(da.power(a - b, 2), axis=-1))
        return da.argwhere(m < 1.0).compute()
    else:
        raise ValueError(f"Invalid task: {task}")


def main(
    engine: str, task: str, dims: Tuple[int, int], chunksize: int, seed: Optional[int], limit: Optional[int]
) -> None:
    rng = np.random.default_rng(seed)

    if task == "binary-dups":
        np_arr = rng.integers(0, 256, size=dims, dtype=np.uint8)
    else:
        np_arr = rng.uniform(0, 1, size=dims).astype(np.float32)

    try:
        with PrintStatementTime():
            if engine == "numpy":
                out = task_numpy(task, np_arr)
            elif engine == "npmt":
                out = np.concatenate(list(progress(task_npmt(task, np_arr, chunksize, limit))))
            elif engine == "npmp":
                out = np.concatenate(
                    list(progress(task_npmp(task, npmp.SharedNdarray.from_array(np_arr), chunksize, limit)))
                )
            elif engine == "dask":
                with ProgressBar():
                    out = task_dask(task, np_arr, chunksize)
            elif engine == "faiss":
                out = task_faiss(task, np_arr, chunksize)
            elif engine == "annoy":
                out = task_annoy(task, np_arr)
            else:
                raise ValueError(engine)
    except MemoryError as e:
        print("MemoryError", e)
        sys.exit(1)
    else:
        print(engine, out.shape)


if __name__ == "__main__":
    from argparse import ArgumentParser

    CHUNKSIZE = 1000

    parser = ArgumentParser()
    parser.add_argument("--engine", choices=("numpy", "npmt", "npmp", "dask", "faiss", "annoy"), required=True)
    parser.add_argument("--dims", type=int, nargs=2, required=True)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threadpool-limit", type=int, default=None)
    parser.add_argument("--task", choices=("l2-dups", "binary-dups"))
    args = parser.parse_args()

    main(args.engine, args.task, args.dims, args.chunksize, args.seed, args.threadpool_limit)
