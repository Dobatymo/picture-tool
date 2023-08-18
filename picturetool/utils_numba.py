from typing import Optional, Union

import numpy as np
from numba import jit, optional
from numba import types as t
from numba_progress import ProgressBar as NumbaProgressBar  # pip install numba-progress
from numba_progress.progress import progressbar_type


@jit(
    [
        t.int64[:, ::1](t.float32[:, ::1], t.float32, optional(progressbar_type)),
        t.int64[:, ::1](t.float32[:, ::], t.float32, optional(progressbar_type)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    parallel=False,
    fastmath=True,
)
def l2squared_duplicates_numba(arr: np.ndarray, threshold: float, progress: Optional[NumbaProgressBar]) -> np.ndarray:
    coords = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            diffs = arr[i] - arr[j]
            norm = np.sum(diffs * diffs)
            if norm <= threshold:
                coords.append([i, j])
        if progress is not None:
            progress.update(1)
    return np.array(coords)


@jit(
    [
        t.int64[:, ::1](t.float32[:, ::1], t.int64, optional(progressbar_type)),
        t.int64[:, ::1](t.float32[:, ::], t.int64, optional(progressbar_type)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    parallel=False,
    fastmath=True,
)
def l2squared_duplicates_topk_numba(arr: np.ndarray, topk: int, progress: Optional[NumbaProgressBar]) -> np.ndarray:
    res = np.empty((arr.shape[0], topk), dtype=np.int64)
    for i in range(arr.shape[0]):
        diffs = arr[None, i, :] - arr
        dists = np.sum(diffs * diffs, axis=-1)
        res[i, :] = np.argpartition(dists, topk)[:topk]
        if progress is not None:
            progress.update(1)
    return res


def numba_duplicates_threshold_pairs(
    metric: str, arr: np.ndarray, threshold: Union[int, float], verbose: bool = False
) -> np.ndarray:
    if metric == "l2-squared":
        if verbose:
            with NumbaProgressBar(total=arr.shape[0]) as progress:
                out = l2squared_duplicates_numba(arr, threshold, progress)
        else:
            out = l2squared_duplicates_numba(arr, threshold, None)
        return out
    else:
        raise ValueError(f"Invalid metric: {metric}")


def numba_duplicates_topk_pairs(metric: str, arr: np.ndarray, topk: int, verbose: bool = False) -> np.ndarray:
    if metric == "l2-squared":
        if verbose:
            with NumbaProgressBar(total=arr.shape[0]) as progress:
                out = l2squared_duplicates_topk_numba(arr, topk, progress)
        else:
            out = l2squared_duplicates_topk_numba(arr, topk, None)
        return out
    else:
        raise ValueError(f"Invalid metric: {metric}")
