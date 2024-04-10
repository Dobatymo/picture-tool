from enum import Enum
from typing import Iterable, Iterator, List, Optional, Tuple, Union, overload

import faiss
import numpy as np
from genutility.callbacks import Progress
from more_itertools import spy

from .utils import slice_idx

FaissIndexTypes = Union[faiss.IndexFlat, faiss.IndexBinary]


class FaissMetric(Enum):
    INNER_PRODUCT = 0
    L2 = 1
    L1 = 2
    Linf = 3
    Lp = 4
    Canberra = 5
    BrayCurtis = 6
    JensenShannon = 7
    Jaccard = 8


def _reconstruct_fixed(index: FaissIndexTypes, i0: int, ni: int) -> np.ndarray:
    """workaround for faiss bug https://github.com/facebookresearch/faiss/issues/2751"""

    if isinstance(index, faiss.IndexBinary):
        query = np.empty((ni, index.d), dtype=np.uint8)
        index.reconstruct_n(i0, ni, faiss.swig_ptr(query))
        return query[:, : index.d // 8]
    else:
        return index.reconstruct_n(i0, ni)  # fixme: a preallocated array could be used here


def faiss_from_array(
    arr: Union[np.ndarray, Iterable[np.ndarray]], norm: str
) -> Union[faiss.IndexFlatL2, faiss.IndexBinaryFlat]:
    if isinstance(arr, np.ndarray):
        arr = [arr]
    elif isinstance(arr, list):
        pass
    else:
        raise TypeError("arr must be np array ot iterable of np arrays")

    head, it = spy(arr)

    if not head:
        raise ValueError("arr iterable cannot be empty")

    if norm == "l2-squared":
        index = faiss.IndexFlatL2(head[0].shape[1])
        assert FaissMetric(index.metric_type).name == "L2"
    elif norm == "hamming":
        index = faiss.IndexBinaryFlat(head[0].shape[1] * 8)
        assert index.metric_type == 1  # why?
    else:
        raise ValueError(f"Invalid norm: {norm}")

    for batch in it:
        if len(batch.shape) != 2:
            raise ValueError("arr must be a 2-dim array")

        if norm == "l2-squared":
            if batch.dtype != np.float32:
                raise ValueError("arr must be of dtype float32")
            index.add(batch)
        elif norm == "hamming":
            if batch.dtype != np.uint8:
                raise ValueError("arr must be of dtype uint8")
            index.add(batch)

    return index


def faiss_duplicates_topk(
    index: FaissIndexTypes, batchsize: int, topk: int, progress: Optional[Progress] = None
) -> Iterator[Tuple[int, int, float]]:
    if batchsize < 1:
        raise ValueError(f"batchsize must be >= 1 (it's {batchsize})")

    if topk < 1:
        raise ValueError(f"topk must be >= 1 (it's {topk})")

    progress = progress or Progress()
    with progress.task(total=index.ntotal, description="Finding duplicates...") as task:
        for i0, ni in slice_idx(index.ntotal, batchsize):
            query = _reconstruct_fixed(index, i0, ni)

            distances, indices = index.search(query, topk)

            rindices = range(i0, i0 + ni)
            for q_indices, q_idx, q_distances in zip(indices, rindices, distances):
                for idx, dist in zip(q_indices, q_distances):
                    if idx == -1:
                        break
                    if idx != q_idx:
                        yield idx, q_idx, dist

            task.advance(ni)


@overload
def faiss_duplicates_threshold(
    index: faiss.IndexFlat, batchsize: int, threshold: float, progress: Optional[Progress] = ...
) -> Iterator[Tuple[int, int, float]]: ...


@overload
def faiss_duplicates_threshold(
    index: faiss.IndexBinary, batchsize: int, threshold: int, progress: Optional[Progress] = ...
) -> Iterator[Tuple[int, int, float]]: ...


def faiss_duplicates_threshold(index, batchsize, threshold, progress=None):
    if batchsize < 1:
        raise ValueError(f"batchsize must be >= 1 (it's {batchsize})")

    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0 (it's {threshold})")

    progress = progress or Progress()
    with progress.task(total=index.ntotal, description="Finding duplicates...") as task:
        for i0, ni in slice_idx(index.ntotal, batchsize):
            query = _reconstruct_fixed(index, i0, ni)

            lims, distances, indices = index.range_search(query, threshold)
            for i in range(ni):
                q_idx = i0 + i
                q_indices = indices[lims[i] : lims[i + 1]]
                q_distances = distances[lims[i] : lims[i + 1]]
                for idx, dist in zip(q_indices, q_distances):
                    if idx < q_idx:
                        yield idx, q_idx, dist

            task.advance(ni)


def faiss_to_pairs(it: Iterable[Tuple[int, int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[List[int]] = []
    dists: List[float] = []

    for a, b, dist in it:
        pairs.append([a, b])
        dists.append(dist)

    if pairs:
        return np.array(pairs), np.array(dists)
    else:
        return np.empty(shape=(0, 2), dtype=np.int64), np.empty(shape=(0,), dtype=np.float32)


def faiss_duplicates_threshold_pairs(
    metric: str,
    arr: Union[np.ndarray, Iterable[np.ndarray]],
    threshold: Union[int, float],
    chunksize: int,
    progress: Progress,
) -> np.ndarray:
    index = faiss_from_array(arr, metric)
    pairs, dists = faiss_to_pairs(faiss_duplicates_threshold(index, chunksize, threshold, progress))
    return pairs


def faiss_duplicates_topk_pairs(
    metric: str, arr: Union[np.ndarray, Iterable[np.ndarray]], topk: int, chunksize: int, progress: Progress
) -> np.ndarray:
    index = faiss_from_array(arr, metric)
    pairs, dists = faiss_to_pairs(faiss_duplicates_topk(index, chunksize, topk, progress))
    return pairs
