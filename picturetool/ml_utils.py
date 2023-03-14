from enum import Enum
from typing import Iterable, Iterator, List, Tuple, Union, overload

import faiss
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm

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


def faiss_from_array(arr: np.ndarray, norm: str) -> Union[faiss.IndexFlatL2, faiss.IndexBinaryFlat]:
    if len(arr.shape) != 2:
        raise ValueError("arr must be a 2-dim array")

    if norm == "l2-squared":
        if arr.dtype != np.float32:
            raise ValueError("arr must be of dtype float32")
        index = faiss.IndexFlatL2(arr.shape[1])
        assert FaissMetric(index.metric_type).name == "L2"
        index.add(arr)
    elif norm == "hamming":
        if arr.dtype != np.uint8:
            raise ValueError("arr must be of dtype uint8")
        index = faiss.IndexBinaryFlat(arr.shape[1] * 8)
        assert index.metric_type == 1  # why?
        index.add(arr)
    else:
        raise ValueError(f"Invalid norm: {norm}")

    return index


def faiss_duplicates_top_k(
    index: FaissIndexTypes, batchsize: int, top_k: int, verbose: bool = False
) -> Iterator[Tuple[int, int, float]]:
    if batchsize < 1:
        raise ValueError("batchsize must be >= 1 (it's {batchsize})")

    if top_k < 1:
        raise ValueError("top_k must be >= 1 (it's {top_k})")

    with tqdm(total=index.ntotal, disable=not verbose) as pbar:
        for i0, ni in slice_idx(index.ntotal, batchsize):
            query = _reconstruct_fixed(index, i0, ni)

            distances, indices = index.search(query, top_k)

            rindices = range(i0, i0 + ni)
            for q_indices, q_idx, q_distances in zip(indices, rindices, distances):
                for idx, dist in zip(q_indices, q_distances):
                    if idx == -1:
                        break
                    if idx != q_idx:
                        yield idx, q_idx, dist

            pbar.update(ni)


@overload
def faiss_duplicates_threshold(
    index: faiss.IndexFlat, batchsize: int, threshold: float, verbose: bool = ...
) -> Iterator[Tuple[int, int, float]]:
    ...


@overload
def faiss_duplicates_threshold(
    index: faiss.IndexBinary, batchsize: int, threshold: int, verbose: bool = ...
) -> Iterator[Tuple[int, int, float]]:
    ...


def faiss_duplicates_threshold(index, batchsize, threshold, verbose=False):
    if batchsize < 1:
        raise ValueError("batchsize must be >= 1 (it's {batchsize})")

    if threshold < 0.0:
        raise ValueError("threshold must be >= 0 (it's {threshold})")

    with tqdm(total=index.ntotal, disable=not verbose) as pbar:
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

            pbar.update(ni)


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


def annoy_from_array(arr: np.ndarray, norm: str, n_trees: int = 100) -> AnnoyIndex:
    if norm == "euclidean":
        index = AnnoyIndex(arr.shape[1], norm)
    elif norm == "hamming":
        index = AnnoyIndex(arr.shape[1] * 8, norm)
    else:
        raise ValueError(f"Invalid norm: {norm}")

    for i in range(arr.shape[0]):
        index.add_item(i, arr[i])
    index.build(n_trees)

    return index


def annoy_duplicates_top_k(index: AnnoyIndex, top_k: int, verbose: bool = False) -> np.ndarray:
    out = []
    for i in tqdm(range(index.get_n_items()), disable=not verbose):
        items, distances = index.get_nns_by_item(i, top_k, include_distances=True)
        out.append(items)
    return np.array(out)