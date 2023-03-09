from typing import Iterator, Tuple, Union, overload

import faiss
from tqdm import tqdm

from .utils import slice_idx


def faiss_duplicates_top_k(
    index: Union[faiss.IndexFlat, faiss.IndexBinary], batchsize: int, top_k: int, verbose: bool = False
) -> Iterator[Tuple[int, int, float]]:
    if batchsize < 1:
        raise ValueError("batchsize must be >= 1 (it's {batchsize})")

    if top_k < 1:
        raise ValueError("top_k must be >= 1 (it's {top_k})")

    with tqdm(total=index.ntotal, disable=not verbose) as pbar:
        for i0, ni in slice_idx(index.ntotal, batchsize):
            query = index.reconstruct_n(i0, ni)
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
    index: faiss.IndexFlat, batchsize: int, threshold: float, verbose: bool
) -> Iterator[Tuple[int, int, float]]:
    ...


@overload
def faiss_duplicates_threshold(
    index: faiss.IndexBinary, batchsize: int, threshold: int, verbose: bool
) -> Iterator[Tuple[int, int, float]]:
    ...


def faiss_duplicates_threshold(index, batchsize, threshold, verbose=False):
    if batchsize < 1:
        raise ValueError("batchsize must be >= 1 (it's {batchsize})")

    if threshold < 0.0:
        raise ValueError("threshold must be >= 0 (it's {threshold})")

    with tqdm(total=index.ntotal, disable=not verbose) as pbar:
        for i0, ni in slice_idx(index.ntotal, batchsize):
            query = index.reconstruct_n(i0, ni)  # fixme: a preallocated array could be used here

            lims, distances, indices = index.range_search(query, threshold)
            for i in range(ni):
                q_idx = i0 + i
                q_indices = indices[lims[i] : lims[i + 1]]
                q_distances = distances[lims[i] : lims[i + 1]]
                for idx, dist in zip(q_indices, q_distances):
                    if idx != q_idx:
                        yield idx, q_idx, dist

            pbar.update(ni)
