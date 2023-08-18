from typing import Optional

import numpy as np
from annoy import AnnoyIndex
from genutility.callbacks import Progress


def annoy_from_array(arr: np.ndarray, norm: str, n_trees: int = 100) -> AnnoyIndex:
    if norm == "euclidean":
        index = AnnoyIndex(arr.shape[1], norm)
    elif norm == "hamming":
        index = AnnoyIndex(arr.shape[1], norm)
    else:
        raise ValueError(f"Invalid norm: {norm}")

    for i in range(arr.shape[0]):
        index.add_item(i, arr[i])
    index.build(n_trees)

    return index


def annoy_duplicates_topk(index: AnnoyIndex, topk: int, progress: Optional[Progress] = None) -> np.ndarray:
    out = []
    progress = progress or Progress()
    for i in progress.track(range(index.get_n_items())):
        items, distances = index.get_nns_by_item(i, topk, include_distances=True)
        out.append(items)
    return np.array(out)
