import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm


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
