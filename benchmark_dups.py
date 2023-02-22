from typing import Optional, Tuple

import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar
from genutility.iter import progress
from genutility.time import PrintStatementTime
from genutility.typing import SizedIterable

from npmp import ChunkedParallel, SharedNdarray


def l2_dups_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.sqrt(np.sum(np.power(a - b, 2), axis=-1))
    return np.argwhere(m < 1)


def l2_dups_numpy(arr: np.ndarray):
    a_arr = arr[None, :, :]
    b_arr = arr[:, None, :]
    return l2_dups_chunk(a_arr, b_arr)


def l2_dups_npmt(arr: np.ndarray, chunksize: int) -> SizedIterable[np.ndarray]:
    if len(arr.shape) != 2:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    a_arr = arr[None, :, :]
    b_arr = arr[:, None, :]

    return ChunkedParallel(l2_dups_chunk, a_arr, b_arr, (chunksize, chunksize), backend="threading", pass_coords=False)


def l2_dups_npmp(sharr: SharedNdarray, chunksize: int) -> SizedIterable[np.ndarray]:
    if len(sharr.shape) != 2:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    a_arr = sharr.reshape((1, sharr.shape[0], sharr.shape[1]))
    b_arr = sharr.reshape((sharr.shape[0], 1, sharr.shape[1]))

    return ChunkedParallel(
        l2_dups_chunk, a_arr, b_arr, (chunksize, chunksize), backend="multiprocessing", pass_coords=False
    )


def l2_dups_dask(np_arr: np.ndarray, chunksize: int):
    arr = da.from_array(np_arr, chunks=(chunksize, -1))
    a = da.broadcast_to(arr[None, :, :], (arr.shape[0], arr.shape[0], arr.shape[1]), chunks=(chunksize, chunksize, -1))
    b = da.broadcast_to(arr[:, None, :], (arr.shape[0], arr.shape[0], arr.shape[1]), chunks=(chunksize, chunksize, -1))
    m = da.sqrt(da.sum(da.power(a - b, 2), axis=-1))
    return da.argwhere(m < 1).compute()


def main(engine: str, dims: Tuple[int, int], chunksize: int, seed: Optional[int]) -> None:
    rng = np.random.default_rng(seed)
    np_arr = rng.uniform(0, 1, size=dims).astype(np.float32)

    try:
        with PrintStatementTime():
            if engine == "numpy":
                out = l2_dups_numpy(np_arr)
            elif engine == "npmt":
                out = np.concatenate(list(progress(l2_dups_npmt(np_arr, chunksize))))
            elif engine == "npmp":
                out = np.concatenate(list(progress(l2_dups_npmp(SharedNdarray.from_array(np_arr), chunksize))))
            elif engine == "dask":
                with ProgressBar():
                    out = l2_dups_dask(np_arr, chunksize)
            else:
                raise ValueError(engine)
    except MemoryError as e:
        print("MemoryError", e)
    else:
        print(engine, out.shape)


if __name__ == "__main__":
    from argparse import ArgumentParser

    CHUNKSIZE = 1000

    parser = ArgumentParser()
    parser.add_argument("--engine", choices=("numpy", "npmt", "npmp", "dask"), required=True)
    parser.add_argument("--dims", type=int, nargs=2, required=True)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    main(args.engine, args.dims, args.chunksize, args.seed)
