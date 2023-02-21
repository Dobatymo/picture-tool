from typing import Optional, Tuple

import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar
from genutility.iter import progress
from genutility.time import PrintStatementTime
from genutility.typing import SizedIterable

from npmp import ChunkedParallel, SharedNdarray


def do(cls, arr):
    a = arr[None, :, :]
    b = arr[:, None, :]
    m = cls.sqrt(cls.sum(cls.power(a - b, 2), axis=-1))
    return cls.argwhere(m < 1)


def l2_dups_chunk(a, b):
    m = np.sqrt(np.sum(np.power(a - b, 2), axis=-1))
    return np.argwhere(m < 1)


def l2_dups_npmp(sharr: SharedNdarray, chunksize: Tuple[int, int]) -> SizedIterable[np.ndarray]:
    if len(sharr.shape) != 2:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    a_arr = sharr.reshape((1, sharr.shape[0], sharr.shape[1]))
    b_arr = sharr.reshape((sharr.shape[0], 1, sharr.shape[1]))

    return ChunkedParallel(l2_dups_chunk, a_arr, b_arr, chunksize, pass_coords=False)


def main(engine: str, dims: Tuple[int, int], chunksize: Tuple[int, int], seed: Optional[int]) -> None:
    rng = np.random.default_rng(seed)
    np_arr = rng.uniform(0, 1, size=dims)

    try:
        with PrintStatementTime():
            if engine == "numpy":
                out = do(np, np_arr)
            elif engine == "npmp":
                out = np.concatenate(list(progress(l2_dups_npmp(SharedNdarray.from_array(np_arr), chunksize))))
            elif engine == "dask":
                with ProgressBar():
                    out = do(da, da.from_array(np_arr, chunks=chunksize)).compute()
            else:
                raise ValueError(engine)
    except MemoryError as e:
        print("MemoryError", e)
    else:
        print(engine, out.shape)


if __name__ == "__main__":
    from argparse import ArgumentParser

    CHUNKSIZE = (1000, 1000)

    parser = ArgumentParser()
    parser.add_argument("--engine", choices=("numpy", "npmp", "dask"), required=True)
    parser.add_argument("--dims", type=int, nargs=2, required=True)
    parser.add_argument("--chunksize", type=int, nargs=2, default=CHUNKSIZE)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    main(args.engine, args.dims, args.chunksize, args.seed)
