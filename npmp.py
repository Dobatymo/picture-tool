from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from functools import reduce
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Generic, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
from genutility.numpy import broadcast_shapes, get_num_chunks

Shape = Tuple[int, ...]
Indices = Tuple[slice, ...]
T = TypeVar("T")


def prod(shape):
    return reduce(lambda a, b: a * b, shape, 1)


class SharedNdarray:
    shm: SharedMemory
    shape: Shape
    dtype: type

    def __init__(self, shm: SharedMemory, shape: Shape, dtype: type) -> None:
        self.shm = shm
        self.shape = shape
        self.dtype = dtype

    @property
    def size(self) -> int:
        return prod(self.shape)

    def getarray(self) -> np.ndarray:
        return np.frombuffer(self.shm.buf, dtype=self.dtype, count=self.size).reshape(self.shape)

    def getbuffer(self) -> memoryview:
        return self.shm.buf

    @classmethod
    def create(cls, shape: Shape, dtype: type) -> "SharedNdarray":
        nbytes = prod(shape)
        shm = SharedMemory(create=True, size=nbytes)
        return SharedNdarray(shm, shape, dtype)

    def reshape(self, shape: Shape) -> "SharedNdarray":
        if self.size != prod(shape):
            raise ValueError("New shape is not compatible with old shape")

        return SharedNdarray(self.shm, shape, self.dtype)

    def __str__(self):
        return f"<SharedNdarray shm.name={self.shm.name} shape={self.shape} dtype={self.dtype.__name__} shm.buf={self.shm.buf[:10].hex()}...>"


def chunked_parallel_task(
    func: Callable[..., Any],
    a_arr: SharedNdarray,
    b_arr: SharedNdarray,
    a_idx: Indices,
    b_idx: Indices,
    coords: Shape,
    **kwargs,
) -> np.ndarray:

    a = a_arr.getarray()
    b = b_arr.getarray()

    return func(a[a_idx], b[b_idx], coords, **kwargs)


def _1d_iter(outshape: Shape, chunkshape: Shape) -> Iterator[Tuple[int]]:
    for x in range(0, outshape[0], chunkshape[0]):
        yield (x,)


def _2d_iter(outshape: Shape, chunkshape: Shape) -> Iterator[Tuple[int, int]]:
    for x in range(0, outshape[0], chunkshape[0]):
        for y in range(0, outshape[1], chunkshape[1]):
            yield x, y


def chunked_parallel(
    func: Callable[..., T],
    a_arr: SharedNdarray,
    b_arr: SharedNdarray,
    chunkshape: Shape,
    parallel: Optional[int] = None,
    **kwargs,
) -> Iterator[T]:

    outshape = broadcast_shapes(a_arr.shape, b_arr.shape)
    select_broadcasted_axis = slice(0, 1)

    if len(outshape) - len(chunkshape) != 1:
        raise ValueError("Length of `chunkshape` must be one less the number of input dimensions")

    futures: List[Future] = []

    with ProcessPoolExecutor(parallel) as executor:
        if len(outshape) == 2:
            for (x,) in _1d_iter(outshape, chunkshape):
                if a_arr.shape[0] != outshape[0]:
                    aix = select_broadcasted_axis
                else:
                    aix = slice(x, x + chunkshape[0])

                if b_arr.shape[0] != outshape[0]:
                    bix = select_broadcasted_axis
                else:
                    bix = slice(x, x + chunkshape[0])

                a_idx = (aix, slice(None))
                b_idx = (bix, slice(None))

                future = executor.submit(chunked_parallel_task, func, a_arr, b_arr, a_idx, b_idx, coords=(x,), **kwargs)
                futures.append(future)

        elif len(outshape) == 3:
            for x, y in _2d_iter(outshape, chunkshape):

                if a_arr.shape[0] != outshape[0]:
                    aix = select_broadcasted_axis
                else:
                    aix = slice(x, x + chunkshape[0])

                if b_arr.shape[0] != outshape[0]:
                    bix = select_broadcasted_axis
                else:
                    bix = slice(x, x + chunkshape[0])

                if a_arr.shape[1] != outshape[1]:
                    aiy = select_broadcasted_axis
                else:
                    aiy = slice(y, y + chunkshape[1])

                if b_arr.shape[1] != outshape[1]:
                    biy = select_broadcasted_axis
                else:
                    biy = slice(y, y + chunkshape[1])

                a_idx = (aix, aiy, slice(None))
                b_idx = (bix, biy, slice(None))
                future = executor.submit(
                    chunked_parallel_task, func, a_arr, b_arr, a_idx, b_idx, coords=(x, y), **kwargs
                )
                futures.append(future)

        else:
            raise ValueError("Input must either be 2 or 3 dimensional")

        for future in as_completed(futures):
            yield future.result()


class ChunkedParallel(Generic[T]):
    def __init__(
        self, func: Callable[..., T], a_arr: SharedNdarray, b_arr: SharedNdarray, chunkshape: Shape, **kwargs: Any
    ) -> None:
        self.func = func
        self.a_arr = a_arr
        self.b_arr = b_arr
        self.chunkshape = chunkshape
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[T]:
        return chunked_parallel(self.func, self.a_arr, self.b_arr, self.chunkshape, **self.kwargs)

    def __len__(self) -> int:
        outshape = broadcast_shapes(self.a_arr.shape, self.b_arr.shape)[:-1]
        return get_num_chunks(np.array(outshape), np.array(self.chunkshape))
