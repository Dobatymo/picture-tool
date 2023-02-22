import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import reduce
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
from genutility.numpy import broadcast_shapes, get_num_chunks
from threadpoolctl import threadpool_limits

Shape = Tuple[int, ...]
Indices = Tuple[slice, ...]
T = TypeVar("T")

THREADPOOL_LIMIT = None


def prod(shape):
    return reduce(lambda a, b: a * b, shape, 1)


class BaseArray:
    shape: Shape


class SharedNdarray(BaseArray):
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

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SharedNdarray":
        shm = SharedMemory(create=True, size=arr.nbytes)
        shm.buf[:] = arr.tobytes()
        return SharedNdarray(shm, arr.shape, arr.dtype)

    def reshape(self, shape: Shape) -> "SharedNdarray":
        if self.size != prod(shape):
            raise ValueError("New shape is not compatible with old shape")

        return SharedNdarray(self.shm, shape, self.dtype)

    def __str__(self):
        return f"<SharedNdarray shm.name={self.shm.name} shape={self.shape} dtype={self.dtype.__name__} shm.buf={self.shm.buf[:10].hex()}...>"


def chunked_parallel_task_mt(
    func: Callable[..., Any],
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    a_idx: Indices,
    b_idx: Indices,
    coords: Optional[Shape],
    **kwargs,
) -> np.ndarray:
    if coords is None:
        return func(a_arr[a_idx], b_arr[b_idx], **kwargs)
    else:
        return func(a_arr[a_idx], b_arr[b_idx], coords, **kwargs)


def chunked_parallel_task_mp(
    func: Callable[..., Any],
    a_arr: SharedNdarray,
    b_arr: SharedNdarray,
    a_idx: Indices,
    b_idx: Indices,
    coords: Optional[Shape],
    **kwargs,
) -> np.ndarray:
    a = a_arr.getarray()
    b = b_arr.getarray()

    if coords is None:
        return func(a[a_idx], b[b_idx], **kwargs)
    else:
        return func(a[a_idx], b[b_idx], coords, **kwargs)


def _1d_iter(outshape: Shape, chunkshape: Shape) -> Iterator[Tuple[int]]:
    for x in range(0, outshape[0], chunkshape[0]):
        yield (x,)


def _2d_iter(outshape: Shape, chunkshape: Shape) -> Iterator[Tuple[int, int]]:
    for x in range(0, outshape[0], chunkshape[0]):
        for y in range(0, outshape[1], chunkshape[1]):
            yield x, y


def chunked_parallel(
    backend: str,
    func: Callable[..., T],
    a_arr: Union[np.ndarray, SharedNdarray],
    b_arr: Union[np.ndarray, SharedNdarray],
    chunkshape: Shape,
    ordered: bool,
    pass_coords: bool,
    parallel: Optional[int] = None,
    **kwargs,
) -> Iterator[T]:
    outshape = broadcast_shapes(a_arr.shape, b_arr.shape)
    select_broadcasted_axis = slice(0, 1)

    if len(outshape) - len(chunkshape) != 1:
        raise ValueError("Length of `chunkshape` must be one less the number of input dimensions")

    """ The default `max_workers=None` value for `ThreadPoolExecutor` uses `min(32, os.cpu_count() + 4),
        because IO workloads are assumed. Here we have CPU bound workloads however.
    """
    if parallel is None:
        parallel = os.cpu_count()

    if backend == "threading":
        executor = ThreadPoolExecutor(parallel)
        executor_task = chunked_parallel_task_mt
    elif backend == "multiprocessing":
        if not isinstance(a_arr, SharedNdarray) or not isinstance(b_arr, SharedNdarray):
            raise TypeError(
                f"Input arrays must be of type `SharedNdarray` not `{type(a_arr)}`, `{type(b_arr)}` for multiprocessing"
            )

        executor = ProcessPoolExecutor(parallel)
        executor_task = chunked_parallel_task_mp
    else:
        raise ValueError(f"Invalid backend: {backend}")

    futures: List[Future] = []

    with executor as exe, threadpool_limits(limits=THREADPOOL_LIMIT):
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

                if pass_coords:
                    coords = (x,)
                else:
                    coords = None

                future = exe.submit(executor_task, func, a_arr, b_arr, a_idx, b_idx, coords, **kwargs)
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
                if pass_coords:
                    coords = (x, y)
                else:
                    coords = None

                future = exe.submit(executor_task, func, a_arr, b_arr, a_idx, b_idx, coords, **kwargs)
                futures.append(future)

        else:
            raise ValueError(f"Input must either be 2 or 3 dimensional. It's {len(outshape)}.")

        if ordered:
            for future in futures:
                yield future.result()
        else:
            for future in as_completed(futures):
                yield future.result()


class ChunkedParallel(Generic[T]):
    def __init__(
        self,
        func: Callable[..., T],
        a_arr: Union[np.ndarray, SharedNdarray],
        b_arr: Union[np.ndarray, SharedNdarray],
        chunkshape: Shape,
        backend: str = "multiprocessing",
        ordered: bool = False,
        pass_coords: bool = True,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.func = func
        self.a_arr = a_arr
        self.b_arr = b_arr
        self.chunkshape = chunkshape
        self.backend = backend
        self.ordered = ordered
        self.pass_coords = pass_coords
        self.parallel = parallel
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[T]:
        return chunked_parallel(
            self.backend,
            self.func,
            self.a_arr,
            self.b_arr,
            self.chunkshape,
            self.ordered,
            self.pass_coords,
            self.parallel,
            **self.kwargs,
        )

    def __len__(self) -> int:
        outshape = broadcast_shapes(self.a_arr.shape, self.b_arr.shape)[:-1]
        return get_num_chunks(np.array(outshape), np.array(self.chunkshape))
