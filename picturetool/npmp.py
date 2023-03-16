import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import reduce, wraps
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from genutility.numpy import broadcast_shapes, get_num_chunks
from threadpoolctl import threadpool_limits

Shape = Tuple[int, ...]
Indices = Tuple[slice, ...]
T = TypeVar("T")

THREADPOOL_LIMIT: Optional[int] = None


def copy_docs(func_with_docs: Callable) -> Callable:
    """Decorator to apply docstring of `func_with_docs` to decorated function."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        inner.__doc__ = func_with_docs.__doc__
        return inner

    return decorator


def prod(shape):
    return reduce(lambda a, b: a * b, shape, 1)


class SharedNdarray:
    shm: SharedMemory
    shape: Shape
    dtype: np.dtype
    offset: int
    strides: Optional[Shape]
    rawsize: int

    def __init__(
        self,
        shm: SharedMemory,
        shape: Shape,
        dtype: np.dtype,
        offset: int = 0,
        strides: Optional[Shape] = None,
        rawsize: Optional[int] = None,
    ) -> None:
        if shm.buf.ndim != 1:
            raise ValueError("The SharedMemory memoryview must be 1-dimensional")

        if strides is not None and len(shape) != len(strides):
            raise ValueError("shape and strides dimensions do not match")

        _rawsize = self._nbytes(shape, dtype)

        if strides is None and rawsize is not None and _rawsize != rawsize:
            raise ValueError("rawsize doesn't match shape and dtype")

        if strides is not None and rawsize is None:
            raise ValueError("rawsize must be specified when strides is not None")

        assert shm.size == shm.buf.nbytes, (shm.size, shm.buf.nbytes)

        self.shm = shm
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.strides = strides
        self.rawsize = rawsize or _rawsize

    @staticmethod
    def _nbytes(shape: Shape, dtype: np.dtype) -> int:
        return prod(shape) * dtype.itemsize

    @property
    def size(self) -> int:
        """Number of elements in the array."""

        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the elements of the array if it were using default strides.
        The actual size of the underlying shared memory buffer might be smaller or larger.
        The actual size of the underlying buffer is `SharedNdarray.shm.size`,
        for the memory required to store the data, see `SharedNdarray.rawsize`.
        """

        return self._nbytes(self.shape, self.dtype)  # not the same as `self.shm.buf.nbytes`

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def getarray(self) -> np.ndarray:
        """Returns a numpy array using the shared memory buffer."""

        if self.strides is None:
            return np.frombuffer(self.shm.buf, dtype=self.dtype, count=self.size).reshape(self.shape)
        else:
            return np.ndarray(self.shape, self.dtype, self.shm.buf, offset=self.offset, strides=self.strides)

    def getbuffer(self) -> memoryview:
        """Returns the raw memoryview of the underlying buffer."""

        return self.shm.buf

    def tobytes(self) -> bytes:
        return self.shm.buf[: self.rawsize].tobytes()

    @classmethod
    def create(cls, shape: Shape, dtype: npt.DTypeLike) -> "SharedNdarray":
        """Creates a SharedNdarray object from a shape and dtype."""

        _dtype = np.dtype(dtype)
        nbytes = cls._nbytes(shape, _dtype)
        shm = SharedMemory(create=True, size=nbytes)
        return cls(shm, shape, _dtype)

    @classmethod
    def from_array(cls, arr: np.ndarray, c_contiguous: bool = True) -> "SharedNdarray":
        """Creates a SharedNdarray object from a numpy array.
        The data is copied and made C-contiguous.
        """

        if arr.base is None:
            c_contiguous = True

        if c_contiguous:
            rawsize = arr.nbytes
            shm = SharedMemory(create=True, size=rawsize)
            shm.buf[:rawsize] = arr.tobytes()
            offset = 0
            strides = None
        else:
            assert arr.base is not None
            rawsize = arr.base.data.nbytes
            shm = SharedMemory(create=True, size=rawsize)
            shm.buf[:rawsize] = arr.base.data.tobytes()
            offset = arr.__array_interface__["data"][0] - arr.base.__array_interface__["data"][0]
            strides = arr.strides

        return cls(shm, arr.shape, arr.dtype, offset, strides, rawsize)

    def reshape(self, shape: Shape) -> "SharedNdarray":
        """Changes the shape of the array to `shape`.
        The underlying shared memory buffer is not copied or modified.
        """

        if self.strides is not None:
            raise ValueError("Cannot reshape arrays with custom strides")

        if self.size != prod(shape):
            raise ValueError("New shape is not compatible with old shape")

        return SharedNdarray(self.shm, shape, self.dtype)

    def astype(self, dtype: npt.DTypeLike) -> "SharedNdarray":
        """Changes the dtype of the array to `dtype`.
        The underlying shared memory buffer is not copied or modified.
        """

        if self.strides is not None:
            raise ValueError("Cannot retype arrays with custom strides")

        return SharedNdarray(self.shm, self.shape, np.dtype(dtype))

    def __str__(self) -> str:
        """String representation of the object.
        `shm.buf` does not show the actual memory contents, but the interpretation regarding offset and strides.
        """

        if self.nbytes > 10:
            shm_buf = f"{self.shm.buf[:10].hex()}..."
        else:
            shm_buf = self.shm.buf[: self.nbytes].hex()
        return f"<SharedNdarray shm.name={self.shm.name} shape={self.shape} dtype={self.dtype.name} strides={self.strides} shm.buf={shm_buf}>"

    def __enter__(self) -> "SharedNdarray":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @copy_docs(SharedMemory.close)
    def close(self) -> None:
        self.shm.close()

    @copy_docs(SharedMemory.unlink)
    def unlink(self) -> None:
        self.shm.unlink()


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
        return func(a_arr[a_idx], b_arr[b_idx], coords=coords, **kwargs)


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
        return func(a[a_idx], b[b_idx], coords=coords, **kwargs)


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
