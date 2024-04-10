import logging
import logging.handlers
import multiprocessing
import os
import platform
import re
import subprocess  # nosec
import sys
import threading
from datetime import datetime, timedelta, tzinfo
from fractions import Fraction
from functools import total_ordering
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypedDict, TypeVar, Union

import networkx as nx
import numpy as np
import pandas as pd
import piexif
from dateutil import tz
from genutility._files import to_dos_path
from genutility.callbacks import Progress
from genutility.datetime import is_aware
from genutility.filesdb import FileDbWithId
from genutility.numpy import hamming_dist_packed
from genutility.typing import SizedIterable
from pandas._typing import ValueKeyFunc
from platformdirs import user_data_dir
from rich.console import ConsoleRenderable
from rich.logging import RichHandler
from rich.text import Text as RichText
from rich.traceback import Traceback
from typing_extensions import NotRequired, Self

from . import npmp

T = TypeVar("T")
Shape = Tuple[int, ...]

APP_NAME = "picture-tool"
APP_AUTHOR = "Dobatymo"
APP_VERSION = "0.1"

DEFAULT_APPDATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
DEFAULT_HASHDB = DEFAULT_APPDATA_DIR / "hashes.sqlite"

GpsT = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]

extensions_images = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".heic", ".heif", ".webp"}
extensions_jpeg = {".jpg", ".jpeg"}
extensions_heif = {".heic", ".heif"}
extensions_tiff = {".tif", ".tiff", ".dng"}
extensions_exif = extensions_jpeg | extensions_heif | extensions_tiff


def _gps_tuple_to_fraction(gps: GpsT) -> Fraction:
    return Fraction(*gps[0]) + Fraction(*gps[1]) / 60 + Fraction(*gps[2]) / 3600


def parse_gpsinfo(gpsinfo: Dict[str, Any]) -> Tuple[float, float]:
    lat = gpsinfo["GPSLatitude"]
    lon = gpsinfo["GPSLongitude"]

    return float(_gps_tuple_to_fraction(lat)), float(_gps_tuple_to_fraction(lon))


def to_datetime(col: pd.Series, in_tz: Optional[tzinfo] = None, out_tz: Optional[tzinfo] = None) -> datetime:
    in_tz = in_tz or tz.tzlocal()
    out = pd.Series(index=col.index, dtype="object")
    for i in range(len(col)):
        s = col.iloc[i]
        if s:
            dt = datetime.fromisoformat(s)
            if not is_aware(dt):  # naive
                dt = dt.replace(tzinfo=in_tz)  # assume timezone
            dt = dt.astimezone(out_tz).replace(tzinfo=None)  # convert to local time and throw away the timezone
            out.iloc[i] = dt
        else:
            out.iloc[i] = None
    return pd.to_datetime(out)


def with_stem(path: Path, stem: str) -> Path:
    return path.with_name(stem + path.suffix)


TUPLE_WITH_ZERO = (0,)


def unique_pairs(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr

    b = arr[:, 0] < arr[:, 1]
    return arr[b]


def array_from_iter(it: Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(it))


def npmp_to_pairs(it: Iterable[np.ndarray]) -> np.ndarray:
    return unique_pairs(array_from_iter(it))


def hamming_duplicates_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, axis: int, threshold: int, coords: Tuple[int, ...] = TUPLE_WITH_ZERO
) -> np.ndarray:
    """Returns a list of coordinate pairs which are duplicates."""

    dists = hamming_dist_packed(a_arr, b_arr, axis)
    indices = np.argwhere(dists <= threshold)
    return indices + np.array(coords)


def hamming_duplicates_topk_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, axis: int, topk: int, coords: Tuple[int, ...] = TUPLE_WITH_ZERO
) -> np.ndarray:
    """Returns a list of coordinate pairs which are duplicates."""

    dists = hamming_dist_packed(a_arr, b_arr, axis)
    indices = np.argpartition(dists, topk, axis=-1)[:, :topk]
    return indices + np.array(coords)


def l2_duplicates_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, threshold: float, coords: Tuple[int, ...] = TUPLE_WITH_ZERO
) -> np.ndarray:
    diff = a_arr - b_arr
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    indices = np.argwhere(dists <= threshold)
    return indices + np.array(coords)


def l2_duplicates_topk_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, topk: int, coords: Tuple[int, ...] = TUPLE_WITH_ZERO
) -> np.ndarray:
    diff = a_arr - b_arr
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    indices = np.argpartition(dists, topk, axis=-1)[:, :topk]
    return indices + np.array(coords)


def l2squared_duplicates_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, threshold: float, coords: Tuple[int, ...] = TUPLE_WITH_ZERO
) -> np.ndarray:
    diff = a_arr - b_arr
    dists = np.sum(diff * diff, axis=-1)
    indices = np.argwhere(dists <= threshold)
    return indices + np.array(coords)


def l2squared_duplicates_topk_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, topk: int, coords: Tuple[int, ...] = TUPLE_WITH_ZERO
) -> np.ndarray:
    diff = a_arr - b_arr
    dists = np.sum(diff * diff, axis=-1)
    indices = np.argpartition(dists, topk, axis=-1)[:, :topk]
    return indices + np.array(coords)


def npmp_duplicates_binary(sharr: npmp.SharedNdarray, chunkshape: Shape, func, **kwargs) -> SizedIterable[np.ndarray]:
    if len(sharr.shape) != 2 or sharr.dtype != np.uint8:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    a_arr = sharr.reshape((1, sharr.shape[0], sharr.shape[1]))
    b_arr = sharr.reshape((sharr.shape[0], 1, sharr.shape[1]))

    return npmp.ChunkedParallel(func, a_arr, b_arr, chunkshape, backend="multiprocessing", **kwargs)


def npmp_duplicates(sharr: npmp.SharedNdarray, chunkshape: Shape, func, **kwargs) -> SizedIterable[np.ndarray]:
    if len(sharr.shape) != 2:
        raise ValueError("Input must be a list of hashes (2-dimensional array)")

    a_arr = sharr.reshape((1, sharr.shape[0], sharr.shape[1]))
    b_arr = sharr.reshape((sharr.shape[0], 1, sharr.shape[1]))

    return npmp.ChunkedParallel(func, a_arr, b_arr, chunkshape, backend="multiprocessing", **kwargs)


def npmt_duplicates_binary(arr: np.ndarray, chunkshape: Shape, func, **kwargs) -> SizedIterable[np.ndarray]:
    if len(arr.shape) != 2 or arr.dtype != np.uint8:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    a_arr = arr[None, :, :]
    b_arr = arr[:, None, :]

    return npmp.ChunkedParallel(func, a_arr, b_arr, chunkshape, backend="threading", **kwargs)


def npmt_duplicates(arr: np.ndarray, chunkshape: Shape, func, **kwargs) -> SizedIterable[np.ndarray]:
    if len(arr.shape) != 2:
        raise ValueError("Input must be a list of hashes (2-dimensional array)")

    a_arr = arr[None, :, :]
    b_arr = arr[:, None, :]

    return npmp.ChunkedParallel(func, a_arr, b_arr, chunkshape, backend="threading", **kwargs)


def buffer_fill(it: Iterable[bytes], buffer: memoryview) -> None:
    pos = 0
    for buf in it:
        size = len(buf)
        buffer[pos : pos + size] = buf
        pos += size


def npmp_duplicates_threshold_pairs(
    metric: str,
    hashes: Union[np.ndarray, List[bytes]],
    threshold: Union[int, float],
    chunksize: int,
    progress: Optional[Progress] = None,
) -> np.ndarray:
    # npmp.THREADPOOL_LIMIT = limit

    if metric == "hamming":
        if not isinstance(threshold, int):
            raise TypeError()
        if isinstance(hashes, np.ndarray):
            sharr = npmp.SharedNdarray.from_array(hashes)
        elif isinstance(hashes, list):
            sharr = npmp.SharedNdarray.create((len(hashes), len(hashes[0])), np.uint8)
            buffer_fill(hashes, sharr.getbuffer())
        else:
            raise TypeError()
        it = npmp_duplicates_binary(
            sharr, (chunksize, chunksize), hamming_duplicates_chunk, axis=-1, threshold=threshold
        )
    elif metric == "l2":
        if not isinstance(threshold, float):
            raise TypeError()
        assert isinstance(hashes, np.ndarray)
        sharr = npmp.SharedNdarray.from_array(hashes)
        it = npmp_duplicates(sharr, (chunksize, chunksize), l2_duplicates_chunk, threshold=threshold)
    elif metric == "l2-squared":
        if not isinstance(threshold, float):
            raise TypeError()
        assert isinstance(hashes, np.ndarray)
        sharr = npmp.SharedNdarray.from_array(hashes)
        it = npmp_duplicates(sharr, (chunksize, chunksize), l2squared_duplicates_chunk, threshold=threshold)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    if progress is not None:
        it = progress.track(it, description="Matching hashes")

    out = npmp_to_pairs(it)
    del sharr
    return out


def npmp_duplicates_topk_pairs(
    metric: str, hashes: Union[np.ndarray, List[bytes]], topk: int, chunksize: int, progress: Optional[Progress] = None
) -> np.ndarray:
    # npmp.THREADPOOL_LIMIT = limit

    if metric == "hamming":
        if isinstance(hashes, np.ndarray):
            sharr = npmp.SharedNdarray.from_array(hashes)
        elif isinstance(hashes, list):
            sharr = npmp.SharedNdarray.create((len(hashes), len(hashes[0])), np.uint8)
            buffer_fill(hashes, sharr.getbuffer())
        else:
            raise TypeError()
        it = npmp_duplicates_binary(sharr, (chunksize, chunksize), hamming_duplicates_topk_chunk, axis=-1, topk=topk)
    elif metric == "l2":
        assert isinstance(hashes, np.ndarray)
        sharr = npmp.SharedNdarray.from_array(hashes)
        it = npmp_duplicates(sharr, (chunksize, chunksize), l2_duplicates_topk_chunk, topk=topk)
    elif metric == "l2-squared":
        assert isinstance(hashes, np.ndarray)
        sharr = npmp.SharedNdarray.from_array(hashes)
        it = npmp_duplicates(sharr, (chunksize, chunksize), l2squared_duplicates_topk_chunk, topk=topk)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    if progress is not None:
        it = progress.track(it, description="Matching hashes")

    out = array_from_iter(it)
    del sharr
    return out


def npmt_duplicates_threshold_pairs(
    metric: str, hashes: np.ndarray, threshold: Union[int, float], chunksize: int, progress: Optional[Progress] = None
) -> np.ndarray:
    # npmp.THREADPOOL_LIMIT = limit

    if metric == "hamming":
        if not isinstance(threshold, int):
            raise TypeError()
        it = npmt_duplicates_binary(
            hashes, (chunksize, chunksize), hamming_duplicates_chunk, axis=-1, threshold=threshold
        )
    elif metric == "l2":
        if not isinstance(threshold, float):
            raise TypeError()
        it = npmt_duplicates(hashes, (chunksize, chunksize), l2_duplicates_chunk, threshold=threshold)
    elif metric == "l2-squared":
        if not isinstance(threshold, float):
            raise TypeError()
        it = npmt_duplicates(hashes, (chunksize, chunksize), l2squared_duplicates_chunk, threshold=threshold)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    if progress is not None:
        it = progress.track(it, description="Matching hashes")

    return npmp_to_pairs(it)


def npmt_duplicates_topk_pairs(
    metric: str, hashes: np.ndarray, topk: int, chunksize: int, progress: Optional[Progress] = None
) -> np.ndarray:
    # npmp.THREADPOOL_LIMIT = limit

    if metric == "hamming":
        it = npmt_duplicates_binary(hashes, (chunksize, chunksize), hamming_duplicates_topk_chunk, axis=-1, topk=topk)
    elif metric == "l2":
        it = npmt_duplicates(hashes, (chunksize, chunksize), l2_duplicates_topk_chunk, topk=topk)
    elif metric == "l2-squared":
        it = npmt_duplicates(hashes, (chunksize, chunksize), l2squared_duplicates_topk_chunk, topk=topk)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    if progress is not None:
        it = progress.track(it, description="Matching hashes")

    return array_from_iter(it)


@total_ordering
class MaxType:
    def __le__(self, other):
        return False

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "Max"


Max = MaxType()


def pd_sort_groups_by_first_row(
    df: pd.DataFrame,
    group_by_column: str,
    sort_by_column: str,
    ascending: bool = True,
    kind: str = "stable",
    key: ValueKeyFunc = None,
) -> pd.DataFrame:
    """Sort groups by first row, leave order within groups intact."""

    idx = (
        df.groupby(group_by_column, sort=False, group_keys=False)
        .nth(0)
        .sort_values(sort_by_column, ascending=ascending, kind=kind, key=key)
        .index
    )
    return df.loc[idx]


class SortValuesKwArgs(TypedDict):
    by: str
    ascending: NotRequired[bool]
    key: NotRequired[ValueKeyFunc]


def pd_sort_within_group_slow(
    df: pd.DataFrame, group_by_column: str, sort_kwargs: Sequence[SortValuesKwArgs], sort_groups: bool = False
):
    """Sort rows within groups, leave the order of the groups intact.
    Specify a list of multiple sorting arguments which are applied in order.
    """

    def multisort(x: pd.DataFrame) -> pd.DataFrame:
        for kwargs in reversed(sort_kwargs):
            x = x.sort_values(kind="stable", inplace=False, ignore_index=False, **kwargs)
        return x

    # set `group_keys` to `True`, otherwise `sort` isn't working...
    # and even with that fix, the whole thing is broken in pandas<1.5...
    df = df.groupby(group_by_column, sort=sort_groups, group_keys=True).apply(multisort)
    # drop column added by `group_keys=True`
    df = df.reset_index(0, drop=True)

    return df


def pd_sort_within_group(
    df: pd.DataFrame, group_by: str, sort_kwargs: Sequence[SortValuesKwArgs], sort_groups: bool = False
):
    """Sort rows within groups, leave the order of the groups intact.
    Specify a list of multiple sorting arguments which are applied in order.
    `group_by`: can be a column name or index level name
    """

    is_index = group_by in df.index.names
    is_column = group_by in df.columns

    if is_index and is_column:
        raise ValueError(f"Ambigus name: {group_by}")
    elif not is_index and not is_column:
        raise ValueError(f"Invalid name: {group_by}")

    if not sort_groups:
        if not is_index:
            df = df.set_index(group_by, drop=True, append=True)

        idx = df.index.get_level_values(group_by).unique()

    for kwargs in reversed(sort_kwargs):
        df = df.sort_values(kind="stable", inplace=False, ignore_index=False, **kwargs)

    if sort_groups:
        if is_index:
            return df.sort_index(level=group_by, kind="stable")
        else:
            return df.sort_values(group_by, kind="stable")
    else:
        if df.index.nlevels == 1:
            multiidx = idx
        else:
            multiidx = (slice(None),) * (df.index.nlevels - 1) + (idx,)

        df = df.loc[multiidx, :]
        if not is_index:
            df = df.reset_index(group_by, drop=False)
        return df


def make_datetime(date: bytes, subsec: Optional[bytes] = None, offset: Optional[bytes] = None) -> datetime:
    """Converts `date` with optional `subsec` and `offset` to a python datetime object.
    Raises ValueError if the input is invalid.
    """

    date_str = date.rstrip(b"\0").decode("ascii")

    if offset:
        offset_str = offset.rstrip(b"\0").decode("ascii")
        dt = datetime.strptime(f"{date_str} {offset_str}", "%Y:%m:%d %H:%M:%S %z")
    else:
        # `dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")`
        # would solve the problem according to the exif spec.
        # However we are a little bit more lenient in our parsing
        # since some apps don't follow the spec exactly.

        m = re.match(r"(\d+)[:-](\d+)[:-](\d+) (\d+):(\d+):(\d+)", date_str)
        if m:
            timetuple = tuple(map(int, m.groups()))
            dt = datetime(*timetuple)  # type: ignore[arg-type]
        else:
            raise ValueError(f"time data '{date_str}' does not match '%Y:%m:%d %H:%M:%S' or '%Y-%m-%d %H:%M:%S'")

    if subsec:
        subsec_str = subsec.rstrip(b"\0").decode("ascii")
        dt += timedelta(seconds=int(subsec_str) / 10 ** len(subsec_str))

    return dt


def get_exif_dates(exif: dict) -> Dict[str, datetime]:
    """Takes a piexif exif dict and returns a dictionary with 'modified', 'original' and 'created' datetimes.
    If the respective times are missing from the input, the are excluded from the output dict.
    The datetime can be naive or aware depending on the information provided.
    """

    d = {
        "modified": (("ImageIFD", "DateTime"), ("ExifIFD", "OffsetTime"), ("ExifIFD", "SubSecTime")),
        "original": (
            ("ExifIFD", "DateTimeOriginal"),
            ("ExifIFD", "OffsetTimeOriginal"),
            ("ExifIFD", "SubSecTimeOriginal"),
        ),
        "digitized": (
            ("ExifIFD", "DateTimeDigitized"),
            ("ExifIFD", "OffsetTimeDigitized"),
            ("ExifIFD", "SubSecTimeDigitized"),
        ),
    }

    m1 = {"ImageIFD": "0th", "ExifIFD": "Exif"}

    m2 = {"ImageIFD": piexif.ImageIFD, "ExifIFD": piexif.ExifIFD}

    out: Dict[str, datetime] = {}

    for field in d.keys():
        try:
            date_idx, offet_idx, subsec_idx = d[field]
            date = exif[m1[date_idx[0]]][getattr(m2[date_idx[0]], date_idx[1])]
            subsec = exif[m1[subsec_idx[0]]].get(getattr(m2[subsec_idx[0]], subsec_idx[1]), None)
            offset = exif[m1[offet_idx[0]]].get(getattr(m2[offet_idx[0]], offet_idx[1]), None)
            out[field] = make_datetime(date, subsec, offset)
        except KeyError:  # missing info
            pass
        except UnicodeDecodeError:  # bad fields
            pass

    return out


class HashDB(FileDbWithId):
    @classmethod
    def derived(cls):
        return [
            ("file_sha256", "BLOB", "?"),
            ("image_sha256", "BLOB", "?"),
            ("phash", "BLOB", "?"),
            ("width", "INTEGER", "?"),
            ("height", "INTEGER", "?"),
            ("exif", "BLOB", "?"),
            ("icc_profile", "BLOB", "?"),
            ("iptc", "BLOB", "?"),
            ("photoshop", "BLOB", "?"),
        ]

    def __init__(self, path: str) -> None:
        FileDbWithId.__init__(self, path, "picture-hashes")


def group_sorted_pairs(pairs: Iterable[Tuple[T, T]]) -> Iterator[List[T]]:
    return ([first] + [second for _, second in group] for first, group in groupby(pairs, key=itemgetter(0)))


def make_groups_greedy(dups: np.ndarray) -> Iterator[List[int]]:
    idx = np.argsort(dups[:, 0])
    return group_sorted_pairs(dups[idx])


def make_groups(dups: np.ndarray) -> Iterator[Set[int]]:
    G = nx.Graph()
    G.add_edges_from(dups)
    return nx.connected_components(G)


def slice_idx(total: int, batchsize: int) -> Iterator[Tuple[int, int]]:
    if batchsize < 1:
        raise ValueError("batchsize must be >=1")

    for s in range(0, total, batchsize):
        e = min(batchsize, total - s)
        yield s, e


def np_sorted(arr: np.ndarray) -> np.ndarray:
    # fixme: is this faster than: `np.array(sorted(arr, key=lambda x: x.tolist()))`?
    if arr.size == 0:
        return arr
    return arr[np.lexsort(arr.T[::-1])]


class CollectingIterable(Iterable[T]):
    collection: List[T]
    exhausted: bool

    def __init__(self, it: Iterable[T]):
        self.it = it
        self.collection = []
        self.exhausted = False

    def __iter__(self) -> Iterator[T]:
        self.collection = []
        for item in self.it:
            self.collection.append(item)
            yield item
        self.exhausted = True


class MyFraction(Fraction):
    def swap(self) -> "MyFraction":
        return MyFraction(self.denominator, self.numerator)

    def limit_denominator(self, max_denominator: int) -> "MyFraction":
        return MyFraction(super().limit_denominator(max_denominator))

    def limit_numerator(self, max_numerator: int) -> "MyFraction":
        if self.numerator == 0:
            return MyFraction(self)
        return self.swap().limit_denominator(max_numerator).swap()

    def is_larger_one(self) -> bool:
        return self.numerator > self.denominator

    def is_one(self) -> bool:
        return self.numerator == self.denominator

    def is_less_one(self) -> bool:
        return self.numerator < self.denominator

    def limit(self, expmin: int, expmax: int) -> "MyFraction":
        out = self
        if self.is_larger_one():
            for exp in range(expmax, expmin - 1, -1):
                try:
                    new = out.limit_numerator(10**exp)
                except ZeroDivisionError:
                    return out
                if new.denominator == 0:
                    return out
                out = new
        else:
            for exp in range(expmax, expmin - 1, -1):
                new = out.limit_denominator(10**exp)
                if new.numerator == 0:
                    return out
                out = new
        return out


def cumsum(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    out[0] = 0
    out[1:] = np.cumsum(arr[:-1])
    return out


def show_in_file_manager(path: str) -> None:
    if platform.system() == "Windows":
        path = to_dos_path(path)
        args = f'explorer /select,"{path}"'
        subprocess.run(args)  # nosec
    else:
        raise RuntimeError("Önly windows implemented")


def open_using_default_app(path: str) -> None:
    if platform.system() == "Windows":  # Windows
        os.startfile(path)  # nosec
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", path])  # nosec
    else:  # Linux variants
        subprocess.call(["xdg-open", path])  # nosec


class QueueListenerContext(logging.handlers.QueueListener):
    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class ThreadRichHandler(RichHandler):
    def __init__(self, *args, verbose: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def render(
        self, *, record: logging.LogRecord, traceback: Optional[Traceback], message_renderable: ConsoleRenderable
    ) -> ConsoleRenderable:
        assert isinstance(message_renderable, RichText)
        if self.verbose:
            highlighter = getattr(record, "highlighter", self.highlighter)
            info = f"P={record.process:<5} T={record.thread:<5} "
            if highlighter:
                message = highlighter(info)
            else:
                message = RichText(info)
            message.append_text(message_renderable)
            message_renderable = message
        return super().render(record=record, traceback=traceback, message_renderable=message_renderable)


class MultiprocessingProcess(multiprocessing.get_context().Process):
    def run(self) -> None:
        try:
            super().run()
        except SystemExit:
            raise
        except KeyboardInterrupt:
            thread = threading.get_ident()
            logging.debug("KeyboardInterrupt in process %s thread %s ", self.name, thread)
            sys.exit(1)
        except BaseException as e:
            thread = threading.get_ident()
            logging.exception("%s in process %s thread %s ", type(e).__name__, self.name, thread)
            sys.exit(1)


def rich_sys_excepthook(type, value, traceback) -> None:
    name = multiprocessing.current_process().name
    exc_info = (type, value, traceback)
    logging.error("%s ignored in %s", type.__name__, name, exc_info=exc_info)


def rich_threading_excepthook(args: threading.ExceptHookArgs) -> None:
    proc = multiprocessing.current_process()
    exc_info = (args.exc_type, args.exc_value, args.exc_traceback)
    threadname = None if args.thread is None else args.thread.name
    logging.error("Thread %s in process %s interrupted", threadname, proc.name, exc_info=exc_info)


def rich_sys_unraisablehook(unraisable) -> None:
    """exc_type: Exception type.
    exc_value: Exception value, can be None.
    exc_traceback: Exception traceback, can be None.
    err_msg: Error message, can be None.
    object: Object causing the exception, can be None."""

    exc_info = (unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)
    if isinstance(unraisable.exc_value, KeyboardInterrupt):
        err_msg = unraisable.err_msg or "KeyboardInterrupt ignored in"
        logging.debug("%s: %r", err_msg, unraisable.object)
    else:
        err_msg = unraisable.err_msg or "Exception ignored in"
        logging.error("%s: %r", err_msg, unraisable.object, exc_info=exc_info)
