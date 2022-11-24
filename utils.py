from datetime import datetime, timedelta, tzinfo
from functools import total_ordering
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, TypedDict

import numpy as np
import pandas as pd
import piexif
from dateutil import tz
from genutility.datetime import is_aware
from genutility.numpy import hamming_dist_packed
from pandas._typing import ValueKeyFunc


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


def hamming_duplicates_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, coords: Tuple[int, ...], axis: int, hamming_threshold: int
) -> np.ndarray:
    dists = hamming_dist_packed(a_arr, b_arr, axis)
    return np.argwhere(np.triu(dists <= hamming_threshold, 1)) + np.array(coords)


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

    idx = df.groupby(group_by_column).nth(0).sort_values(sort_by_column, ascending=ascending, kind=kind, key=key).index
    return df.loc[idx]


def pd_sort_within_group(
    df: pd.DataFrame,
    group_by_column: str,
    sort_by_column: str,
    ascending: bool = True,
    kind: str = "stable",
    key: ValueKeyFunc = None,
) -> pd.DataFrame:
    """Sort rows within groups, leave the order of the groups intact."""

    return df.groupby(group_by_column, group_keys=False).apply(
        lambda x: x.sort_values(
            sort_by_column, ascending=ascending, inplace=False, kind=kind, ignore_index=False, key=key
        )
    )


class SortValuesKwArgs(TypedDict):
    by: str
    ascending: bool
    key: ValueKeyFunc


def pd_sort_within_group_multiple(df: pd.DataFrame, group_by_column: str, sort_kwargs: Sequence[SortValuesKwArgs]):
    """Sort rows within groups, leave the order of the groups intact.
    Specify a list of multiple sorting arguments which are applied in order.
    """

    def multisort(x: pd.DataFrame) -> pd.DataFrame:
        for kwargs in sort_kwargs:
            x = x.sort_values(kind="stable", inplace=False, ignore_index=False, **kwargs)
        return x

    return df.groupby(group_by_column, group_keys=False).apply(multisort)


def make_datetime(date: bytes, subsec: Optional[bytes] = None, offset: Optional[bytes] = None) -> datetime:
    date_str = date.rstrip(b"\0").decode("ascii")

    if offset:
        offset_str = offset.rstrip(b"\0").decode("ascii")
        dt = datetime.strptime(f"{date_str} {offset_str}", "%Y:%m:%d %H:%M:%S %z")
    else:
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")

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

    m1 = {
        "ImageIFD": "0th",
        "ExifIFD": "Exif",
    }

    m2 = {
        "ImageIFD": piexif.ImageIFD,
        "ExifIFD": piexif.ExifIFD,
    }

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
