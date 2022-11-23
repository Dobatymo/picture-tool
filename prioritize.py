import fnmatch
import re

import numpy as np
import pandas as pd
from pandas.core.arrays.integer import Int32Dtype, Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype


def str_bool(col: pd.Series) -> pd.Series:
    return col.str.len() > 0


def str_length(col: pd.Series) -> pd.Series:
    return col.str.len()


def str_match(col: pd.Series, pattern: str) -> pd.Series:
    try:
        return ~col.str.match(pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Regex error: {e}")


def str_fnmatch(col: pd.Series, fn_pattern: str) -> pd.Series:
    re_pattern = fnmatch.translate(fn_pattern)
    return str_match(col, re_pattern)


def int_bool(col: pd.Series) -> pd.Series:
    return col.isnull()


def int_value(col: pd.Series) -> pd.Series:
    return col


def dt_bool(col: pd.Series) -> pd.Series:
    return col.isnull()


def dt_value(col: pd.Series) -> pd.Series:
    return col


np_int32 = type(np.dtype("int32"))
np_int64 = type(np.dtype("int64"))
np_datetime = type(np.dtype("datetime64"))
pd_string = StringDtype
pd_datetime = DatetimeTZDtype
pd_int32 = Int32Dtype
pd_int64 = Int64Dtype

functions = {
    (pd_string, "Available"): str_bool,
    (pd_string, "Length"): str_length,
    (pd_string, "Match regex"): str_match,
    (pd_string, "Match wildcards"): str_fnmatch,
    (np_int32, "Available"): int_bool,
    (np_int64, "Available"): int_bool,
    (pd_int32, "Available"): int_bool,
    (pd_int64, "Available"): int_bool,
    (np_int32, "Value"): int_value,
    (np_int64, "Value"): int_value,
    (pd_int32, "Value"): int_value,
    (pd_int64, "Value"): int_value,
    (np_datetime, "Available"): dt_bool,
    (np_datetime, "Value"): dt_value,
    (pd_datetime, "Available"): dt_bool,
    (pd_datetime, "Value"): dt_value,
}
