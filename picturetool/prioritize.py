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


def str_match(col: pd.Series, re_pattern: str) -> pd.Series:
    try:
        return ~col.str.match(re_pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Regex error: {e}")


def str_count(col: pd.Series, re_pattern: str) -> pd.Series:
    try:
        return col.str.count(re_pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Regex error: {e}")


def str_fnmatch(col: pd.Series, fn_pattern: str) -> pd.Series:
    re_pattern = fnmatch.translate(fn_pattern)
    return str_match(col, re_pattern)


def t_value(col: pd.Series) -> pd.Series:
    return col


def t_bool(col: pd.Series) -> pd.Series:
    return col.isnull()


np_int32 = type(np.dtype("int32"))
np_int64 = type(np.dtype("int64"))
np_bool = type(np.dtype("bool_"))
np_datetime = type(np.dtype("datetime64"))
pd_string = StringDtype
pd_datetime = DatetimeTZDtype
pd_int32 = Int32Dtype
pd_int64 = Int64Dtype

functions = {
    (pd_string, "Available"): str_bool,
    (pd_string, "Alphabetical"): t_value,
    (pd_string, "Length"): str_length,
    (pd_string, "Match regex"): str_match,
    (pd_string, "Match wildcards"): str_fnmatch,
    (pd_string, "Count regex"): str_count,
    (np_int32, "Available"): t_bool,
    (np_int64, "Available"): t_bool,
    (pd_int32, "Available"): t_bool,
    (pd_int64, "Available"): t_bool,
    (np_int32, "Value"): t_value,
    (np_int64, "Value"): t_value,
    (pd_int32, "Value"): t_value,
    (pd_int64, "Value"): t_value,
    (np_datetime, "Available"): t_bool,
    (np_datetime, "Value"): t_value,
    (pd_datetime, "Available"): t_bool,
    (pd_datetime, "Value"): t_value,
    (np_bool, "Value"): t_value,
}
