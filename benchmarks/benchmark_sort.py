import numpy as np
import pandas as pd
from genutility.time import PrintStatementTime

from picturetool.utils import pd_sort_within_group, pd_sort_within_group_slow

a = np.random.randint(0, 10000, (50000, 10))
df = pd.DataFrame(a, columns=[f"col{i}" for i in range(10)]).set_index("col0").astype("string")
# df = df.sort_index()

with PrintStatementTime():
    df_1 = pd_sort_within_group(df, "col0", [{"by": "col1"}])

with PrintStatementTime():
    df_2 = pd_sort_within_group_slow(df, "col0", [{"by": "col1"}])

pd.testing.assert_frame_equal(df_1, df_2)

with PrintStatementTime():
    df_1 = pd_sort_within_group(df, "col0", [{"by": "col1"}, {"by": "col2"}])

with PrintStatementTime():
    df_2 = pd_sort_within_group_slow(df, "col0", [{"by": "col1"}, {"by": "col2"}])

pd.testing.assert_frame_equal(df_1, df_2)

df_copy = df.copy()
with PrintStatementTime():
    df_copy.sort_values("col1", kind="stable", inplace=True)

with PrintStatementTime():
    df.sort_values("col1", kind="stable", inplace=False)

with PrintStatementTime():
    df.groupby("col0", group_keys=False).count()

with PrintStatementTime():
    df.groupby("col0", group_keys=False).apply(lambda df: df)
