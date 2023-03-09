import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from genutility.test import MyTestCase, parametrize

from picturetool.utils import (
    Max,
    hamming_duplicates_chunk,
    l2_duplicates_chunk,
    l2squared_duplicates_chunk,
    make_datetime,
    make_groups,
    pd_sort_groups_by_first_row,
    pd_sort_within_group,
    pd_sort_within_group_slow,
    to_datetime,
    with_stem,
)


class TestUtils(MyTestCase):
    def test_pd_sort_groups_by_first_row(self):
        df = pd.DataFrame({"group": [], "value": []}).set_index("group")
        truth = df
        result = pd_sort_groups_by_first_row(df, "group", "value", True)
        pd.testing.assert_frame_equal(truth, result)

        df = pd.DataFrame({"group": [1, 1, 2, 2], "value": [1, 2, 3, 4]}).set_index("group")

        truth = pd.DataFrame({"group": [1, 1, 2, 2], "value": [1, 2, 3, 4]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", True)
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame({"group": [2, 2, 1, 1], "value": [3, 4, 1, 2]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", False)
        pd.testing.assert_frame_equal(truth, result)

        df = pd.DataFrame({"group": [1, 1, 2, 2], "value": [2, 1, 4, 3]}).set_index("group")

        truth = pd.DataFrame({"group": [1, 1, 2, 2], "value": [2, 1, 4, 3]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", True)
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame({"group": [2, 2, 1, 1], "value": [4, 3, 2, 1]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", False)
        pd.testing.assert_frame_equal(truth, result)

    def test_pd_sort_groups_by_first_row_stable(self):
        df = pd.DataFrame({"group": [1, 2, 3, 4], "value": [1, 1, 2, 1]}).set_index("group")

        truth = pd.DataFrame({"group": [1, 2, 4, 3], "value": [1, 1, 1, 2]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", True, "stable")
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame({"group": [3, 1, 2, 4], "value": [2, 1, 1, 1]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", False, "stable")
        pd.testing.assert_frame_equal(truth, result)

    def test_pd_sort_groups_by_first_row_not_stable(self):
        df = pd.DataFrame({"group": [1, 2, 3, 4], "value": [1, 1, 2, 1]}).set_index("group")

        truth = pd.DataFrame({"group": [2, 1, 4, 3], "value": [1, 1, 1, 2]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", True, "heapsort")
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame({"group": [3, 1, 2, 4], "value": [2, 1, 1, 1]}).set_index("group")
        result = pd_sort_groups_by_first_row(df, "group", "value", False, "heapsort")
        pd.testing.assert_frame_equal(truth, result)

    def test_pd_sort_within_group_index(self):
        df = pd.DataFrame({"group": [1, 1, 1, 2], "col1": [1, 1, 2, 2], "col2": [1, 2, 3, 4]}).set_index("group")

        truth = df
        result = pd_sort_within_group(df, "group", [])
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [])
        pd.testing.assert_frame_equal(truth, result)

        truth = df
        result = pd_sort_within_group(df, "group", [{"by": "col1", "ascending": True}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [{"by": "col1", "ascending": True}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame({"group": [1, 1, 1, 2], "col1": [2, 1, 1, 2], "col2": [3, 1, 2, 4]}).set_index("group")
        result = pd_sort_within_group(df, "group", [{"by": "col1", "ascending": False}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [{"by": "col1", "ascending": False}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame({"group": [1, 1, 1, 2], "col1": [1, 1, 2, 2], "col2": [2, 1, 3, 4]}).set_index("group")
        result = pd_sort_within_group(
            df, "group", [{"by": "col1", "ascending": True}, {"by": "col2", "ascending": False}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(
            df, "group", [{"by": "col1", "ascending": True}, {"by": "col2", "ascending": False}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame({"group": [1, 1, 1, 2], "col1": [2, 1, 1, 2], "col2": [3, 2, 1, 4]}).set_index("group")
        result = pd_sort_within_group(
            df, "group", [{"by": "col2", "ascending": False}, {"by": "col1", "ascending": True}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(
            df, "group", [{"by": "col2", "ascending": False}, {"by": "col1", "ascending": True}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)

    def test_pd_sort_within_group_column(self):
        df = pd.DataFrame({"group": [1, 1, 1, 2], "col1": [1, 1, 2, 2], "col2": [1, 2, 3, 4]})

        truth = df
        result = pd_sort_within_group(df, "group", [])
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [])
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame(
            {None: [0, 1, 2, 3], "group": [1, 1, 1, 2], "col1": [1, 1, 2, 2], "col2": [1, 2, 3, 4]}
        ).set_index(None)
        result = pd_sort_within_group(df, "group", [{"by": "col1", "ascending": True}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [{"by": "col1", "ascending": True}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame(
            {None: [2, 0, 1, 3], "group": [1, 1, 1, 2], "col1": [2, 1, 1, 2], "col2": [3, 1, 2, 4]}
        ).set_index(None)
        result = pd_sort_within_group(df, "group", [{"by": "col1", "ascending": False}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [{"by": "col1", "ascending": False}], sort_groups=False)
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame(
            {None: [1, 0, 2, 3], "group": [1, 1, 1, 2], "col1": [1, 1, 2, 2], "col2": [2, 1, 3, 4]}
        ).set_index(None)
        result = pd_sort_within_group(
            df, "group", [{"by": "col1", "ascending": True}, {"by": "col2", "ascending": False}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(
            df, "group", [{"by": "col1", "ascending": True}, {"by": "col2", "ascending": False}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)

        truth = pd.DataFrame(
            {None: [2, 1, 0, 3], "group": [1, 1, 1, 2], "col1": [2, 1, 1, 2], "col2": [3, 2, 1, 4]}
        ).set_index(None)
        result = pd_sort_within_group(
            df, "group", [{"by": "col2", "ascending": False}, {"by": "col1", "ascending": True}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(
            df, "group", [{"by": "col2", "ascending": False}, {"by": "col1", "ascending": True}], sort_groups=False
        )
        pd.testing.assert_frame_equal(truth, result)

    def test_pd_sort_within_group_index_sorted(self):
        df = pd.DataFrame(
            {"group": [2, 1, 1, 1, 3, 3], "col1": [1, 1, 2, 2, 3, 3], "col2": [6, 5, 4, 3, 2, 1]}
        ).set_index("group")

        truth = pd.DataFrame(
            {"group": [1, 1, 1, 2, 3, 3], "col1": [1, 2, 2, 1, 3, 3], "col2": [5, 4, 3, 6, 2, 1]}
        ).set_index("group")
        result = pd_sort_within_group(df, "group", [], sort_groups=True)
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [], sort_groups=True)
        pd.testing.assert_frame_equal(truth, result)

    def test_pd_sort_within_group_column_sorted(self):
        df = pd.DataFrame({"group": [2, 1, 1, 1, 3, 3], "col1": [1, 1, 2, 2, 3, 3], "col2": [6, 5, 4, 3, 2, 1]})

        truth = pd.DataFrame(
            {
                None: [1, 2, 3, 0, 4, 5],
                "group": [1, 1, 1, 2, 3, 3],
                "col1": [1, 2, 2, 1, 3, 3],
                "col2": [5, 4, 3, 6, 2, 1],
            }
        ).set_index(None)
        result = pd_sort_within_group(df, "group", [], sort_groups=True)
        pd.testing.assert_frame_equal(truth, result)
        result = pd_sort_within_group_slow(df, "group", [], sort_groups=True)
        pd.testing.assert_frame_equal(truth, result)

    @parametrize(
        (b"2000:01:01 00:00:00", None, None, datetime.fromisoformat("2000-01-01T00:00:00")),
        (b"2000-01-01 00:00:00", None, None, datetime.fromisoformat("2000-01-01T00:00:00")),
        (b"2000:01:01 00:00:00\0", None, None, datetime.fromisoformat("2000-01-01T00:00:00")),
        (b"2000:01:01 00:00:00", b"", None, datetime.fromisoformat("2000-01-01T00:00:00")),
        (b"2000:01:01 00:00:00", b"123", None, datetime.fromisoformat("2000-01-01T00:00:00.123")),
        (b"2000:01:01 00:00:00", b"123456", None, datetime.fromisoformat("2000-01-01T00:00:00.123456")),
        (b"2000:01:01 00:00:00", b"123\0", None, datetime.fromisoformat("2000-01-01T00:00:00.123")),
        (b"2000:01:01 00:00:00", b"123", b"+0100", datetime.fromisoformat("2000-01-01T00:00:00.123+01:00")),
        (b"2000:01:01 00:00:00", None, b"+0100\0", datetime.fromisoformat("2000-01-01T00:00:00+01:00")),
    )
    def test_make_datetime(self, date, subsec, offset, truth):
        result = make_datetime(date, subsec, offset)
        self.assertEqual(truth, result)

    @parametrize(
        (b"2000:01:01 00:00:00\xff", None, None, UnicodeDecodeError),
        (b"2000:02:30 00:00:00", None, None, ValueError),
        (b"0:0:0 0:0:0", None, None, ValueError),
        (b"", None, None, ValueError),
        (b"asd", None, None, ValueError),
    )
    def test_make_datetime_fail(self, date, subsec, offset, exception):
        with self.assertRaises(exception):
            make_datetime(date, subsec, offset)

    def test_max(self):
        for val in [None, 0, math.inf]:
            self.assertLess(val, Max)
            self.assertGreater(Max, val)
            self.assertNotEqual(val, Max)

        self.assertEqual(Max, Max)

        d = {Max: None, None: None, 0: None, math.inf: None}
        self.assertEqual(4, len(d))

        self.assertEqual([0, Max], sorted([Max, 0]))
        self.assertEqual([None, Max], sorted([Max, None]))
        self.assertEqual([math.inf, Max], sorted([Max, math.inf]))

    @parametrize(
        (Path("asd"), "qwe", Path("qwe")),
        (Path("asd.ext"), "qwe", Path("qwe.ext")),
        (Path("dir/asd.ext"), "qwe", Path("dir/qwe.ext")),
    )
    def test_with_stem(self, path, stem, truth):
        result = with_stem(path, stem)
        self.assertEqual(truth, result)

    @parametrize(
        (Path("."), "qwe"),
    )
    def test_with_stem_fail(self, path, stem):
        with self.assertRaises(ValueError):
            with_stem(path, stem)

    def test_hamming_duplicates_chunk(self):
        arr = np.packbits([[0, 0], [0, 1], [1, 0], [1, 1]], axis=-1, bitorder="little")
        a, b = np.broadcast_arrays(arr[None, :, :], arr[:, None, :])

        truth = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        result = hamming_duplicates_chunk(a, b, (0, 0), -1, 2)
        np.testing.assert_array_equal(truth, result)

        truth = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        result = hamming_duplicates_chunk(a, b, (0, 0), -1, 1)
        np.testing.assert_array_equal(truth, result)

    def test_l2_duplicates_chunk(self):
        arr = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        a = arr[None, :, :]
        b = arr[:, None, :]

        truth = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        result = l2_duplicates_chunk(a, b, 2)
        np.testing.assert_array_equal(truth, result)

        truth = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        result = l2_duplicates_chunk(a, b, 1)
        np.testing.assert_array_equal(truth, result)

    def test_l2squared_duplicates_chunk(self):
        arr = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        a = arr[None, :, :]
        b = arr[:, None, :]

        truth = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        result = l2squared_duplicates_chunk(a, b, 2.1)
        np.testing.assert_array_equal(truth, result)

        truth = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        result = l2squared_duplicates_chunk(a, b, 1.1)
        np.testing.assert_array_equal(truth, result)

    def test_to_datetime(self):
        s_aware1 = "2000-01-01T00:00:00.000000+04:00"
        s_aware2 = "2000-01-01T00:00:00.000001+00:00"
        s_naive = "2000-01-01T00:00:00"
        ds = pd.Series([s_aware1, s_aware2, s_naive, ""])

        in_tz = timezone.utc
        truth = pd.to_datetime(
            pd.Series(["1999-12-31 20:00:00", "2000-01-01 00:00:00.000001", "2000-01-01 00:00:00", ""])
        )
        result = to_datetime(ds, in_tz, timezone.utc)
        pd.testing.assert_series_equal(truth, result)

        in_tz = timezone(timedelta(hours=10), name="test")
        truth = pd.to_datetime(
            pd.Series(["1999-12-31 20:00:00", "2000-01-01 00:00:00.000001", "1999-12-31 14:00:00", ""])
        )
        result = to_datetime(ds, in_tz, timezone.utc)
        pd.testing.assert_series_equal(truth, result)

    def test_make_groups(self):
        truth = [{1, 2, 3}, {4, 5}]
        arr = np.array([[1, 2], [1, 3], [2, 3], [4, 5]])
        result = make_groups(arr)
        self.assertIterEqual(truth, result)


if __name__ == "__main__":
    import unittest

    unittest.main()
