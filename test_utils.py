import math
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from genutility.test import parametrize

from utils import Max, hamming_duplicates_chunk, make_datetime, pd_sort_groups_by_first_row, with_stem


class TestUtils(unittest.TestCase):
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

    @parametrize(
        (b"2000:01:01 00:00:00", None, None, datetime.fromisoformat("2000-01-01T00:00:00")),
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


if __name__ == "__main__":
    unittest.main()
