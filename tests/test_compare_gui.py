from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd

from picturetool.compare_gui import GroupedPictureModel, QtCore, TableWidget


class CompareGuiTest(TestCase):
    def test_priorities_follow_rows_for_unequal_groups(self):
        for groups, expected in [
            ([2, 2, 2, 1, 1], [0, 1, 2, 0, 1]),
            ([5, 2, 2, 9, 9, 9], [0, 0, 1, 0, 1, 2]),
            ([1, 1, 2, 2, 2], [0, 1, 0, 1, 2]),
        ]:
            with self.subTest(groups=groups):
                model = GroupedPictureModel()
                paths = [f"{i}.jpg" for i in range(len(groups))]
                model.load_df(pd.DataFrame({"group": groups, "path": paths}))
                self.assertEqual(model.df["priority"].tolist(), expected)
                self.assertEqual(str(model.df["priority"].dtype), "int32")
                self.assertEqual(model.df["path"].tolist(), paths)

    def test_reference_unchecking_uses_priority(self):
        model = GroupedPictureModel()
        model.load_df(
            pd.DataFrame(
                {
                    "group": [2, 2, 2, 1, 1],
                    "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
                    "priority": [1, 0, 2, 0, 1],
                    "checked": [True] * 5,
                }
            )
        )
        self.assertEqual(model.df["checked"].tolist(), [True, False, True, False, True])
        self.assertEqual(model.df["priority"].tolist(), [1, 0, 2, 0, 1])

    def test_sorted_csv_round_trip_preserves_selection(self):
        model = GroupedPictureModel()
        model.load_df(
            pd.DataFrame(
                {
                    "group": [1, 1, 2, 2, 2],
                    "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
                    "mod_date": pd.to_datetime(["2026-09-11"] * 5),
                }
            )
        )
        model.set_checked(4, True)
        model.sort(0, QtCore.Qt.DescendingOrder)
        with TemporaryDirectory() as folder:
            path = str(Path(folder) / "sorted.csv")
            TableWidget.to_file(SimpleNamespace(model=model), path)
            target = SimpleNamespace(model=GroupedPictureModel(), assume_input_timezone="utc", picture_window=Mock())
            self.assertEqual(TableWidget.read_file(target, path), (5, 2))
        self.assertEqual(target.model.df.index.tolist(), [2, 2, 2, 1, 1])
        self.assertEqual(target.model.df["checked"].tolist(), [False, False, True, False, False])
        self.assertEqual(target.model.df["priority"].tolist(), [0, 1, 2, 0, 1])

    def test_single_file_group_access(self):
        model = GroupedPictureModel()
        self.assertEqual(
            model.load_df(pd.DataFrame({"group": [1], "path": ["a.jpg"], "width": [32], "height": [16]})), (1, 1)
        )
        self.assertIsInstance(model.get_group(1), pd.DataFrame)
        pd.testing.assert_frame_equal(model.get_group_by_row_idx(0), model.get_group(1))
        self.assertEqual(model.get_group_max_size(1), (32, 16))
        model.set_col_reference(0, "path")
        self.assertEqual(model.get(0)["overwrite"], {"path": "a.jpg"})
        model.set_checked_by_file(1, "a.jpg", False)
        self.assertEqual(model.get(0)["priority"], 0)
        for mode in (
            "next-in-group",
            "prev-in-group",
            "first-in-group",
            "last-in-group",
            "next-group",
            "prev-group",
            "first-group",
            "last-group",
        ):
            with self.subTest(mode=mode), self.assertRaises(IndexError):
                model.get_file(1, "a.jpg", mode)

    def test_navigation_between_single_and_multiple_file_groups(self):
        model = GroupedPictureModel()
        model.load_df(pd.DataFrame({"group": [1, 2, 2, 3], "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]}))
        for group, path, mode, expected in [
            (1, "a.jpg", "next-group", (2, "b.jpg")),
            (2, "b.jpg", "prev-group", (1, "a.jpg")),
            (2, "c.jpg", "next-group", (3, "d.jpg")),
            (3, "d.jpg", "prev-group", (2, "b.jpg")),
            (3, "d.jpg", "first-group", (1, "a.jpg")),
            (1, "a.jpg", "last-group", (3, "d.jpg")),
        ]:
            with self.subTest(group=group, mode=mode):
                self.assertEqual(model.get_file(group, path, mode)[:2], expected)

    def test_single_file_group_round_trip(self):
        model = GroupedPictureModel()
        model.load_df(pd.DataFrame({"group": [1], "path": ["a.jpg"], "mod_date": pd.to_datetime(["2026-09-11"])}))
        model.set_overwrite(0, "Path", "renamed.jpg")
        with TemporaryDirectory() as folder:
            for suffix in ("csv", "json", "parquet"):
                with self.subTest(format=suffix):
                    path = str(Path(folder) / f"single.{suffix}")
                    TableWidget.to_file(SimpleNamespace(model=model), path)
                    target = SimpleNamespace(
                        model=GroupedPictureModel(), assume_input_timezone="utc", picture_window=Mock()
                    )
                    self.assertEqual(TableWidget.read_file(target, path), (1, 1))
                    self.assertEqual(target.model.df.index.tolist(), [1])
                    self.assertEqual(target.model.df["path"].tolist(), ["a.jpg"])
                    self.assertEqual(target.model.df["priority"].tolist(), [0])
                    self.assertEqual(target.model.df["overwrite"].tolist(), [{"Path": "renamed.jpg"}])

    def test_overwrite_round_trip(self):
        model = GroupedPictureModel()
        model.load_df(
            pd.DataFrame(
                {"group": [1, 1], "path": ["a.jpg", "b.jpg"], "mod_date": pd.to_datetime(["2026-09-11", "2026-09-11"])}
            )
        )
        values = [{"Path": "a's.jpg"}, {"Path": "b.jpg"}]
        for row, value in enumerate(values):
            model.set_overwrite(row, "Path", value["Path"])
        source = SimpleNamespace(model=model)
        with TemporaryDirectory() as folder:
            for suffix in ("csv", "json", "parquet"):
                with self.subTest(format=suffix):
                    path = str(Path(folder) / f"saved.{suffix}")
                    TableWidget.to_file(source, path)
                    target = SimpleNamespace(
                        model=GroupedPictureModel(), assume_input_timezone="utc", picture_window=Mock()
                    )
                    self.assertEqual(TableWidget.read_file(target, path), (2, 1))
                    self.assertEqual(target.model.df["overwrite"].tolist(), values)
                    self.assertEqual(target.model.df["path"].tolist(), ["a.jpg", "b.jpg"])

    def make_model(self, groups):
        model = GroupedPictureModel()
        model.load_df(
            pd.DataFrame(
                {
                    "group": [group for group in groups for _ in range(3)],
                    "path": [f"{group}-{i}.jpg" for group in groups for i in range(3)],
                }
            )
        )
        return model

    def test_get_file_all_navigation_modes(self):
        for groups in [(0, 1, 2), (1, 4, 9), (9, 4, 1), (5, 2, 9)]:
            first, middle, last = groups
            model = self.make_model(groups)
            cases = [
                ("next-in-group", middle, 1, middle, 2),
                ("prev-in-group", middle, 1, middle, 0),
                ("first-in-group", middle, 2, middle, 0),
                ("last-in-group", middle, 0, middle, 2),
                ("next-group", middle, 1, last, 0),
                ("prev-group", middle, 1, first, 0),
                ("first-group", last, 2, first, 0),
                ("last-group", first, 2, last, 0),
            ]
            before = model.df.copy(deep=True)
            for mode, group, file_idx, expected_group, expected_idx in cases:
                with self.subTest(groups=groups, mode=mode):
                    result = model.get_file(group, f"{group}-{file_idx}.jpg", mode)
                    self.assertEqual(
                        result[:4], (expected_group, f"{expected_group}-{expected_idx}.jpg", False, expected_idx)
                    )
                    expected_row = model.df.loc[expected_group].iloc[expected_idx]
                    pd.testing.assert_series_equal(result[4], expected_row)
            pd.testing.assert_frame_equal(model.df, before)

    def test_get_file_boundaries(self):
        for groups in [(0,), (0, 1, 2), (5, 2, 9)]:
            model = self.make_model(groups)
            # Exercise both table edges and boundaries between adjacent groups.
            for group in groups:
                for mode, file_idx in [
                    ("prev-in-group", 0),
                    ("next-in-group", 2),
                    ("first-in-group", 0),
                    ("last-in-group", 2),
                ]:
                    with self.subTest(groups=groups, group=group, mode=mode), self.assertRaises(IndexError):
                        model.get_file(group, f"{group}-{file_idx}.jpg", mode)
            for mode, group in [
                ("prev-group", groups[0]),
                ("next-group", groups[-1]),
                ("first-group", groups[0]),
                ("last-group", groups[-1]),
            ]:
                with self.subTest(groups=groups, mode=mode), self.assertRaises(IndexError):
                    model.get_file(group, f"{group}-1.jpg", mode)

    def test_get_file_returns_checked_state_and_priority(self):
        model = self.make_model((5, 2, 9))
        model.set_checked(4, True)
        result = model.get_file(2, "2-0.jpg", "next-in-group")
        self.assertEqual(result[:4], (2, "2-1.jpg", True, 1))
        self.assertIs(type(result[0]), int)
        self.assertIs(type(result[2]), bool)
        self.assertIs(type(result[3]), int)
        pd.testing.assert_series_equal(result[4], model.df.iloc[4])

    def test_get_file_rejects_unknown_mode(self):
        model = self.make_model((0,))
        with self.assertRaisesRegex(ValueError, "Invalid mode: unsupported"):
            model.get_file(0, "0-1.jpg", "unsupported")

    def test_previous_image_stops_at_first(self):
        model = GroupedPictureModel()
        model.load_df(pd.DataFrame({"group": [0, 0], "path": ["a.jpg", "b.jpg"]}))
        with self.assertRaises(IndexError):
            model.get_file(0, "a.jpg", "prev-in-group")
        self.assertEqual(model.get_file(0, "b.jpg", "prev-in-group")[1], "a.jpg")

    def test_group_navigation_uses_display_order(self):
        model = GroupedPictureModel()
        # Deliberately nonconsecutive and descending IDs.
        model.load_df(pd.DataFrame({"group": [2, 2, 0, 0], "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]}))
        self.assertEqual(model.get_file(2, "a.jpg", "next-group")[:2], (0, "c.jpg"))
        self.assertEqual(model.get_file(0, "c.jpg", "prev-group")[:2], (2, "a.jpg"))
        with self.assertRaises(IndexError):
            model.get_file(2, "a.jpg", "prev-group")
        with self.assertRaises(IndexError):
            model.get_file(0, "c.jpg", "next-group")
