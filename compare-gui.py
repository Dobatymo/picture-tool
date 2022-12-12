# pip install pandas pillow pillow-heif PySide2

import json
import logging
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import timedelta, timezone
from functools import lru_cache
from inspect import signature
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple

import humanize
import numpy as np
import pandas as pd
from genutility._files import to_dos_path
from genutility.time import MeasureTime
from PIL import Image
from pillow_heif import register_heif_opener
from PySide2 import QtCore, QtGui, QtWidgets

from prioritize import functions
from utils import SortValuesKwArgs, pd_sort_groups_by_first_row, pd_sort_within_group, to_datetime

register_heif_opener()

APP_NAME = "compare-gui"
PIC_CACHE_SIZE = 4


def cumsum(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    out[0] = 0
    out[1:] = np.cumsum(arr[:-1])
    return out


def get_priorities(df: pd.DataFrame) -> np.ndarray:
    return pd.Series(
        list(chain.from_iterable(range(i) for i in df.groupby("group").count()["path"])),
        dtype="int32",
    ).values


def set_reference_unchecked(df: pd.DataFrame) -> None:
    idx = cumsum(df.groupby("group").count()["path"].values)
    df.iloc[idx, df.columns.get_loc("checked")] = False


@lru_cache(maxsize=PIC_CACHE_SIZE)
def read_qt_pixmap(path: str) -> QtGui.QPixmap:

    """Uses `pillow` to read a QPixmap from `path`.
    This supports more image formats than Qt directly.
    """

    """ doesn't work...
    # import this one late to use the correct qt lib
    from PIL.ImageQt import toqpixmap
    with Image.open(path) as img:
        return toqpixmap(img)
    """

    modemap = {
        "L": QtGui.QImage.Format_Grayscale8,
        "RGB": QtGui.QImage.Format_RGB888,
        "RGBA": QtGui.QImage.Format_RGBA8888,
        "RGBX": QtGui.QImage.Format_RGBX8888,
        "RGBa": QtGui.QImage.Format_RGBA8888_Premultiplied,
    }

    channelmap = {
        "L": 1,
        "RGB": 3,
        "RGBA": 4,
        "RGBX": 4,
    }

    with Image.open(path) as img:
        if img.mode not in modemap or (img.width * channelmap[img.mode]) % 4 != 0:
            # Unsupported image mode or image scanlines not 32-bit aligned
            img = img.convert("RGBA")

        # Are there any copies created below?
        # Because the img buffer will be freed at the end of the block.
        # Maybe Qt's `cleanupFunction` should be used instead of a context manager.

        qimg_format = modemap[img.mode]
        b = QtCore.QByteArray(img.tobytes())
        qimg = QtGui.QImage(b, img.size[0], img.size[1], qimg_format)
        return QtGui.QPixmap.fromImage(qimg)


def show_in_file_manager(path: str) -> None:

    if platform.system() == "Windows":
        path = to_dos_path(path)
        args = f'explorer /select,"{path}"'
        subprocess.run(args)
    else:
        raise RuntimeError("Önly windows implemented")


def open_using_default_app(path: str) -> None:

    if platform.system() == "Windows":  # Windows
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", path])
    else:  # Linux variants
        subprocess.call(["xdg-open", path])


class AspectRatioPixmapLabel(QtWidgets.QLabel):
    def __init__(
        self,
        fit_to_widget: bool = True,
        fixed_size: Optional[QtCore.QSize] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:

        """If `fit_to_widget` is True, the label will be resized to match the parent widgets size.
        If it's False, it will be resized to the images original size.
        """

        super().__init__(parent)
        self.setMinimumSize(1, 1)
        self.setScaledContents(False)  # we do the scaling ourselves
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.pm: Optional[QtGui.QPixmap] = None

        # set properties
        self.fit_to_widget = fit_to_widget
        self.fixed_size = fixed_size

    def clear(self) -> None:
        super().clear()
        self.pm = None

    def _scaled_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap:
        assert self.pm is not None
        return self.pm.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def resize_pixmap(self, size: QtCore.QSize):
        # don't overwrite `super().resize()` here by accident
        super().setPixmap(self._scaled_pixmap(size))

    def scale(self) -> None:
        assert self.pm is not None
        if self.fixed_size:
            self.resize_pixmap(self.fixed_size)
            self.adjustSize()
        elif self.fit_to_widget:
            self.resize_pixmap(self.size())
        else:
            self.resize_pixmap(self.pm.size())
            self.adjustSize()

    def setPixmap(self, pm: QtGui.QPixmap) -> None:
        self.pm = pm
        self.scale()

    @property
    def fit_to_widget(self) -> bool:
        return self._fit_to_widget

    @fit_to_widget.setter
    def fit_to_widget(self, value: bool) -> None:
        self._fit_to_widget = value
        if self.pm is not None:
            self.scale()

    @property
    def fixed_size(self) -> Optional[QtCore.QSize]:
        return self._fixed_size

    @fixed_size.setter
    def fixed_size(self, value: Optional[QtCore.QSize]) -> None:
        self._fixed_size = value
        if self.pm is not None:
            self.scale()

    # qt virtual

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:

        # The label doesn't get `resizeEvent`s when `QScrollArea.widgetResizable()` is False.
        # `resizeEvent`s are however triggered by the labels `self.adjustSize()`,
        # so when setting a new pixmap, a resize event could still be triggered
        # even if `self.fit_to_widget` is False.

        # print("resizeEvent", self.fit_to_widget, self.fixed_size)

        if self.pm is not None and self.fit_to_widget and self.fixed_size is None:
            self.scale()
        super().resizeEvent(event)


def slice_to_list(value: slice) -> list:
    return list(range(value.stop)[value])


def iloc_by_index_and_bool(df, index: int, bool_idx: pd.Series) -> int:

    if bool_idx.dtype != "bool":
        raise TypeError(f"bool_idx must be a 'bool' series, not '{bool_idx.dtype}'")

    ilocs = np.array(slice_to_list(df.index.get_loc(index)))[bool_idx]
    assert len(ilocs) == 1
    return ilocs[0].item()


class GroupedPictureModel(QtCore.QAbstractTableModel):
    row_checked = QtCore.Signal(int, str, bool)
    row_reference = QtCore.Signal(int, str)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        df = pd.DataFrame({}, columns=["group", "path"])
        self.load_df(df)

    def set_df(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self.df = df
        self.endResetModel()

    def load_df(self, df: pd.DataFrame) -> Tuple[int, int]:
        df = df.set_index("group")

        if "priority" not in df:
            priority = get_priorities(df)
            assert not isinstance(priority, pd.Series), "Cannot assign series because of wrong index"
            df["priority"] = priority

        if "checked" not in df:
            df["checked"] = False
        else:
            set_reference_unchecked(df)

        self.set_df(df)

        num_files = len(df)
        num_groups = len(df.groupby("group").count())
        return num_files, num_groups

    def get(self, row_idx: int) -> pd.Series:
        return self.df.iloc[row_idx]

    def set_checked(self, row_idx: int, checked: bool) -> None:
        col_idx = self.df.columns.get_loc("checked")
        self.df.iat[row_idx, col_idx] = checked
        row = self.df.iloc[row_idx]
        group = row.name.item()
        self.row_checked.emit(group, row["path"], checked)

    def set_reference(self, row_idx: int) -> None:
        row = self.df.iloc[row_idx]
        group = row.name.item()
        path = row["path"]

        self.set_reference_by_file(group, path)
        self.row_reference.emit(group, path)

    def _get_iloc_by(self, group: int, key: str, value: Any) -> int:
        idx = (self.df.loc[group][key] == value).astype("bool")  # without the cast it's 'boolean'
        return iloc_by_index_and_bool(self.df, group, idx)

    def set_checked_by_file(self, group: int, path: str, checked: bool) -> None:
        row = self._get_iloc_by(group, "path", path)
        col = self.df.columns.get_loc("checked")
        self.df.iat[row, col] = checked

        index = self.createIndex(row, col)
        self.dataChanged.emit(index, index)

    def set_reference_by_file(self, group: int, path: str) -> None:
        new_row = self._get_iloc_by(group, "path", path)
        col_prio = self.df.columns.get_loc("priority")
        col_check = self.df.columns.get_loc("checked")
        priority = self.df.iat[new_row, col_prio]
        assert priority != 0, "selected row is already the reference"

        old_row = self._get_iloc_by(group, "priority", 0)

        self.df.iat[new_row, col_prio] = 0
        self.df.iat[old_row, col_prio] = priority

        assert not self.df.iat[new_row, col_check]
        assert not self.df.iat[old_row, col_check]

        self.df.iloc[new_row], self.df.iloc[old_row] = self.df.iloc[old_row], self.df.iloc[new_row]

        index = self.createIndex(new_row, col_prio)
        self.dataChanged.emit(index, index)
        index = self.createIndex(old_row, col_prio)
        self.dataChanged.emit(index, index)

    def get_group_max_size(self, group: int) -> Optional[Tuple[int, int]]:
        w = self.df.loc[group]["width"].max()
        h = self.df.loc[group]["height"].max()
        if w is pd.NA or h is pd.NA:
            return None
        else:
            return w, h

    def get_file(self, group: int, path: str, mode: str) -> Tuple[int, str, bool, int]:
        row = self._get_iloc_by(group, "path", path)
        try:
            if mode == "next-in-group":
                result = self.df.iloc[row + 1]
                if result.name.item() != group:
                    raise IndexError("File was last in group")
            elif mode == "prev-in-group":
                result = self.df.iloc[row - 1]
                if result.name.item() != group:
                    raise IndexError("File was first in group")
            elif mode == "first-in-group":
                result = self.df.loc[group].iloc[0]
                if result["path"] == path:
                    raise IndexError("File was first in group")
            elif mode == "last-in-group":
                result = self.df.loc[group].iloc[-1]
                if result["path"] == path:
                    raise IndexError("File was last in group")
            elif mode == "next-group":
                result = self.df.loc[group + 1].iloc[0]
            elif mode == "prev-group":
                result = self.df.loc[group - 1].iloc[0]
            elif mode == "first-group":
                first_group = self.df.index[0]
                if first_group == group:
                    raise IndexError("File was first in group")
                result = self.df.loc[first_group].iloc[0]
            elif mode == "last-group":
                last_group = self.df.index[-1]
                if last_group == group:
                    raise IndexError("File was first in group")
                result = self.df.loc[last_group].iloc[0]
            else:
                raise ValueError(f"Invalid mode: {mode}")

            return result.name.item(), result["path"], result["checked"].item(), result["priority"].item()

        except IndexError:
            raise IndexError("File was first/last in group")
        except KeyError:
            raise IndexError("Group was first/last in list")

    # required by Qt

    def rowCount(self, index: QtCore.QModelIndex) -> int:
        return self.df.shape[0]

    def columnCount(self, index: QtCore.QModelIndex) -> int:
        return self.df.shape[1]  # plus index, minus priority and checked

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int) -> Optional[str]:
        if orientation == QtCore.Qt.Orientation.Horizontal:
            if section == 0:
                if role == QtCore.Qt.DisplayRole:
                    return self.df.index.name
                elif role == QtCore.Qt.ToolTipRole:
                    return f"Sort by {self.df.index.name}"
                elif role == QtCore.Qt.StatusTipRole:
                    return f"Sort by {self.df.index.name}"
            else:
                if role == QtCore.Qt.DisplayRole:
                    return self.df.columns[section - 1]
                elif role == QtCore.Qt.ToolTipRole:
                    return f"Sort {self.df.index.name} by {self.df.columns[section-1]} of reference file"
                elif role == QtCore.Qt.StatusTipRole:
                    return f"Sort {self.df.index.name} by {self.df.columns[section-1]} of reference file"

        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        if col == 0 and self.df.iat[row, self.df.columns.get_loc("priority")] > 0:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable

        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        if role == QtCore.Qt.DisplayRole:
            try:
                if col == 0:
                    return str(self.df.index[row])
                elif col == 1:
                    return to_dos_path(self.df.iat[row, 0])
                else:
                    cell = self.df.iat[row, col - 1]
                    if pd.isna(cell):
                        return ""
                    else:
                        return str(cell)
            except KeyError:
                logging.warning("Invalid table access at %d, %d", row, col)

        elif role == QtCore.Qt.CheckStateRole:
            if col == 0 and self.df.iat[row, self.df.columns.get_loc("priority")] > 0:
                if self.df.iat[row, self.df.columns.get_loc("checked")]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked

        elif role == QtCore.Qt.TextAlignmentRole:
            if col == 0:
                return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)  # bug

        return None

    def setData(self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.DisplayRole) -> Optional[bool]:
        if not index.isValid():
            return None

        row = index.row()

        if role == QtCore.Qt.CheckStateRole:
            self.set_checked(row, value == QtCore.Qt.Checked)
            return True

        return None

    def sort(self, column: int, order: QtCore.Qt.SortOrder = QtCore.Qt.AscendingOrder) -> None:
        ascending = order == QtCore.Qt.AscendingOrder
        if column == 0:
            self.beginResetModel()
            self.df.sort_index(ascending=ascending, inplace=True, kind="stable")
            self.endResetModel()
        elif column > 0:
            self.beginResetModel()
            self.df = pd_sort_groups_by_first_row(self.df, "group", self.df.columns[column - 1], ascending)
            self.endResetModel()


class GroupedPictureView(QtWidgets.QTableView):
    def __init__(self, wordwrap: bool = False, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSortingEnabled(True)
        self.setWordWrap(wordwrap)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.horizontalHeader().setSectionsMovable(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def get_row_by_index(self, index: QtCore.QModelIndex) -> pd.Series:
        return self.model().get(index.row())

    def open_file(self, index: QtCore.QModelIndex) -> None:
        path = self.get_row_by_index(index)["path"]
        open_using_default_app(path)

    def make_reference(self, index: QtCore.QModelIndex) -> None:
        self.model().set_checked(index.row(), False)
        self.model().set_reference(index.row())

    def open_directory(self, index: QtCore.QModelIndex) -> None:
        path = self.get_row_by_index(index)["path"]
        show_in_file_manager(path)

    # qt virtual

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        index = self.indexAt(event.pos())
        if not index.isValid():
            return

        is_reference = self.get_row_by_index(index)["priority"] == 0

        open_file_action = QtWidgets.QAction("Open file", self)
        open_file_action.triggered.connect(lambda checked: self.open_file(index))

        open_directory_action = QtWidgets.QAction("Open directory", self)
        open_directory_action.triggered.connect(lambda checked: self.open_directory(index))

        reference_action = QtWidgets.QAction("Make reference", self)
        reference_action.setEnabled(not is_reference)
        reference_action.triggered.connect(lambda checked: self.make_reference(index))

        contextMenu = QtWidgets.QMenu(self)
        contextMenu.addAction(open_file_action)
        contextMenu.addAction(open_directory_action)
        contextMenu.addAction(reference_action)
        contextMenu.exec_(event.globalPos())


class PrioritizeWidget(QtWidgets.QWidget):
    prioritize = QtCore.Signal(pd.DataFrame)

    @staticmethod
    def get_dtypefuncs(functions: Dict[Tuple[type, str], Callable]) -> Dict[type, List[str]]:
        dtypefuncs = defaultdict(list)
        for dtype, name in functions.keys():
            dtypefuncs[dtype].append(name)
        return dict(dtypefuncs)

    @staticmethod
    def get_funcsargs(functions: Dict[Tuple[type, str], Callable]) -> Dict[str, int]:
        funcsargs = {(name, len(signature(func).parameters) - 1) for (dtype, name), func in functions.items()}
        out = dict(funcsargs)
        assert len(funcsargs) == len(out)
        return out

    def __init__(self, df, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.df = df

        self.dtypefuncs = self.get_dtypefuncs(functions)  # type: ignore[arg-type]
        self.funcsargs = self.get_funcsargs(functions)  # type: ignore[arg-type]

        self.dropdown1 = QtWidgets.QComboBox(self)
        self.dropdown1.addItems(df.columns)
        self.dropdown1.currentIndexChanged.connect(self.on_dropdown_col_changed)
        self.dropdown2 = QtWidgets.QComboBox(self)
        self.dropdown2.currentIndexChanged.connect(self.on_dropdown_func_change)
        self.lineedit = QtWidgets.QLineEdit(self)
        self.dropdown3 = QtWidgets.QComboBox(self)
        self.dropdown3.addItems(["Ascending", "Descending"])
        self.button_add = QtWidgets.QPushButton("&Add", self)
        self.button_add.clicked.connect(self.on_add)
        self.button_prioritize = QtWidgets.QPushButton("&Prioritize", self)
        self.button_prioritize.clicked.connect(self.on_prioritize)
        self.button_close = QtWidgets.QPushButton("&Close", self)
        self.button_close.clicked.connect(self.on_close)

        self.buttons_top = QtWidgets.QHBoxLayout()
        self.buttons_top.setSpacing(2)
        self.buttons_top.addWidget(self.dropdown1)
        self.buttons_top.addWidget(self.dropdown2)
        self.buttons_top.addWidget(self.lineedit)
        self.buttons_top.addWidget(self.dropdown3)
        self.buttons_top.addWidget(self.button_add)

        self.listview = QtWidgets.QListWidget(self)
        self.listview.setMovement(QtWidgets.QListView.Free)
        self.listview.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listview.setSortingEnabled(False)
        self.listview.setDragEnabled(True)
        self.listview.setDropIndicatorShown(True)
        self.listview.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.buttons_bottom = QtWidgets.QHBoxLayout()
        self.buttons_bottom.setSpacing(2)
        self.buttons_bottom.addWidget(self.button_prioritize)
        self.buttons_bottom.addWidget(self.button_close)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addLayout(self.buttons_top)
        self.layout.addWidget(self.listview)
        self.layout.addLayout(self.buttons_bottom)

        self.on_dropdown_col_changed(0)  # init to first

    def get_commands(self) -> List[Dict[str, Any]]:
        return [json.loads(self.listview.item(item).text()) for item in range(self.listview.count())]

    # callbacks

    def on_prioritize(self) -> None:

        # sorting by multiple keys, ie. using secondary keys to break ambiguities by the first key,
        # requires sorting in reverse key order with a stable sorting algorithm

        sorts_kwargs: List[SortValuesKwArgs] = []

        for command in self.get_commands():
            col = command["column"]
            strfunc = command["function"]
            args = command["args"]
            ascending = True if command["order"] == "Ascending" else False  # noqa: F841

            dtype = type(self.df.dtypes[self.df.columns.get_loc(col)])
            func = functions[(dtype, strfunc)]

            argstr = ", ".join(map(repr, args))
            keyfunc = lambda col: func(col, *args)  # type: ignore[operator] # noqa: E731

            try:
                # df.sort_values() eats exceptions in key function, so test first
                keyfunc(self.df[col])
            except ValueError as e:
                logging.error("Prioritization failed. %s(%s, %s): %s", func.__name__, col, argstr, e)
                break

            sorts_kwargs.append({"by": col, "ascending": ascending, "key": keyfunc})

        else:  # no break
            self.df = pd_sort_within_group(self.df, "group", sorts_kwargs)
            self.df["priority"] = get_priorities(self.df)
            set_reference_unchecked(self.df)
            self.prioritize.emit(self.df)

    def on_close(self) -> None:
        self.parent().close()

    def on_add(self) -> None:
        args = []
        if self.lineedit.isEnabled():
            args.append(self.lineedit.text())

        out = {
            "column": self.dropdown1.currentText(),
            "function": self.dropdown2.currentText(),
            "args": args,
            "order": self.dropdown3.currentText(),
        }
        self.listview.addItem(json.dumps(out))

    def on_dropdown_col_changed(self, index: int) -> None:
        if self.dropdown1.count() > 0:
            dtype = type(self.df.dtypes[index])
            funcs = self.dtypefuncs[dtype]
            self.dropdown2.clear()
            self.dropdown2.addItems(funcs)

    def on_dropdown_func_change(self, index: int) -> None:
        if self.dropdown2.count() > 0:
            self.lineedit.clear()
            func = self.dropdown2.itemText(index)
            args = self.funcsargs[func]
            if args == 0:
                self.lineedit.setEnabled(False)
            else:
                self.lineedit.setEnabled(True)

    # qt virtual

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.Delete):
            for item in self.listview.selectedItems():
                self.listview.takeItem(self.listview.row(item))
            event.accept()
        else:
            event.ignore()


class PrioritizeWindow(QtWidgets.QMainWindow):
    def __init__(
        self, df, parent: Optional[QtWidgets.QWidget] = None, flags: QtCore.Qt.WindowFlags = QtCore.Qt.WindowFlags()
    ) -> None:
        super().__init__(parent, flags)

        self.prioritize_widget = PrioritizeWidget(df, parent)
        self.setCentralWidget(self.prioritize_widget)


class MyScrollArea(QtWidgets.QScrollArea):

    arrow_keys = [QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in self.arrow_keys and event.modifiers() == QtCore.Qt.NoModifier:
            event.ignore()
        else:
            super().keyPressEvent(event)


class PictureWidget(QtWidgets.QWidget):

    picture_checked = QtCore.Signal(int, str, bool)
    picture_reference = QtCore.Signal(int, str)

    load_next_in_group = QtCore.Signal(int, str)
    load_prev_in_group = QtCore.Signal(int, str)
    load_next_group = QtCore.Signal(int, str)
    load_prev_group = QtCore.Signal(int, str)

    load_file_failed = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.button1 = QtWidgets.QPushButton("&Check", self)
        self.button1.setCheckable(True)
        self.button1.clicked[bool].connect(self.on_check)
        self.button2 = QtWidgets.QPushButton("Make &reference", self)
        self.button2.setCheckable(True)
        self.button2.clicked[bool].connect(self.on_make_reference)
        self.label = AspectRatioPixmapLabel()

        self.scroll = MyScrollArea(self)
        self.scroll.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.scroll.setWidget(self.label)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.scroll)

        self.buttons = QtWidgets.QHBoxLayout()
        self.buttons.setContentsMargins(0, 0, 0, 0)
        self.buttons.addWidget(self.button1)
        self.buttons.addWidget(self.button2)
        self.layout.addLayout(self.buttons)

        self.pic_path = None
        self.pic_group = None

        # set properties
        self._fixed_size = None  # set first
        self.fit_to_window = True

    def on_check(self, checked: bool) -> None:
        if self.pic_path is not None:
            self.picture_checked.emit(self.pic_group, self.pic_path, checked)

    def on_make_reference(self, checked: bool) -> None:
        assert checked
        self.button1.setChecked(False)
        self.button1.setEnabled(False)
        self.button2.setEnabled(False)
        if self.pic_path is not None:
            self.picture_checked.emit(self.pic_group, self.pic_path, False)
            self.picture_reference.emit(self.pic_group, self.pic_path)

    @property
    def fit_to_window(self) -> bool:
        return self._fit_to_window

    @fit_to_window.setter
    def fit_to_window(self, value: bool) -> None:
        self._fit_to_window = value
        if self.fixed_size is None:
            self.scroll.setWidgetResizable(value)
            self.label.fit_to_widget = value
        else:
            self.label._fit_to_widget = value

    @property
    def fixed_size(self) -> Optional[Tuple[int, int]]:
        return self._fixed_size

    @fixed_size.setter
    def fixed_size(self, value: Optional[Tuple[int, int]]) -> None:
        self._fixed_size = value
        if value is None:
            self.scroll.setWidgetResizable(self.label.fit_to_widget)
            self.label.fixed_size = None
        else:
            self.scroll.setWidgetResizable(False)
            self.label.fixed_size = QtCore.QSize(*value)

    def check_picture(self, group: int, path: str, checked: bool) -> bool:
        if group == self.pic_group and path == self.pic_path:
            self.button1.setChecked(checked)
            return True
        else:
            return False

    def reference_picture(self, group: int, path: str) -> bool:
        if group == self.pic_group and path == self.pic_path:
            self.button1.setChecked(False)
            self.button1.setEnabled(False)
            self.button2.setChecked(True)
            self.button2.setEnabled(False)
            return True
        else:
            return False

    def load_picture(self, group: int, path: str, checked: bool, reference: bool) -> bool:
        self.pic_group = group
        self.pic_path = path

        try:
            self.pixmap = read_qt_pixmap(path)

            # fixme: This can disable scale normalization if there's no width/high information.
            # There are not GUI indicators for this however.
            if self.fixed_size is not None:  # only update if it was set previously
                self.fixed_size = self.parent().model.get_group_max_size(group)

            self.label.setPixmap(self.pixmap)
            self.button1.setChecked(checked)
            self.button1.setEnabled(not reference)
            self.button2.setChecked(reference)
            self.button2.setEnabled(not reference)
            return True
        except (FileNotFoundError, ValueError) as e:
            self.load_file_failed.emit(f"{type(e).__name__}: {e}")
            self.label.clear()
            self.button1.setChecked(checked)
            self.button1.setEnabled(False)
            self.button2.setChecked(reference)
            self.button2.setEnabled(False)
            return False

    # qt virtual

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.MoveToNextChar):
            self.load_next_in_group.emit(self.pic_group, self.pic_path)
            event.accept()
        elif event.matches(QtGui.QKeySequence.MoveToPreviousChar):
            self.load_prev_in_group.emit(self.pic_group, self.pic_path)
            event.accept()
        elif event.matches(QtGui.QKeySequence.MoveToNextLine):
            self.load_next_group.emit(self.pic_group, self.pic_path)
            event.accept()
        elif event.matches(QtGui.QKeySequence.MoveToPreviousLine):
            self.load_prev_group.emit(self.pic_group, self.pic_path)
            event.accept()
        else:
            event.ignore()


class PictureWindow(QtWidgets.QMainWindow):

    request_on_top = QtCore.Signal(bool)

    def __init__(
        self, parent: Optional[QtWidgets.QWidget] = None, flags: QtCore.Qt.WindowFlags = QtCore.Qt.WindowFlags()
    ) -> None:
        super().__init__(parent, flags)

        self.picture = PictureWidget(self)
        self.statusbar_group = QtWidgets.QLabel(self)
        self.statusbar_priority = QtWidgets.QLabel(self)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.model: Optional[GroupedPictureModel] = None

        self.setCentralWidget(self.picture)
        self.statusbar.addWidget(self.statusbar_group)
        self.statusbar.addWidget(self.statusbar_priority)
        self.setStatusBar(self.statusbar)

        button_fit_to_window = QtWidgets.QAction("&Fit to window", self)
        button_fit_to_window.setCheckable(True)
        button_fit_to_window.setChecked(True)
        button_fit_to_window.setStatusTip("Resize picture to fit to window")
        button_fit_to_window.triggered[bool].connect(self.on_fit_to_window)

        button_stay_on_top = QtWidgets.QAction("&Stay on top", self)
        button_stay_on_top.setCheckable(True)
        button_stay_on_top.setChecked(False)
        button_stay_on_top.setStatusTip("Have the windows always stay on top")
        button_stay_on_top.triggered[bool].connect(self.on_stay_on_top)

        button_normalize_scale = QtWidgets.QAction("&Normalize scale", self)
        button_normalize_scale.setCheckable(True)
        button_normalize_scale.setChecked(False)
        button_normalize_scale.setStatusTip(
            "Resize the pictures to match the resolution of the largest picture in the group"
        )
        button_normalize_scale.triggered[bool].connect(self.on_normalize_scale)

        menu = self.menuBar()

        picture_menu = menu.addMenu("&Picture")
        picture_menu.addAction(button_fit_to_window)
        picture_menu.addAction(button_stay_on_top)
        picture_menu.addAction(button_normalize_scale)

        self.picture.load_file_failed.connect(self.on_load_file_failed)

    def on_load_file_failed(self, text: str):
        logging.warning(text)
        self.statusbar.showMessage(text)

    def on_fit_to_window(self, checked: bool) -> None:
        self.picture.fit_to_window = checked

    def on_stay_on_top(self, checked: bool) -> None:
        self.request_on_top.emit(checked)

    def on_normalize_scale(self, checked: bool) -> None:
        assert self.model
        assert self.picture.pic_group
        if checked:
            self.picture.fixed_size = self.model.get_group_max_size(self.picture.pic_group)
        else:
            self.picture.fixed_size = None

    # qt virtual

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.hide()
        event.accept()


class TableWidget(QtWidgets.QWidget):
    def __init__(self, picture_window: PictureWindow, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.assume_input_timezone = "local"

        self.picture_window = picture_window
        self.view = GroupedPictureView(parent=self)
        self.model = GroupedPictureModel()
        self.view.setModel(self.model)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.view)

        self.view.activated.connect(self.row_activated)
        self.picture_window.picture.picture_checked.connect(self.on_pic_checked)
        self.picture_window.picture.picture_reference.connect(self.on_pic_reference)
        self.model.row_checked.connect(self.on_row_checked)
        self.model.row_reference.connect(self.on_row_reference)

        self.picture_window.picture.load_next_in_group.connect(self.on_load_next_in_group)
        self.picture_window.picture.load_prev_in_group.connect(self.on_load_prev_in_group)
        self.picture_window.picture.load_next_group.connect(self.on_load_next_group)
        self.picture_window.picture.load_prev_group.connect(self.on_load_prev_group)

    def load_picture(self, group: int, path: str, checked: bool, priority: int) -> None:
        self.picture_window.picture.load_picture(group, path, checked, priority == 0)
        self.picture_window.setWindowTitle(to_dos_path(path))
        self.picture_window.statusbar_group.setText(f"Group: {group}")
        self.picture_window.statusbar_priority.setText(f"File: {priority}")

    # callbacks

    def on_pic_checked(self, group: int, path: str, checked: bool) -> None:
        self.model.set_checked_by_file(group, path, checked)
        self.view.viewport().repaint()

    def on_pic_reference(self, group: int, path: str) -> None:
        self.model.set_reference_by_file(group, path)
        self.view.viewport().repaint()

    def on_row_checked(self, group: int, path: str, checked: bool) -> None:
        self.picture_window.picture.check_picture(group, path, checked)

    def on_row_reference(self, group: int, path: str) -> None:
        self.picture_window.picture.reference_picture(group, path)

    def on_load_next_in_group(self, group: int, path: str) -> None:
        try:
            group, path, checked, priority = self.model.get_file(group, path, "next-in-group")
            self.load_picture(group, path, checked, priority)
        except IndexError:
            pass

    def on_load_prev_in_group(self, group: int, path: str) -> None:
        try:
            group, path, checked, priority = self.model.get_file(group, path, "prev-in-group")
            self.load_picture(group, path, checked, priority)
        except IndexError:
            pass

    def on_load_next_group(self, group: int, path: str) -> None:
        try:
            group, path, checked, priority = self.model.get_file(group, path, "next-group")
            self.load_picture(group, path, checked, priority)
        except IndexError:
            pass

    def on_load_prev_group(self, group: int, path: str) -> None:
        try:
            group, path, checked, priority = self.model.get_file(group, path, "prev-group")
            self.load_picture(group, path, checked, priority)
        except IndexError:
            pass

    def read_file(self, path: str) -> Tuple[int, int]:
        in_tz = {
            "local": None,
            "utc": timezone.utc,
        }[self.assume_input_timezone]

        if path.endswith(".csv"):
            df = pd.read_csv(
                path,
                header=0,
                keep_default_na=False,
                parse_dates=["mod_date"],
                dtype={
                    "filesize": "int64",
                    "width": "Int32",
                    "height": "Int32",
                    "priority": "int32",
                    "checked": "bool",
                },
            )
            df = df.convert_dtypes(infer_objects=False)
            date_cols = list({c for c in df.columns if "date" in c} - {"mod_date"})
            for col in date_cols:
                df[col] = to_datetime(df[col], in_tz)

        elif path.endswith(".parquet"):
            df = pd.read_parquet(path).reset_index()
        elif path.endswith(".json"):
            df = pd.read_json(path, orient="table")
        else:
            raise ValueError(f"Invalid file extension: {path}")

        return self.model.load_df(df)

    def to_file(self, path: str) -> None:
        if path.endswith(".csv"):
            self.model.df.to_csv(path)
        elif path.endswith(".parquet"):
            self.model.df.to_parquet(path)
        elif path.endswith(".json"):
            # fixme: to_json drops timezone info from datetimes
            # https://github.com/pandas-dev/pandas/issues/12997
            self.model.df.to_json(path, "table", force_ascii=False, indent=2)
        else:
            raise ValueError(f"Invalid file extension: {path}")

    def has_data(self) -> bool:
        return len(self.model.df) > 0

    def row_activated(self, index: QtCore.QModelIndex) -> None:
        row = self.model.get(index.row())

        group = row.name.item()
        path = row["path"]
        checked = row["checked"].item()
        priority = row["priority"].item()

        self.load_picture(group, path, checked, priority)
        self.picture_window.show()


class TableWindow(QtWidgets.QMainWindow):
    def __init__(
        self, parent: Optional[QtWidgets.QWidget] = None, flags: QtCore.Qt.WindowFlags = QtCore.Qt.WindowFlags()
    ):
        super().__init__(parent, flags)
        self.filename: Optional[str] = None

        self.picture_window = PictureWindow()
        self.picture_window.resize(800, 600)
        self.table = TableWidget(self.picture_window)
        self.setCentralWidget(self.table)
        self.picture_window.model = self.table.model

        button_open = QtWidgets.QAction("&Open", self)
        button_open.setStatusTip("Open list of image groups")
        button_open.triggered.connect(self.on_file_open)

        button_save = QtWidgets.QAction("&Save", self)
        button_save.setStatusTip("Save list of image groups with selection")
        button_save.triggered.connect(self.on_file_save)

        button_exit = QtWidgets.QAction("E&xit", self)
        button_exit.setStatusTip("Close the application")
        button_exit.triggered.connect(self.on_app_exit)

        self.button_prioritize = QtWidgets.QAction("&Prioritize", self)
        self.button_prioritize.setStatusTip("Sort files within groups")
        self.button_prioritize.triggered.connect(self.on_prioritize)
        self.button_prioritize.setEnabled(False)

        button_fullscreen = QtWidgets.QAction("&Fullscreen", self)
        button_fullscreen.setCheckable(True)
        button_fullscreen.setChecked(False)
        button_fullscreen.setStatusTip("Show window in fullscreen mode")
        button_fullscreen.triggered[bool].connect(self.on_fullscreen)

        self.statusbar_groups = QtWidgets.QLabel(self)
        self.statusbar_files = QtWidgets.QLabel(self)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.addWidget(self.statusbar_groups)
        self.statusbar.addWidget(self.statusbar_files)
        self.setStatusBar(self.statusbar)

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_open)
        file_menu.addAction(button_save)
        file_menu.addAction(button_exit)

        edit_menu = menu.addMenu("&Edit")
        edit_menu.addAction(self.button_prioritize)

        view_menu = menu.addMenu("&View")
        view_menu.addAction(button_fullscreen)

        self.picture_window.request_on_top.connect(self.on_set_top)

    def set_temporary_status(self, text: str) -> None:
        logging.debug(text)
        self.statusbar.showMessage(text)

    def warn_user(self, title: str, text: str) -> None:
        logging.error(text)
        QtWidgets.QMessageBox.warning(self, title, text)

    def read_file(self, path: str) -> None:
        try:
            with MeasureTime() as stopwatch:
                num_files, num_groups = self.table.read_file(path)
                time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
                self.statusbar_groups.setText(f"Groups: {num_groups}")
                self.statusbar_files.setText(f"Files: {num_files}")
                self.set_temporary_status(f"Loaded {num_files} files in {num_groups} groups in {time_delta}")
        except (ValueError, KeyError) as e:
            self.warn_user("Reading file failed", f"Reading {path} failed: {e}")
        else:
            self.button_prioritize.setEnabled(True)

    # qt callbacks

    def on_set_top(self, checked: bool) -> None:
        parent = self if checked else None
        pos = self.picture_window.pos()
        self.picture_window.setParent(parent)
        # `setParent()` with a parent resets the window flags and moves the window to (0, 0)...
        # get the `QtCore.Qt.Window` flag back and move position
        # Default window flags are `QtCore.Qt.WindowFlags(-2013204479)`.
        # >>> [f"0x{2**i:08x}" for i in np.nonzero(np.array([int(i) for i in np.binary_repr(-2013204479 + 2**32, 32)]) == 1)[0]]
        # ['0x00000001', '0x00000010', '0x00010000', '0x00020000', '0x00040000', '0x00080000', '0x80000000']
        # these look strange...
        self.picture_window.setWindowFlag(QtCore.Qt.Window)
        self.picture_window.move(pos)
        self.picture_window.show()

    def on_file_open(self, checked: bool) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        # dialog.setMimeTypeFilters(["text/csv"])
        dialog.setNameFilters(
            [
                "Data file (*.csv *.parquet *.json)",
            ]
        )
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)

        if dialog.exec_():
            path = dialog.selectedFiles()[0]
            self.read_file(path)

    def on_file_save(self, checked: bool) -> None:

        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        # dialog.setMimeTypeFilters(["text/csv"])
        dialog.setNameFilters(
            [
                "CSV file (*.csv)",
                "Parquet file (*.parquet)",
                "JSON file (*.json)",
            ]
        )
        dialog.setDefaultSuffix("csv")
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)
        if self.filename:
            dialog.selectFile(self.filename)

        if dialog.exec_():
            self.filename = dialog.selectedFiles()[0]
            assert self.filename  # for mypy
            try:
                self.table.to_file(self.filename)
            except ImportError as e:  # no pyarrow or fastparquet
                self.warn_user("Writing file failed", f"Missing dependencies for file export: {e}")

    def on_prioritize(self) -> None:
        prioritize_window = PrioritizeWindow(self.table.model.df, parent=self)
        prioritize_window.prioritize_widget.prioritize.connect(self.on_prioritize_done)
        prioritize_window.setWindowTitle("Prioritize")
        prioritize_window.show()

    def on_prioritize_done(self, df: pd.DataFrame) -> None:
        self.table.model.set_df(df)

    def on_app_exit(self, checked: bool) -> None:
        self.close()

    def on_fullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    # qt virtual

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.table.has_data():
            q = QtWidgets.QMessageBox()
            q.setWindowTitle(APP_NAME)
            q.setText("Are you sure?")
            q.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            q.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = q.exec()

            if ret == QtWidgets.QMessageBox.Ok:
                event.accept()
                self.picture_window.close()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":

    from argparse import ArgumentParser

    from genutility.args import is_file

    parser = ArgumentParser()
    parser.add_argument("--in-path", type=is_file, help="Allowed file types are csv, parquet and json")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])

    widget = TableWindow()
    widget.setWindowTitle(APP_NAME)
    widget.resize(800, 600)
    if args.in_path:
        widget.read_file(os.fspath(args.in_path))
    widget.show()

    sys.exit(app.exec_())
