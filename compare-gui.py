# pip install pandas pillow pillow-heif PySide2

import logging
import os
import platform
import subprocess
import sys
from itertools import chain
from typing import Any, Optional

import numpy as np
import pandas as pd
from genutility._files import to_dos_path
from PIL import Image
from pillow_heif import register_heif_opener
from PySide2 import QtCore, QtGui, QtWidgets

register_heif_opener()

APP_NAME = "compare-gui"


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
    def __init__(self, fit_to_widget: bool = True, parent: Optional[QtWidgets.QWidget] = None) -> None:

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

    def clear(self) -> None:
        super().clear()
        self.pm = None

    def scaledPixmap(self, size: QtCore.QSize) -> QtGui.QPixmap:
        assert self.pm is not None
        return self.pm.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def scale(self):
        assert self.pm is not None
        if self.fit_to_widget:
            super().setPixmap(self.scaledPixmap(self.size()))
        else:
            super().setPixmap(self.scaledPixmap(self.pm.size()))
            self.adjustSize()

    def setPixmap(self, pm: QtGui.QPixmap) -> None:
        self.pm = pm
        self.scale()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:

        # The label doesn't get `resizeEvent`s when `QScrollArea.widgetResizable()` is False.
        # `resizeEvent`s are however triggered by the labels `self.adjustSize()`,
        # so when setting a new pixmap, a resize event could still be triggered
        # even if `self.fit_to_widget` is False.

        if self.pm is not None and self.fit_to_widget:
            self.scale()
        super().resizeEvent(event)

    @property
    def fit_to_widget(self) -> bool:
        return self._fit_to_widget

    @fit_to_widget.setter
    def fit_to_widget(self, value: bool) -> None:
        self._fit_to_widget = value
        if self.pm is not None:
            self.scale()


def slice_to_list(value: slice):
    return list(range(value.stop)[value])


def iloc_by_index_and_bool(df, index, bool_idx) -> int:
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

    def load_df(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self.df = df.set_index("group")

        if "priority" not in df:
            priority = pd.Series(
                list(chain.from_iterable(range(i) for i in self.df.groupby("group").count()["path"])),
                dtype="int32",
            ).values
            assert not isinstance(priority, pd.Series), "Cannot assign series because of wrong index"
            self.df["priority"] = priority

        if "checked" not in df:
            self.df["checked"] = False

        self.cols = {name: self.df.columns.get_loc(name) for name in ("priority", "checked")}
        self.endResetModel()

    def get(self, row_idx: int) -> pd.Series:
        return self.df.iloc[row_idx]

    def set_checked(self, row_idx: int, checked: bool) -> None:
        col_idx = self.cols["checked"]
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

    def set_checked_by_file(self, group: int, path: str, checked: bool) -> None:
        idx = self.df.loc[group]["path"] == path
        row = iloc_by_index_and_bool(self.df, group, idx)
        col = self.cols["checked"]
        self.df.iat[row, col] = checked

        index = self.createIndex(row, col)
        self.dataChanged.emit(index, index)

    def set_reference_by_file(self, group: int, path: str) -> None:
        idx = self.df.loc[group]["path"] == path
        new_row = iloc_by_index_and_bool(self.df, group, idx)
        col_prio = self.cols["priority"]
        col_check = self.cols["checked"]
        priority = self.df.iat[new_row, col_prio]
        assert priority != 0, "selected row is already the reference"
        idx = self.df.loc[group]["priority"] == 0
        old_row = iloc_by_index_and_bool(self.df, group, idx)

        self.df.iat[new_row, col_prio] = 0
        self.df.iat[old_row, col_prio] = priority

        assert not self.df.iat[new_row, col_check]
        assert not self.df.iat[old_row, col_check]

        self.df.iloc[new_row], self.df.iloc[old_row] = self.df.iloc[old_row], self.df.iloc[new_row]

        index = self.createIndex(new_row, col_prio)
        self.dataChanged.emit(index, index)
        index = self.createIndex(old_row, col_prio)
        self.dataChanged.emit(index, index)

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

        if col == 0 and self.df.iat[row, self.cols["priority"]] > 0:
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
                    return str(self.df.iat[row, col - 1])
            except KeyError:
                logging.warning("Invalid table access at %d, %d", row, col)

        elif role == QtCore.Qt.CheckStateRole:
            if col == 0 and self.df.iat[row, self.cols["priority"]] > 0:
                if self.df.iat[row, self.cols["checked"]]:
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
            column = self.df.columns[column - 1]
            idx = self.df.groupby("group").nth(0).sort_values(column, ascending=ascending, kind="stable").index
            self.beginResetModel()
            self.df = self.df.loc[idx]
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

    def open_file(self, index: QtCore.QModelIndex) -> None:
        path = self.get_row_by_index(index)["path"]
        open_using_default_app(path)

    def make_reference(self, index: QtCore.QModelIndex) -> None:
        self.model().set_checked(index.row(), False)
        self.model().set_reference(index.row())

    def open_directory(self, index: QtCore.QModelIndex) -> None:
        path = self.get_row_by_index(index)["path"]
        show_in_file_manager(path)


class PictureWidget(QtWidgets.QWidget):

    picture_checked = QtCore.Signal(int, str, bool)
    picture_reference = QtCore.Signal(int, str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.button1 = QtWidgets.QPushButton("&Check", self)
        self.button1.setCheckable(True)
        self.button1.clicked[bool].connect(self.onCheck)
        self.button2 = QtWidgets.QPushButton("Make &reference", self)
        self.button2.setCheckable(True)
        self.button2.clicked[bool].connect(self.onMakeReference)
        self.label = AspectRatioPixmapLabel()

        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.scroll.setWidget(self.label)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.scroll)

        self.buttons = QtWidgets.QHBoxLayout()
        self.buttons.addWidget(self.button1)
        self.buttons.addWidget(self.button2)
        self.layout.addLayout(self.buttons)

        self.pic_path = None
        self.pic_group = None

        # set properties
        self.fit_to_window = True

    def onCheck(self, checked: bool) -> None:
        if self.pic_path is not None:
            self.picture_checked.emit(self.pic_group, self.pic_path, checked)

    def onMakeReference(self, checked: bool) -> None:
        assert checked
        self.button1.setChecked(False)
        self.button1.setEnabled(False)
        self.button2.setEnabled(False)
        if self.pic_path is not None:
            self.picture_checked.emit(self.pic_group, self.pic_path, False)
            self.picture_reference.emit(self.pic_group, self.pic_path)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        event.ignore()

    @property
    def fit_to_window(self) -> bool:
        return self._fit_to_window

    @fit_to_window.setter
    def fit_to_window(self, value: bool) -> None:
        self._fit_to_window = value
        self.scroll.setWidgetResizable(value)
        self.label.fit_to_widget = value

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
        try:
            self.pixmap = read_qt_pixmap(path)
            self.label.setPixmap(self.pixmap)
            self.button1.setChecked(checked)
            self.button1.setEnabled(not reference)
            self.button2.setChecked(reference)
            self.button2.setEnabled(not reference)
            self.pic_group = group
            self.pic_path = path
            return True
        except (FileNotFoundError, ValueError) as e:
            logging.warning("%s: %s", type(e).__name__, e)
            self.label.clear()
            self.button1.setChecked(checked)
            self.button1.setEnabled(False)
            self.button2.setChecked(reference)
            self.button2.setEnabled(False)
            self.pic_group = None
            self.pic_path = None
            return False


class PictureWindow(QtWidgets.QMainWindow):

    request_on_top = QtCore.Signal(bool)

    def __init__(
        self, parent: Optional[QtWidgets.QWidget] = None, flags: QtCore.Qt.WindowFlags = QtCore.Qt.WindowFlags()
    ) -> None:
        super().__init__(parent, flags)

        self.picture = PictureWidget()
        self.setCentralWidget(self.picture)
        self.setStatusBar(QtWidgets.QStatusBar(self))

        button_fit_to_window = QtWidgets.QAction("&Fit to window", self)
        button_fit_to_window.setCheckable(True)
        button_fit_to_window.setChecked(True)
        button_fit_to_window.setStatusTip("Resize picture to fit to window")
        button_fit_to_window.triggered[bool].connect(self.onFitToWindow)

        button_stay_on_top = QtWidgets.QAction("&Stay on top", self)
        button_stay_on_top.setCheckable(True)
        button_stay_on_top.setChecked(False)
        button_stay_on_top.setStatusTip("Have the windows always stay on top")
        button_stay_on_top.triggered[bool].connect(self.onStayOnTop)

        button_normalize_scale = QtWidgets.QAction("&Normalize scale", self)
        button_normalize_scale.setCheckable(True)
        button_normalize_scale.setChecked(False)
        button_normalize_scale.setStatusTip(
            "Resize the pictures to match the resolution of the largest picture in the group"
        )
        button_normalize_scale.triggered[bool].connect(self.onNormalizeScale)

        menu = self.menuBar()

        picture_menu = menu.addMenu("&Picture")
        picture_menu.addAction(button_fit_to_window)
        picture_menu.addAction(button_stay_on_top)
        picture_menu.addAction(button_normalize_scale)

    def onFitToWindow(self, checked: bool) -> None:
        self.picture.fit_to_window = checked

    def onStayOnTop(self, checked: bool) -> None:
        self.request_on_top.emit(checked)

    def onNormalizeScale(self, checked: bool) -> None:
        print("not implemented yet", checked)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.hide()
        event.accept()


class TableWidget(QtWidgets.QWidget):
    def __init__(self, picture_window: PictureWindow, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.picture_window = picture_window
        self.view = GroupedPictureView(parent=self)
        self.model = GroupedPictureModel()
        self.view.setModel(self.model)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.view)

        self.view.activated.connect(self.row_activated)
        self.picture_window.picture.picture_checked.connect(self.onPicChecked)
        self.picture_window.picture.picture_reference.connect(self.onPicReference)
        self.model.row_checked.connect(self.onRowChecked)
        self.model.row_reference.connect(self.onRowReference)

    def onPicChecked(self, group: int, path: str, checked: bool) -> None:
        self.model.set_checked_by_file(group, path, checked)
        self.view.viewport().repaint()

    def onPicReference(self, group: int, path: str) -> None:
        self.model.set_reference_by_file(group, path)
        self.view.viewport().repaint()

    def onRowChecked(self, group: int, path: str, checked: bool) -> None:
        self.picture_window.picture.check_picture(group, path, checked)

    def onRowReference(self, group: int, path: str) -> None:
        self.picture_window.picture.reference_picture(group, path)

    def load_csv(self, path: str) -> None:
        df = pd.read_csv(
            path,
            header=0,
            keep_default_na=False,
            parse_dates=["mod_date"],
            dtype={"priority": "int32", "checked": "bool"},
        )
        self.model.load_df(df)

    def to_csv(self, path: str) -> None:
        self.model.df.to_csv(path)

    def has_data(self) -> bool:
        return len(self.model.df) > 0

    def row_activated(self, index: QtCore.QModelIndex) -> None:
        row = self.model.get(index.row())

        group = row.name.item()
        path = row["path"]
        checked = row["checked"].item()
        priority = row["priority"].item() == 0

        self.picture_window.picture.load_picture(group, path, checked, priority)
        self.picture_window.setWindowTitle(to_dos_path(path))
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

        button_open = QtWidgets.QAction("&Open", self)
        button_open.setStatusTip("Open list of image groups")
        button_open.triggered.connect(self.onOpen)

        button_save = QtWidgets.QAction("&Save", self)
        button_save.setStatusTip("Save list of image groups with selection")
        button_save.triggered.connect(self.onSave)

        button_exit = QtWidgets.QAction("E&xit", self)
        button_exit.setStatusTip("Close the application")
        button_exit.triggered.connect(self.onExit)

        button_fullscreen = QtWidgets.QAction("&Fullscreen", self)
        button_fullscreen.setCheckable(True)
        button_fullscreen.setChecked(False)
        button_fullscreen.setStatusTip("Show window in fullscreen mode")
        button_fullscreen.triggered[bool].connect(self.onFullscreen)

        self.setStatusBar(QtWidgets.QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_open)
        file_menu.addAction(button_save)
        file_menu.addAction(button_exit)

        view_menu = menu.addMenu("&View")
        view_menu.addAction(button_fullscreen)

        self.picture_window.request_on_top.connect(self.onTop)

    def onTop(self, checked: bool) -> None:
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

    def onOpen(self, checked: bool) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dialog.setMimeTypeFilters(["text/csv"])
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)

        if dialog.exec_():
            path = dialog.selectedFiles()[0]
            self.load_csv(path)

    def onSave(self, checked: bool) -> None:

        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setMimeTypeFilters(["text/csv"])
        dialog.setDefaultSuffix("csv")
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)
        if self.filename:
            dialog.selectFile(self.filename)

        if dialog.exec_():
            self.filename = dialog.selectedFiles()[0]
            assert self.filename  # for mypy
            self.table.to_csv(self.filename)

    def load_csv(self, path: str) -> None:
        self.table.load_csv(path)

    def onExit(self, checked: bool) -> None:
        self.close()

    def onFullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def closeEvent(self, event) -> None:
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
    parser.add_argument("--csv-path", type=is_file)
    args = parser.parse_args()

    app = QtWidgets.QApplication([])

    widget = TableWindow()
    widget.setWindowTitle(APP_NAME)
    widget.resize(800, 600)
    if args.csv_path:
        widget.load_csv(args.csv_path)
    widget.show()

    sys.exit(app.exec_())
