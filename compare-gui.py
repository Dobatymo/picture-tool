# pip install pandas pillow pillow-heif PySide2

import logging
import platform
import subprocess
import sys
from typing import Optional

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
        "RGB": QtGui.QImage.Format_RGB888,
        "RGBA": QtGui.QImage.Format_RGBA8888,
        "RGBX": QtGui.QImage.Format_RGBX8888,
    }

    channelmap = {
        "RGB": 3,
        "RGBA": 4,
        "RGBX": 4,
    }

    with Image.open(path) as img:
        try:
            format = modemap[img.mode]
        except KeyError:
            raise ValueError(f"Unsupported image mode: {img.mode}")

        if (img.width * channelmap[img.mode]) % 4 != 0:
            # Image scanlines not 32-bit aligned
            img = img.convert("RGBA")
            format = modemap[img.mode]

        b = QtCore.QByteArray(img.tobytes())
        qimg = QtGui.QImage(b, img.size[0], img.size[1], format)
        return QtGui.QPixmap.fromImage(qimg)


class AspectRatioPixmapLabel(QtWidgets.QLabel):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(1, 1)
        self.setScaledContents(False)
        self.pm: Optional[QtGui.QPixmap] = None

    def setPixmap(self, pm: QtGui.QPixmap) -> None:
        self.pm = pm
        super().setPixmap(self.scaledPixmap())

    def scaledPixmap(self) -> QtGui.QPixmap:
        assert self.pm is not None
        return self.pm.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def resizeEvent(self, event) -> None:
        if self.pm is not None:
            super().setPixmap(self.scaledPixmap())
        super().resizeEvent(event)


class GroupedPictureModel(QtCore.QAbstractTableModel):

    columns = ["group", "path", "filesize"]

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.df = pd.DataFrame({}, columns=self.columns).set_index("group")

    def load_df(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self.df = df
        self.endResetModel()

    # required by Qt

    def rowCount(self, index: QtCore.QModelIndex) -> int:
        return self.df.shape[0]

    def columnCount(self, index: QtCore.QModelIndex) -> int:
        return self.df.shape[1] + 1

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
                    return f"Sort by {self.df.columns[section-1]}"
                elif role == QtCore.Qt.StatusTipRole:
                    return f"Sort by {self.df.columns[section-1]}"

        return None

    def get(self, row: int) -> pd.Series:
        return self.df.iloc[row]

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> Optional[str]:
        if not index.isValid():
            return None

        if role == QtCore.Qt.DisplayRole:
            row, col = index.row(), index.column()

            try:
                if col == 0:
                    return str(self.df.index[row])
                elif col == 1:
                    return to_dos_path(self.df.iloc[row, 0])
                else:
                    return str(self.df.iloc[row, col - 1])
            except KeyError:
                logging.warning("Invalid table access at %d, %d", row, col)

        return None


def show_in_file_manager(path: str) -> None:

    if platform.system() == "Windows":
        if path.startswith("\\\\?\\"):
            path = path[4:]
        args = f'explorer /select,"{path}"'
        print(args)
        subprocess.run(args)
    else:
        raise RuntimeError("Önly windows implemented")


class GroupedPictureView(QtWidgets.QTableView):
    def __init__(self, wordwrap: bool = False, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSortingEnabled(True)

    def setWordWrap(self, on: bool) -> None:
        self.setWordWrap(on)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        index = self.indexAt(event.pos())
        if not index.isValid():
            return

        open_directory_action = QtWidgets.QAction("Open file", self)
        open_directory_action.triggered.connect(lambda event: self.open_file(event, index))

        open_directory_action = QtWidgets.QAction("Open directory", self)
        open_directory_action.triggered.connect(lambda event: self.open_directory(event, index))

        contextMenu = QtWidgets.QMenu(self)
        contextMenu.addAction(open_directory_action)
        contextMenu.exec_(event.globalPos())

    def open_directory(self, event, index):
        path = self.model().get(index.row())["path"]
        show_in_file_manager(path)


class PictureWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.button = QtWidgets.QPushButton("Click me!")
        self.label = AspectRatioPixmapLabel()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)

    def load_picture(self, path: str) -> None:
        try:
            self.pixmap = read_qt_pixmap(path)
            self.label.setPixmap(self.pixmap)
        except (FileNotFoundError, ValueError) as e:
            logging.warning("%s: %s", type(e).__name__, e)


class PictureWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.picture = PictureWidget()
        self.setCentralWidget(self.picture)

    def closeEvent(self, event) -> None:
        self.hide()
        event.accept()


class TableWidget(QtWidgets.QWidget):
    def __init__(self, picture_window: PictureWindow):
        super().__init__()

        self.picture_window = picture_window
        self.view = GroupedPictureView(self)
        self.model = GroupedPictureModel()
        self.view.setModel(self.model)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.view)

        self.view.selectionModel().selectionChanged.connect(self.selected_row)

    def load_csv(self, path: str):
        df = pd.read_csv(
            path,
            names=["group", "path", "filesize", "mod_date", "date_taken", "maker", "model"],
            index_col="group",
            keep_default_na=False,
        )

        self.model.load_df(df)

    def has_data(self) -> bool:
        return len(self.model.df) > 0

    def selected_row(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection):
        try:
            path = self.model.data(selected.value(0).indexes()[1])
            assert isinstance(path, str)
            self.picture_window.picture.load_picture(path)
            self.picture_window.setWindowTitle(to_dos_path(path))
            self.picture_window.show()
        except IndexError:
            logging.warning("IndexError: %s", selected.indexes())


class TableWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.picture_window = PictureWindow()
        self.picture_window.resize(800, 600)
        self.table = TableWidget(self.picture_window)
        self.setCentralWidget(self.table)

        button_open = QtWidgets.QAction("&Open", self)
        button_open.setStatusTip("Open list of image groups")
        button_open.triggered.connect(self.onOpen)

        button_exit = QtWidgets.QAction("E&xit", self)
        button_exit.setStatusTip("Close the application")
        button_exit.triggered.connect(self.onExit)

        self.setStatusBar(QtWidgets.QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_open)
        file_menu.addAction(button_exit)

    def onOpen(self, event):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dialog.setNameFilter("CSV files (*.csv)")
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)

        if dialog.exec_():
            path = dialog.selectedFiles()[0]
            self.table.load_csv(path)
            self.picture_window.show()

    def onExit(self, event):
        self.close()

    def closeEvent(self, event) -> None:
        if self.table.has_data():
            q = QtWidgets.QMessageBox()
            q.setText(APP_NAME)
            q.setInformativeText("Are you sure?")
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
    app = QtWidgets.QApplication([])

    widget = TableWindow()
    widget.setWindowTitle(APP_NAME)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
