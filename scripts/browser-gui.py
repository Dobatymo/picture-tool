import logging
import os
import shutil
from io import BytesIO
from pathlib import Path
from queue import Queue
from typing import Any, Iterator, Tuple

import piexif
from filemeta.exif import exif_table
from genutility.cache import cache
from genutility.exceptions import NoResult
from genutility.filesdb import FileDbSimple
from genutility.filesystem import scandir_ext
from genutility.pillow import exifinfo
from geopy.distance import distance as geodistance
from geopy.geocoders import Nominatim
from PIL import Image
from PySide2.QtCore import QDir, QPoint, QSize, Qt, QThread, Signal, Slot
from PySide2.QtGui import QIcon, QKeyEvent, QPixmap, QWheelEvent
from PySide2.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from picturetool.utils import extensions, parse_gpsinfo

APP_NAME = "Photo manager"
APP_ID = "photo-manager"
PLACEHOLDER_PATH = "data/placeholder.png"

ThumbsQueueT = "Queue[Union[str, Tuple[QListWidgetItem, Path]]]"


def icon_from_data(data: bytes) -> QIcon:
    icon = QIcon()
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    icon.addPixmap(pixmap)
    return icon


class ThumbnailDB(FileDbSimple):
    @classmethod
    def derived(cls):
        return [
            ("thumbnail", "BLOB", "?"),
        ]

    def __init__(self, path: str):
        FileDbSimple.__init__(self, path, "thumbnails")


class ThumbnailWorker(QThread):
    signal = Signal(QListWidgetItem, QIcon)

    def __init__(self, db: ThumbnailDB, q: ThumbsQueueT, size: Tuple[int, int]) -> None:
        QThread.__init__(self)
        self.db = db
        self.q = q
        self.size = size

    def create_thumbnail(self, path: Path) -> bytes:
        img = Image.open(path, mode="r")
        stream = BytesIO()

        img.thumbnail(self.size)
        img.save(stream, format="jpeg")

        return stream.getvalue()

    def run(self) -> None:
        while True:
            elm = self.q.get()
            if elm is None:
                self.db.close()
                break
            if elm == "commit":
                self.db.commit()
                continue

            item, path = elm

            data = self.create_thumbnail(path)
            self.db.add(path, derived={"thumbnail": data}, commit=False)
            icon = icon_from_data(data)

            self.signal.emit(item, icon)

    def stop(self) -> None:
        self.q.put(None)


class ExifDialog(QDialog):
    def __init__(self, path, parent, **kwargs) -> None:
        QDialog.__init__(self, parent, **kwargs)

        self.setWindowTitle("Image information")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["IFD", "Name", "Value"])

        for ifd, key, key_label, value, value_label in exif_table(os.fspath(path)):
            self.add_row(ifd, key_label, value_label)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)

    def add_row(self, ifd: str, key: str, value: str) -> None:
        count = self.table.rowCount()
        self.table.insertRow(count)
        self.table.setItem(count, 0, QTableWidgetItem(ifd))
        self.table.setItem(count, 1, QTableWidgetItem(key))
        self.table.setItem(count, 2, QTableWidgetItem(value))


class PhotoListWidget(QListWidget):
    def __init__(self, dbpath: str = "thumbs.sqlite", iconsize: Tuple[int, int] = (200, 200), **kwargs) -> None:
        QListWidget.__init__(self, **kwargs)

        self.db = ThumbnailDB(dbpath)

        self.icon_x, self.icon_y = iconsize
        self.m_factor = 1.5
        self.reset_icon_size()
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

        self.itemDoubleClicked.connect(self.show_item)
        self.customContextMenuRequested.connect(self.show_context)

        with open(PLACEHOLDER_PATH, "rb") as fr:
            self.placeholder_icon = icon_from_data(fr.read())

        self.thumbsqueue: ThumbsQueueT = Queue()
        self.thumbsthread = ThumbnailWorker(self.db, self.thumbsqueue, iconsize)
        self.thumbsthread.signal.connect(self.thumbs_signal_handler)
        self.thumbsthread.start()

    def __del__(self):
        self.db.close()
        self.thumbsthread.stop()
        self.thumbsthread.wait()

    def get_thumb_from_db(self, path: Path) -> bytes:
        (data,) = self.db.get(path, only={"thumbnail"})
        return data

    def show_settings(self) -> None:
        dialog = QDialog()
        dialog.open()

    def reset_icon_size(self) -> None:
        self.multiplier = 1.0
        self.setIconSize(QSize(self.icon_x, self.icon_y))

    @Slot(QListWidgetItem)
    def show_item(self, item: QListWidgetItem) -> None:
        self.show_settings()
        print(item)

    def show_selected_items(self) -> None:
        for item in self.selectedItems():
            print(item)

    def autorotate_selected_items(self) -> None:
        for item in self.selectedItems():
            print(item)

    def show_exif_selected(self):
        for item in self.selectedItems():
            src = item.data(Qt.UserRole)
            dlg = ExifDialog(src, self)
            dlg.exec_()

    @Slot(QPoint)
    def show_context(self, pos: QPoint) -> None:
        pos = self.mapToGlobal(pos)

        menu = QMenu()
        menu.addAction("Show", self.show_selected_items)
        menu.addAction("Show exif", self.show_exif_selected)
        menu.addAction("Delete", self.delete_selected_items)
        menu.addAction("Auto-rotate", self.autorotate_selected_items)
        menu.exec_(pos)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            scrolllength = event.angleDelta().y()
            if scrolllength > 0:
                self.multiplier *= self.m_factor * abs(scrolllength) / 120
            else:
                self.multiplier /= self.m_factor * abs(scrolllength) / 120

            x, y = int(self.icon_x * self.multiplier), int(self.icon_y * self.multiplier)
            self.setIconSize(QSize(x, y))
        else:
            QListWidget.wheelEvent(self, event)

    def delete_selected_items(self, backup: bool = True) -> None:
        for item in reversed(self.selectedItems()):
            src = item.data(Qt.UserRole)
            destdir = src.parent / "deleted"
            destdir.mkdir(parents=True, exist_ok=True)
            destfile = destdir / src.name
            if destfile.exists():
                raise RuntimeError(f"Cannot delete {src}. Destination already exists.")
            else:
                logging.info("Deleting %s to %s", src, destfile)
                shutil.move(src, destfile)
                self.takeItem(self.row(item))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_0:
            self.reset_icon_size()
        elif event.key() == Qt.Key_Delete:
            self.delete_selected_items(backup=True)
        else:
            QListWidget.keyPressEvent(self, event)

    def addItemFromPath(self, path: Path) -> QListWidgetItem:
        icon = QIcon(os.fspath(path))
        item = QListWidgetItem(icon, path.name)
        item.setData(Qt.UserRole, path)
        self.addItem(item)
        return item

    def addItemFromPlaceholder(self, path: Path) -> QListWidgetItem:
        item = QListWidgetItem(self.placeholder_icon, path.name)
        item.setData(Qt.UserRole, path)
        self.addItem(item)
        return item

    def addItemFromData(self, path: Path, data: Any) -> QListWidgetItem:
        icon = icon_from_data(data)
        item = QListWidgetItem(icon, path.name)
        item.setData(Qt.UserRole, path)
        self.addItem(item)
        return item

    def queue_thumbnail(self, item: QListWidgetItem, path: Path) -> None:
        self.thumbsqueue.put((item, path))

    def queue_commit(self):
        self.thumbsqueue.put("commit")

    @Slot(QListWidgetItem, QIcon)
    def thumbs_signal_handler(self, item: QListWidgetItem, icon: QIcon):
        item.setIcon(icon)

    def load_from_path(self, basepath: Path) -> None:
        if not basepath.exists():
            msg = QMessageBox()
            msg.setWindowTitle(APP_NAME)
            msg.setText("Invalid directory.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
            return

        self.clear()
        for _path in scandir_ext(basepath, extensions):
            path = Path(_path)
            try:
                data = self.get_thumb_from_db(path)
                item = self.addItemFromData(path, data)
            except NoResult:
                exif_dict = piexif.load(os.fspath(path))
                data = exif_dict["thumbnail"]
                if data:
                    item = self.addItemFromData(path, data)
                else:
                    item = self.addItemFromPlaceholder(path)
                self.queue_thumbnail(item, path)

        self.queue_commit()


class MyWidget(QWidget):
    def __init__(self, basepath: Path) -> None:
        QWidget.__init__(self)

        self.basepath = basepath
        self.geolocator = Nominatim(user_agent=APP_ID)

        self.pathfield = QComboBox()
        self.pathfield.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.pathfield.setEditable(True)
        self.button_browse = QPushButton("&Browse")

        self.searchfield = QComboBox()
        self.searchfield.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.searchfield.setEditable(True)
        self.button_search = QPushButton("&Search")

        self.viewer = PhotoListWidget()
        self.viewer.load_from_path(self.basepath)

        self.button_close = QPushButton("Close")

        self.layout = QGridLayout(self)
        self.layout.addWidget(self.pathfield, 0, 0)
        self.layout.addWidget(self.button_browse, 0, 1)
        self.layout.addWidget(self.searchfield, 1, 0)
        self.layout.addWidget(self.button_search, 1, 1)
        self.layout.addWidget(self.viewer, 2, 0, 1, 2)
        self.layout.addWidget(self.button_close, 3, 0, 1, 2)
        self.setLayout(self.layout)

        self.pathfield.lineEdit().returnPressed.connect(self.slot_browse)
        self.button_browse.clicked.connect(self.slot_browse_dialog)
        self.searchfield.lineEdit().returnPressed.connect(self.slot_search)
        self.button_search.clicked.connect(self.slot_search)
        self.button_close.clicked.connect(self.slot_close)

        menubar = QMenuBar()
        self.layout.setMenuBar(menubar)

        menu_file = menubar.addMenu("File")
        menu_file.addAction("Browse")
        menu_file.addAction("Open viewer")
        action_quit = menu_file.addAction("Quit")
        action_quit.triggered.connect(self.slot_close)

        menubar.addMenu("Settings")
        menu_help = menubar.addMenu("Help")
        menu_help.addAction("About")

    @Slot()
    def slot_browse(self) -> None:
        directory = QDir.toNativeSeparators(self.pathfield.currentText())

        if directory:
            self.viewer.load_from_path(Path(directory))

    @Slot()
    def slot_browse_dialog(self) -> None:
        directory = QDir.toNativeSeparators(QFileDialog.getExistingDirectory(self, "Find Files", QDir.currentPath()))

        if directory:
            if self.pathfield.findText(directory) == -1:
                self.pathfield.addItem(directory)
            self.pathfield.setCurrentIndex(self.pathfield.findText(directory))

            self.viewer.load_from_path(Path(directory))

    def load_path_location(self, lat, lon) -> None:
        def asd() -> Iterator[Tuple[float, Path]]:
            for entry in scandir_ext(self.basepath, extensions):
                path = Path(entry)
                with Image.open(os.fspath(path)) as img:
                    info = exifinfo(img)
                    try:
                        gps = info["GPSInfo"]
                    except KeyError:
                        continue
                img_lat, img_lon = parse_gpsinfo(gps)
                distance = geodistance((lat, lon), (img_lat, img_lon)).km
                logging.debug("Target: %s, %s Image: %s, %s Distance: %s", lat, lon, img_lat, img_lon, distance)
                yield distance, path

        sortedbydistance = sorted(asd())

        self.viewer.clear()
        for distance, path in sortedbydistance:
            self.viewer.addItemFromPath(path)

    def name2longlat(self, name: str) -> None:
        return self.geolocator.geocode(name).raw

    @Slot()
    def slot_search(self) -> None:
        addr = self.searchfield.currentText().lower()
        if addr:
            location = cache(Path("geoloc.json"), serializer="json")(self.name2longlat)(addr)
            self.searchfield.setCurrentText(location["display_name"])
            self.load_path_location(location["lat"], location["lon"])

    @Slot()
    def slot_close(self) -> None:
        self.close()


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("path", default=Path.cwd(), nargs="?")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    widget = MyWidget(args.path)
    widget.setWindowTitle(APP_NAME)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
