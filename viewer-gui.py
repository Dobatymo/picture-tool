import logging
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Optional

from natsort import os_sorted
from PySide2 import QtCore, QtGui, QtWidgets

from shared_gui import PixmapViewer, QImageWithBuffer, read_qt_image


class PictureCache(QtCore.QObject):

    pic_loaded = QtCore.Signal(Path, QImageWithBuffer)
    pic_load_failed = QtCore.Signal(Path, Exception)

    def __init__(self, size: Optional[int] = None):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.get = lru_cache(size)(self._call)

    def _call(self, path: Path) -> Future:
        return self.executor.submit(read_qt_image, os.fspath(path))

    def put(self, path: Path) -> None:
        logging.debug("Cache put %s", path)
        self.get(path)

    def load(self, path: Path) -> None:
        logging.debug("Cache request %s", path)
        self.get(path).add_done_callback(partial(self.on_finished, path))

    def cache_info(self):
        return self.get.cache_info()

    def cache_clear(self):
        return self.get.cache_clear()

    @QtCore.Slot(Path, Future)
    def on_finished(self, path: Path, future: Future) -> None:
        try:
            image = future.result()
            logging.debug("Cache request fullfilled %s", path)
            self.pic_loaded.emit(path, image)
        except Exception as e:
            logging.debug("Cache request error for <%s>. %s: %s", path, type(e).__name__, e)
            self.pic_load_failed.emit(path, e)


class PictureWindow(QtWidgets.QMainWindow):

    extensions = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp"}
    cache_size = 10

    def __init__(
        self, parent: Optional[QtWidgets.QWidget] = None, flags: QtCore.Qt.WindowFlags = QtCore.Qt.WindowFlags()
    ) -> None:
        super().__init__(parent, flags)

        self.viewer = PixmapViewer(self)
        self.setCentralWidget(self.viewer)

        self.statusbar_number = QtWidgets.QLabel(self)
        self.statusbar_filename = QtWidgets.QLabel(self)
        self.statusbar_scale = QtWidgets.QLabel(self)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.addWidget(self.statusbar_number)
        self.statusbar.addWidget(self.statusbar_filename)
        self.statusbar.addWidget(self.statusbar_scale)
        self.setStatusBar(self.statusbar)

        self.button_fit_to_window = QtWidgets.QAction("&Fit to window", self)
        self.button_fit_to_window.setCheckable(True)
        self.button_fit_to_window.setChecked(True)
        self.button_fit_to_window.setStatusTip("Resize picture to fit to window")
        self.button_fit_to_window.triggered[bool].connect(self.on_fit_to_window)

        button_rotate_cw = QtWidgets.QAction("&Rotate clockwise", self)
        button_rotate_cw.setStatusTip("Rotate picture clockwise (view only)")
        button_rotate_cw.triggered.connect(self.on_rotate_cw)

        button_rotate_ccw = QtWidgets.QAction("&Rotate counter-clockwise", self)
        button_rotate_ccw.setStatusTip("Rotate picture counter-clockwise (view only)")
        button_rotate_ccw.triggered.connect(self.on_rotate_ccw)

        menu = self.menuBar()
        picture_menu = menu.addMenu("&View")
        picture_menu.addAction(self.button_fit_to_window)
        picture_menu.addAction(button_rotate_cw)
        picture_menu.addAction(button_rotate_ccw)

        self.cache = PictureCache(self.cache_size)
        self.cache.pic_loaded.connect(self.on_pic_loaded)
        self.cache.pic_load_failed.connect(self.on_pic_load_failed)

        self.viewer.scale_changed.connect(self.on_scale_changed)

        self.paths: List[Path] = []
        self.idx: int = -1

    def on_pic_loaded(self, path: Path, image: QImageWithBuffer):
        idx = self.paths.index(path) + 1
        self.statusbar_number.setText(f"{idx}/{len(self.paths)}")
        self.statusbar_filename.setText(path.name)
        self.viewer.setPixmap(image.get_pixmap())

    def on_pic_load_failed(self, path: Path, image: QImageWithBuffer):
        idx = self.paths.index(path) + 1
        self.statusbar_number.setText(f"{idx}/{len(self.paths)}")
        self.statusbar_filename.setText(None)
        self.viewer.clear()

    def _get_pic_paths(self, path: Path) -> List[Path]:
        from datetime import timedelta

        import humanize
        from genutility.time import MeasureTime

        paths = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in self.extensions]
        logging.debug("Found %d files in <%s>", len(paths), path)
        with MeasureTime() as stopwatch:
            out = os_sorted(paths)
            time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
        logging.debug("Sorting %d picture paths took %s", len(paths), time_delta)
        return out

    def try_preload(self, idx: int) -> None:
        try:
            self.cache.put(self.paths[idx])
        except IndexError:
            pass

    def load_pictures(self, paths: List[Path]) -> None:
        logging.debug("Loading pictures: %s", ", ".join(map(os.fspath, paths)))
        if len(paths) == 0:
            return
        elif len(paths) == 1:
            self.paths = self._get_pic_paths(paths[0].parent)
            self.path_idx = self.paths.index(paths[0])
        else:
            self.paths = paths
            self.path_idx = 0

        path = self.paths[self.path_idx]
        self.cache.put(path)
        self.try_preload(self.path_idx + 1)
        self.try_preload(self.path_idx - 1)
        self.cache.load(path)

    def load_next(self):
        if self.path_idx < len(self.paths) - 1:
            self.path_idx += 1
            self.try_preload(self.path_idx + 1)
            self.cache.load(self.paths[self.path_idx])

    def load_prev(self):
        if self.path_idx > 0:
            self.path_idx -= 1
            self.try_preload(self.path_idx - 1)
            self.cache.load(self.paths[self.path_idx])

    def set_fit_to_window(self, checked: bool) -> None:
        self.button_fit_to_window.setChecked(checked)
        self.viewer.fit_to_window = checked

    # signal handlers

    @QtCore.Slot()
    def on_rotate_cw(self):
        tr = QtGui.QTransform()
        tr.rotate(90)
        self.viewer.label.transform(tr)

    @QtCore.Slot()
    def on_rotate_ccw(self):
        tr = QtGui.QTransform()
        tr.rotate(-90)
        self.viewer.label.transform(tr)

    @QtCore.Slot(bool)
    def on_fit_to_window(self, checked: bool) -> None:
        self.viewer.fit_to_window = checked

    @QtCore.Slot(float)
    def on_scale_changed(self, scale: float) -> None:
        self.statusbar_scale.setText(f"{scale:.03f}")

    # qt event handlers

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Delete and event.modifiers() == QtCore.Qt.NoModifier:
            print("Key_Delete")
            event.accept()
        elif event.key() == QtCore.Qt.Key_Left and event.modifiers() == QtCore.Qt.NoModifier:
            self.load_prev()
            event.accept()
        elif event.key() == QtCore.Qt.Key_Right and event.modifiers() == QtCore.Qt.NoModifier:
            self.load_next()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":

    from argparse import ArgumentParser

    from genutility.args import is_file

    parser = ArgumentParser()
    parser.add_argument("paths", metavar="PATH", type=is_file, nargs="+", help="Open image file")
    parser.add_argument("--mode", choices=("fit", "original"), default="fit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("PIL").setLevel(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    app = QtWidgets.QApplication([])

    window = PictureWindow()
    window.set_fit_to_window(args.mode == "fit")
    window.showMaximized()

    QtCore.QTimer.singleShot(0, lambda: window.load_pictures(args.paths))

    sys.exit(app.exec_())
