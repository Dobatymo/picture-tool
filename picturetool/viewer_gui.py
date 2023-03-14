import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from fractions import Fraction
from functools import _CacheInfo, lru_cache, partial
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar

import humanize
import reverse_geocoder
from genutility.time import MeasureTime
from natsort import os_sorted
from PySide2 import QtCore, QtGui, QtWidgets

from .shared_gui import PixmapViewer, QImageWithBuffer, QSystemTrayIconWithMenu, read_qt_image

T = TypeVar("T")


def gps_dms_to_dd(dms: List[Fraction]) -> float:
    return float(dms[0] + dms[1] / 60 + dms[2] / 3600)


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

    def cache_info(self) -> _CacheInfo:
        return self.get.cache_info()

    def cache_clear(self) -> None:
        self.get.cache_clear()

    @QtCore.Slot(Path, Future)
    def on_finished(self, path: Path, future: Future) -> None:
        try:
            image = future.result()
            logging.debug("Cache request fullfilled %s", path)
            self.pic_loaded.emit(path, image)
        except Exception as e:
            logging.debug("Cache request error for <%s>. %s: %s", path, type(e).__name__, e)
            self.pic_load_failed.emit(path, e)


class WindowManager(Generic[T]):
    app_name: str

    def __init__(self, app_name: str, window_cls: Type[T]) -> None:
        self.app_name = app_name
        self.window_cls = window_cls

        self.windows: Set[T] = set()  # keep references to windows around
        self.tray: Optional[QSystemTrayIconWithMenu] = None
        self.create_kwargs: Dict[str, Any] = {}

    def set_create_args(self, **kwargs) -> None:
        self.create_kwargs = kwargs

    def make_tray(self, app: QtCore.QCoreApplication) -> None:
        assert self.tray is None
        app.setQuitOnLastWindowClosed(False)

        icon = QtWidgets.QFileIconProvider().icon(QtWidgets.QFileIconProvider.Computer)
        menu = QtWidgets.QMenu()

        action_open = QtWidgets.QAction("Open new window", menu)
        action_open.triggered.connect(self.create)

        action_quit = QtWidgets.QAction("Close app", menu)
        action_quit.setMenuRole(QtWidgets.QAction.QuitRole)
        action_quit.triggered.connect(app.quit)

        menu.addAction(action_open)
        menu.addAction(action_quit)

        self.tray = QSystemTrayIconWithMenu(icon, menu)
        self.tray.setToolTip(self.app_name)
        self.tray.doubleclicked.connect(self.create)

    @QtCore.Slot()
    def create(self) -> T:
        return self._create(**self.create_kwargs)

    @QtCore.Slot(dict)
    def create_from_args(self, args: dict) -> None:
        kwargs = self.create_kwargs.copy()
        kwargs.update(args)
        paths = kwargs.pop("paths", [])
        _ = kwargs.pop("verbose")  # ignore
        window = self._create(**kwargs)
        window.load_pictures(paths)

    def _create(self, *, maximized: bool, resolve_city_names: bool, mode: str) -> T:
        kwargs = locals()
        kwargs.pop("self")
        logging.debug("Created new window: %s", kwargs)
        window = self.window_cls(self, resolve_city_names)
        self.windows.add(window)
        window.set_fit_to_window(mode == "fit")
        if maximized:
            window.showMaximized()
        else:
            window.show()
        window.activateWindow()
        if self.tray is not None:
            self.tray.setVisible(False)
        return window

    def destroy(self, window: T) -> None:
        logging.debug("Destroying one window. Currently %d windows.", len(self.windows))
        self.windows.remove(window)
        if not self.windows and self.tray is not None:
            self.tray.setVisible(True)


class PictureWindow(QtWidgets.QMainWindow):
    last_action: str
    loaded: Optional[dict]
    path_idx: int

    extensions = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp"}
    cache_size = 10
    deleted_subdir = "deleted"
    delete_mode = "move-to-subdir"

    def __init__(
        self,
        wm: WindowManager,
        resolve_city_names: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
        flags: QtCore.Qt.WindowFlags = QtCore.Qt.WindowFlags(),
    ) -> None:
        super().__init__(parent, flags)

        self.last_action = "none"

        self.wm = wm
        self.resolve_city_names = resolve_city_names
        self.viewer = PixmapViewer(self)
        self.setCentralWidget(self.viewer)

        self.statusbar_number = QtWidgets.QLabel(self)
        self.statusbar_filename = QtWidgets.QLabel(self)
        self.statusbar_make = QtWidgets.QLabel(self)
        self.statusbar_model = QtWidgets.QLabel(self)
        self.statusbar_info = QtWidgets.QLabel(self)
        self.statusbar_scale = QtWidgets.QLabel(self)
        self.statusbar = QtWidgets.QStatusBar(self)

        self.statusbar.addWidget(self.statusbar_number)
        self.statusbar.addWidget(self.statusbar_filename)
        self.statusbar.addWidget(self.statusbar_make)
        self.statusbar.addWidget(self.statusbar_model)
        self.statusbar.addWidget(self.statusbar_info)
        self.statusbar.addWidget(self.statusbar_scale)
        self.setStatusBar(self.statusbar)

        button_file_open = QtWidgets.QAction("&Open", self)
        button_file_open.setStatusTip("Open file(s)")
        button_file_open.triggered.connect(self.on_file_open)

        button_file_close = QtWidgets.QAction("&Close window", self)
        button_file_close.setStatusTip("Close window")
        button_file_close.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_C))
        button_file_close.triggered.connect(self.close)

        button_file_quit = QtWidgets.QAction("&Close app", self)
        button_file_quit.setStatusTip("Close app")
        button_file_quit.setMenuRole(QtWidgets.QAction.QuitRole)
        button_file_quit.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_Q))
        button_file_quit.triggered.connect(QtCore.QCoreApplication.instance().quit)

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
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_file_open)
        file_menu.addAction(button_file_close)
        file_menu.addAction(button_file_quit)

        view_menu = menu.addMenu("&View")
        view_menu.addAction(self.button_fit_to_window)
        view_menu.addAction(button_rotate_cw)
        view_menu.addAction(button_rotate_ccw)

        self.cache = PictureCache(self.cache_size)
        self.cache.pic_loaded.connect(self.on_pic_loaded)
        self.cache.pic_load_failed.connect(self.on_pic_load_failed)

        self.viewer.scale_changed.connect(self.on_scale_changed)

        self.paths: List[Path] = []
        self._view_clear()

        if self.resolve_city_names:
            self.rg = reverse_geocoder

    def _view_clear(self) -> None:
        self.path_idx = -1
        self.loaded = None
        self.setWindowTitle(self.wm.app_name)
        self.statusbar_number.setText(None)
        self.statusbar_filename.setText(None)
        self.statusbar_make.setText(None)
        self.statusbar_model.setText(None)
        self.statusbar_info.setText(None)
        self.viewer.clear()

    def _get_pic_paths(self, path: Path) -> List[Path]:
        with MeasureTime() as stopwatch:
            paths = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in self.extensions]
            time_delta_1 = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
        with MeasureTime() as stopwatch:
            out = os_sorted(paths)
            time_delta_2 = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
        logging.debug(
            "Found %d pictures in <%s>. Reading took %s, sorting %s", len(paths), path, time_delta_1, time_delta_2
        )
        return out

    def try_preload(self, idx: int) -> None:
        try:
            self.cache.put(self.paths[idx])
        except IndexError:
            pass

    def load_pictures(self, paths: List[Path]) -> None:
        logging.debug("Loading pictures from: %s", ", ".join(f"<{os.fspath(p)}>" for p in paths))
        if len(paths) == 0:
            return
        elif len(paths) == 1:
            p = paths[0]
            if p.is_file():
                self.paths = self._get_pic_paths(p.parent)
                self.path_idx = self.paths.index(p)
            elif p.is_dir():
                self.paths = self._get_pic_paths(p)
                self.path_idx = 0
            else:
                assert False, f"{p} is not file or directory"
        else:
            self.paths = paths
            self.path_idx = 0

        if self.paths:
            path = self.paths[self.path_idx]
            self.cache.put(path)
            self.try_preload(self.path_idx + 1)
            self.try_preload(self.path_idx - 1)
            self.cache.load(path)
        else:
            self._view_clear()

    def load_next(self):
        if self.path_idx < len(self.paths) - 1:
            self.path_idx += 1
            self.try_preload(self.path_idx + 1)
            self.cache.load(self.paths[self.path_idx])
            self.last_action = "next"

    def load_prev(self):
        if self.path_idx > 0:
            self.path_idx -= 1
            self.try_preload(self.path_idx - 1)
            self.cache.load(self.paths[self.path_idx])
            self.last_action = "prev"

    def load_first(self):
        if self.path_idx != 0:
            self.path_idx = 0
            self.try_preload(self.path_idx + 1)
            self.cache.load(self.paths[self.path_idx])
            self.last_action = "first"

    def load_last(self):
        if self.path_idx != len(self.paths) - 1:
            self.path_idx = len(self.paths) - 1
            self.try_preload(self.path_idx - 1)
            self.cache.load(self.paths[self.path_idx])
            self.last_action = "last"

    def set_fit_to_window(self, checked: bool) -> None:
        self.button_fit_to_window.setChecked(checked)
        self.viewer.fit_to_window = checked

    def _load_after_delete(self, idx: int) -> None:
        first_idx = 0
        last_idx = len(self.paths) - 1

        if self.last_action in ("none", "next", "last"):
            self.path_idx = min(idx, last_idx)
        elif self.last_action in ("prev", "first"):
            self.path_idx = max(first_idx, idx - 1)
        else:
            assert False, f"Unhandled last action: {self.last_action}"

        if self.paths:
            self.cache.load(self.paths[self.path_idx])
        else:
            self._view_clear()

    def delete_current(self) -> None:
        assert self.loaded is not None
        path: Path = self.loaded["path"]
        idx: int = self.loaded["idx"]

        if self.delete_mode == "move-to-subdir":
            ret = QtWidgets.QMessageBox.question(self, "Delete file?", f"Do you want to delete <{path}>?")
            if ret == QtWidgets.QMessageBox.StandardButton.Yes:
                target_dir = path.parent / self.deleted_subdir
                target = target_dir / path.name
                if target.exists():
                    QtWidgets.QMessageBox.warning(
                        self, "Cannot delete file", f"<{path}> could not be deleted because <{target}> already exists."
                    )
                else:
                    target_dir.mkdir(parents=False, exist_ok=True)
                    try:
                        path.rename(target)
                    except FileNotFoundError:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Cannot delete file",
                            f"<{path}> could not be deleted because it only existed in cache.",
                        )

                    del_path = self.paths.pop(idx)
                    assert del_path == path
                    self._load_after_delete(idx)
            elif ret == QtWidgets.QMessageBox.StandardButton.No:
                pass
            else:
                assert False
        else:
            raise RuntimeError("Not implemented yet")

    # signal handlers

    @lru_cache(1000)
    def get_location(self, lat: List[Fraction], lon: List[Fraction]) -> Optional[str]:
        lat_lon = gps_dms_to_dd(lat), gps_dms_to_dd(lon)
        return self.rg.get(lat_lon)["name"]

    def make_cam_info_string(self, meta: dict) -> str:
        if self.resolve_city_names:
            try:
                with MeasureTime() as stopwatch:
                    meta["city"] = self.get_location(meta["gps-lat"], meta["gps-lon"])
                    time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
                logging.debug("Resolving GPS coordinates took %s", time_delta)
            except KeyError:
                pass

        vals = {
            "FL (mm)": "focal-length",
            "F": "f-number",
            "E": "exposure-time",
            "ISO": "iso",
            "A": "aperture-value",
            "City": "city",
        }

        return ", ".join(f"{k}: {meta[v]}" for k, v in vals.items() if meta.get(v))

    @QtCore.Slot(Path, QImageWithBuffer)
    def on_pic_loaded(self, path: Path, image: QImageWithBuffer):
        idx = self.paths.index(path)
        self.loaded = {"path": path, "idx": idx}
        self.setWindowTitle(f"{path.name} - {self.wm.app_name}")
        self.statusbar_number.setText(f"{idx + 1}/{len(self.paths)}")
        self.statusbar_filename.setText(path.name)
        self.statusbar_make.setText(image.meta.get("make"))
        self.statusbar_model.setText(image.meta.get("model"))
        self.statusbar_info.setText(self.make_cam_info_string(image.meta))

        self.viewer.setPixmap(image.get_pixmap())

    @QtCore.Slot(Path, Exception)
    def on_pic_load_failed(self, path: Path, e: Exception):
        idx = self.paths.index(path)
        if isinstance(e, FileNotFoundError):
            del_path = self.paths.pop(idx)
            assert del_path == path
            self._load_after_delete(idx)
        else:
            self.statusbar_number.setText(f"{idx + 1}/{len(self.paths)}")
            self.statusbar_filename.setText(None)
            self.statusbar_make.setText(None)
            self.statusbar_model.setText(None)
            self.statusbar_info.setText(None)
            self.viewer.clear()
            QtWidgets.QMessageBox.warning(
                self, "Loading picture failed", f"Loading <{path}> failed. {type(e).__name__}: {e}"
            )

    @QtCore.Slot()
    def on_file_open(self):
        name_filters = " ".join(f"*{ext}" for ext in self.extensions)
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dialog.setNameFilters([f"Image files ({name_filters})"])
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)

        if dialog.exec_():
            paths = list(map(Path, dialog.selectedFiles()))
            self.load_pictures(paths)

        self.activateWindow()

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
            event.accept()
            self.delete_current()
        elif event.key() == QtCore.Qt.Key_Left and event.modifiers() == QtCore.Qt.NoModifier:
            self.load_prev()
            event.accept()
        elif event.key() == QtCore.Qt.Key_Right and event.modifiers() == QtCore.Qt.NoModifier:
            self.load_next()
            event.accept()
        elif event.key() == QtCore.Qt.Key_Home and event.modifiers() == QtCore.Qt.NoModifier:
            self.load_first()
            event.accept()
        elif event.key() == QtCore.Qt.Key_End and event.modifiers() == QtCore.Qt.NoModifier:
            self.load_last()
            event.accept()
        else:
            event.ignore()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.wm.destroy(self)
        event.accept()


class PyServer(QtCore.QThread):
    name: str
    message_received = QtCore.Signal(dict)

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self) -> None:
        with Listener(self.name, "AF_PIPE") as listener:
            while True:
                with listener.accept() as conn:
                    try:
                        msg = conn.recv()
                        logging.debug("Received message: %s", msg)
                    except EOFError:
                        logging.error("Receiving message failed: EOFError")
                        continue
                    except Exception:
                        logging.exception("Receiving message failed")
                        continue

                    if msg is None:
                        break

                    self.message_received.emit(msg)

    def stop(self) -> None:
        with Client(self.name, "AF_PIPE") as conn:
            conn.send(None)