import logging
import os
from collections import Counter, deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from datetime import timedelta
from fractions import Fraction
from functools import _CacheInfo, lru_cache, partial
from multiprocessing.connection import Client, Listener
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from typing import Counter as CounterT
from typing import Deque, Dict, Generic, List, Optional, Sequence, Set, Tuple, Type, TypeVar

import humanize
from genutility.time import MeasureTime
from houtu import ReverseGeocode
from natsort import os_sorted
from PIL import ImageOps
from PySide2 import QtCore, QtGui, QtWidgets

from .shared_gui import (
    ImageTransformT,
    ImperfectTransform,
    PixmapViewer,
    QImageWithBuffer,
    QSystemTrayIconWithMenu,
    _equalize_hist_skimage,
    _grayscale,
    crop_half_save,
    read_qt_image,
    rotate_save,
)
from .utils import MyFraction, extensions_images

T = TypeVar("T")
IMAGE_LOADER_NUM_WORKERS = 2
IMAGE_CACHE_SIZE = 10
UNDO_LIST_SIZE = 10
QT_DEFAULT_WINDOW_FLAGS = QtCore.Qt.WindowFlags()
_grayscale.__name__ = "grayscale"


def gps_dms_to_dd(dms: Sequence[Fraction]) -> float:
    """Degrees Minutes Seconds to Decimal Degrees"""

    return float(dms[0] + dms[1] / 60 + dms[2] / 3600)


def nice_meta(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return MyFraction(obj).limit(4, 9)
    return obj


class PictureCache(QtCore.QObject):
    pic_loaded = QtCore.Signal(Path, QImageWithBuffer)
    pic_load_failed = QtCore.Signal(Path, Exception)
    pic_load_skipped = QtCore.Signal(Path)

    process: List[ImageTransformT]
    outstanding_loads: Deque[Tuple[Path, Tuple[ImageTransformT, ...]]]
    ignored_loads: CounterT[Tuple[Path, Tuple[ImageTransformT, ...]]]

    def __init__(self, size: Optional[int] = None) -> None:
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=IMAGE_LOADER_NUM_WORKERS)
        self.get = lru_cache(maxsize=size)(self._call)
        self.process = []
        self.outstanding_loads = deque()
        self.ignored_loads = Counter()
        self.futures: Set[Future] = set()  # set of non-cached futures
        self.futures_lock = Lock()

    def _call(self, path: Path, process: Tuple[ImageTransformT, ...]) -> Future:
        future = self.executor.submit(read_qt_image, os.fspath(path), process=process)
        with self.futures_lock:
            self.futures.add(future)
        return future

    def put(self, path: Path) -> None:
        """Put image in cache using threadpool.
        If the image is already in the cache, do nothing.

        If the `put` or `load` methods are called from the same thread,
        it's guaranteed that all calls after the first one load from cache
        (if it wasn't already evicted again).
        """

        process = tuple(self.process)
        logging.debug("Cache put <%s> %r", path, process)
        try:
            future = self.get(path, process)
        except RuntimeError as e:
            logging.info("Failed to put to cache: %s", e)
        else:
            future.add_done_callback(partial(self.on_put_done, path, process))

    def load(self, path: Path) -> None:
        """Load image using threadpool.
        If the image is not in the cache yet, put it there.

        If the `put` or `load` methods are called from the same thread,
        it's guaranteed that all calls after the first one load from cache
        (if it wasn't already evicted again).
        """

        process = tuple(self.process)
        logging.debug("Cache load <%s> %r", path, process)
        self.outstanding_loads.append((path, process))
        try:
            future = self.get(path, process)
        except RuntimeError as e:
            logging.info("Failed to load from cache: %s", e)
        else:
            future.add_done_callback(partial(self.on_load_done, path, process))

    def cache_info(self) -> _CacheInfo:
        return self.get.cache_info()

    def cache_clear(self) -> None:
        self.get.cache_clear()

    def on_put_done(self, path: Path, process: Tuple[ImageTransformT, ...], future: Future) -> None:
        try:
            with self.futures_lock:
                self.futures.remove(future)
            cached = False
        except KeyError:
            cached = True

        try:
            future.result()
            logging.debug("Cache put fulfilled (cached=%s) <%s> %r", cached, path, process)
        except CancelledError:
            logging.debug("Cache put cancelled (cached=%s) <%s> %r", cached, path, process)
        except Exception:
            logging.exception("Cache put error (cached=%s) <%s> %r", cached, path, process)

    def on_load_done(self, path: Path, process: Tuple[ImageTransformT, ...], future: Future) -> None:
        try:
            with self.futures_lock:
                self.futures.remove(future)
            cached = False
        except KeyError:
            cached = True

        key = (path, process)

        if self.ignored_loads[key] == 0:
            while True:
                key_ = self.outstanding_loads.popleft()
                if key == key_:
                    break
                else:
                    self.ignored_loads.update([key_])
        elif self.ignored_loads[key] == 1:
            del self.ignored_loads[key]
            self.pic_load_skipped.emit(path)
            return
        elif self.ignored_loads[key] > 1:
            self.ignored_loads.subtract([key])
            self.pic_load_skipped.emit(path)
            return
        else:
            assert False  # noqa: B011

        try:
            image = future.result()
            logging.debug("Cache load fulfilled (cached=%s) <%s> %r", cached, path, process)
            self.pic_loaded.emit(path, image)
        except CancelledError as e:
            logging.debug("Cache load cancelled (cached=%s) <%s> %r", cached, path, process)
            self.pic_load_failed.emit(path, e)
        except Exception as e:
            logging.exception("Cache load error (cached=%s) <%s> %r", cached, path, process)
            self.pic_load_failed.emit(path, e)

    def close(self) -> None:
        # make a thread-safe copy of the set, since `cancel` can modify the set during iteration.
        # don't call `cancel` under the lock,
        # since this might immediately call the callback method which will deadlock
        with self.futures_lock:
            futures = self.futures.copy()
        for future in futures:
            future.cancel()
        self.executor.shutdown(wait=True)  # python39: cancel_futures=True
        assert not self.futures
        logging.info("clearing cache")
        self.cache_clear()


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

        action_about = QtWidgets.QAction("About", menu)
        action_about.setMenuRole(QtWidgets.QAction.AboutRole)
        action_about.triggered.connect(self.about)

        action_quit = QtWidgets.QAction("Close app", menu)
        action_quit.setMenuRole(QtWidgets.QAction.QuitRole)
        action_quit.triggered.connect(app.quit)

        menu.addAction(action_open)
        menu.addAction(action_about)
        menu.addAction(action_quit)

        self.tray = QSystemTrayIconWithMenu(icon, menu)
        self.tray.setToolTip(self.app_name)
        self.tray.doubleclicked.connect(self.create)

    @QtCore.Slot()
    def create(self) -> T:
        return self._create(**self.create_kwargs)

    @QtCore.Slot()
    def about(self) -> None:
        pass

    @QtCore.Slot(dict)
    def create_from_args(self, args: dict) -> None:
        kwargs = self.create_kwargs.copy()
        kwargs.update(args)
        paths = kwargs.pop("paths", [])
        _ = kwargs.pop("verbose")  # ignore
        _ = kwargs.pop("debug")  # ignore
        window = self._create(**kwargs)
        window.load_pictures(paths)

    def _create(self, *, maximized: bool, resolve_city_names: bool, mode: str, config: dict) -> T:
        kwargs = locals()
        kwargs.pop("self")
        logging.debug("Created new window: %s", kwargs)
        window = self.window_cls(self, resolve_city_names, config)
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
        logging.debug("Destroying one window. Currently %d window(s).", len(self.windows))
        window.close()
        self.windows.remove(window)
        if not self.windows and self.tray is not None:
            self.tray.setVisible(True)


class PictureWindow(QtWidgets.QMainWindow):
    rg: ReverseGeocode = None

    last_action: str
    loaded: Optional[dict]  # information about currently loaded picture
    path_idx: int  # index of the picture which is supposed to be loaded

    extensions = extensions_images
    cache_size = IMAGE_CACHE_SIZE
    deleted_subdir = "deleted"
    delete_mode = "move-to-subdir"

    def __init__(
        self,
        wm: WindowManager,
        resolve_city_names: bool,
        config: dict,
        parent: Optional[QtWidgets.QWidget] = None,
        flags: QtCore.Qt.WindowFlags = QT_DEFAULT_WINDOW_FLAGS,
    ) -> None:
        super().__init__(parent, flags)

        self.last_action = "none"
        self.num_busy = 0

        self.wm = wm
        self.resolve_city_names = resolve_city_names
        self.viewer = PixmapViewer(self)
        self.setCentralWidget(self.viewer)

        self.statusbar_number = QtWidgets.QLabel(self)
        self.statusbar_filename = QtWidgets.QLabel(self)
        self.statusbar_filesize = QtWidgets.QLabel(self)
        self.statusbar_resolution = QtWidgets.QLabel(self)
        self.statusbar_make = QtWidgets.QLabel(self)
        self.statusbar_model = QtWidgets.QLabel(self)
        self.statusbar_info = QtWidgets.QLabel(self)
        self.statusbar_scale = QtWidgets.QLabel(self)
        self.statusbar_transforms = QtWidgets.QLabel(self)
        self.statusbar = QtWidgets.QStatusBar(self)

        self.statusbar.addWidget(self.statusbar_number)
        self.statusbar.addWidget(self.statusbar_filename)
        self.statusbar.addWidget(self.statusbar_filesize)
        self.statusbar.addWidget(self.statusbar_resolution)
        self.statusbar.addWidget(self.statusbar_make)
        self.statusbar.addWidget(self.statusbar_model)
        self.statusbar.addWidget(self.statusbar_info)
        self.statusbar.addWidget(self.statusbar_scale)
        self.statusbar.addWidget(self.statusbar_transforms)
        self.setStatusBar(self.statusbar)

        button_file_open = QtWidgets.QAction("&Open", self)
        button_file_open.setStatusTip("Open file(s)")
        button_file_open.triggered.connect(self.on_file_open)

        button_file_close = QtWidgets.QAction("&Close window", self)
        button_file_close.setStatusTip("Close window")
        button_file_close.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape))
        button_file_close.triggered.connect(self.close)

        button_file_quit = QtWidgets.QAction("Close &app", self)
        button_file_quit.setStatusTip("Close app")
        button_file_quit.setMenuRole(QtWidgets.QAction.QuitRole)
        button_file_quit.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_Q))
        button_file_quit.triggered.connect(QtCore.QCoreApplication.instance().quit)

        self.button_fit_to_window = QtWidgets.QAction("&Fit to window", self)
        self.button_fit_to_window.setCheckable(True)
        self.button_fit_to_window.setChecked(True)
        self.button_fit_to_window.setStatusTip("Resize picture to fit to window")
        self.button_fit_to_window.triggered[bool].connect(self.on_fit_to_window)

        button_view_rotate_cw = QtWidgets.QAction("&Rotate clockwise", self)
        button_view_rotate_cw.setStatusTip("Rotate picture clockwise (view only)")
        button_view_rotate_cw.triggered.connect(self.on_view_rotate_cw)

        button_view_rotate_ccw = QtWidgets.QAction("&Rotate counter-clockwise", self)
        button_view_rotate_ccw.setStatusTip("Rotate picture counter-clockwise (view only)")
        button_view_rotate_ccw.triggered.connect(self.on_view_rotate_ccw)

        button_view_fullscreen = QtWidgets.QAction("&Fullscreen", self)
        button_view_fullscreen.setCheckable(True)
        button_view_fullscreen.setChecked(False)
        button_view_fullscreen.setStatusTip("Show picture in fullscreen mode")
        button_view_fullscreen.triggered[bool].connect(self.on_view_fullscreen)
        button_view_fullscreen.setShortcut(QtGui.QKeySequence.FullScreen)

        button_grayscale = QtWidgets.QAction("&Grayscale", self)
        button_grayscale.setCheckable(True)
        button_grayscale.setStatusTip("Convert to grayscale")
        button_grayscale.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_G))
        button_grayscale.triggered[bool].connect(self.on_filter_grayscale)

        button_autocontrast = QtWidgets.QAction("&Maximize (normalize) image contrast", self)
        button_autocontrast.setCheckable(True)
        button_autocontrast.setStatusTip(
            "This function calculates a histogram of the input image (or mask region), removes cutoff percent of the lightest and darkest pixels from the histogram, and remaps the image so that the darkest pixel becomes black (0), and the lightest becomes white (255)."
        )
        button_autocontrast.triggered[bool].connect(self.on_filter_autocontrast)

        button_equalize = QtWidgets.QAction("&Histogram equalization", self)
        button_equalize.setCheckable(True)
        button_equalize.setStatusTip(
            "This function applies a non-linear mapping to the input image, in order to create a uniform distribution of grayscale values in the output image."
        )
        button_equalize.triggered[bool].connect(self.on_filter_equalize)

        button_crop_bottom = QtWidgets.QAction("&Crop bottom half", self)
        button_crop_bottom.setStatusTip("Losslessly crop away the bottom half of the image (create new file)")
        button_crop_bottom.triggered[bool].connect(self.on_crop_bottom)

        button_crop_top = QtWidgets.QAction("&Crop top half", self)
        button_crop_top.setStatusTip("Losslessly crop away the top half of the image (create new file)")
        button_crop_top.triggered[bool].connect(self.on_crop_top)

        button_edit_rotate_cw = QtWidgets.QAction("&Rotate clockwise", self)
        button_edit_rotate_cw.setStatusTip("Losslessly rotate picture clockwise (create new file)")
        button_edit_rotate_cw.triggered.connect(self.on_edit_rotate_cw)

        button_edit_rotate_180 = QtWidgets.QAction("&Rotate 180 degrees", self)
        button_edit_rotate_180.setStatusTip("Losslessly rotate picture 180 degrees (create new file)")
        button_edit_rotate_180.triggered.connect(self.on_edit_rotate_180)

        button_edit_rotate_ccw = QtWidgets.QAction("&Rotate counter-clockwise", self)
        button_edit_rotate_ccw.setStatusTip("Losslessly rotate picture counter-clockwise (create new file)")
        button_edit_rotate_ccw.triggered.connect(self.on_edit_rotate_ccw)

        self.menu = self.menuBar()
        file_menu = self.menu.addMenu("&File")
        file_menu.menuAction().setStatusTip("Basic app operations")
        file_menu.addAction(button_file_open)
        file_menu.addAction(button_file_close)
        file_menu.addAction(button_file_quit)

        view_menu = self.menu.addMenu("&View")
        view_menu.menuAction().setStatusTip("Actions here only affect the view, they don't modify the image file")
        view_menu.addAction(self.button_fit_to_window)
        view_menu.addAction(button_view_rotate_cw)
        view_menu.addAction(button_view_rotate_ccw)
        view_menu.addAction(button_view_fullscreen)

        filters_menu = self.menu.addMenu("&Filters")
        filters_menu.menuAction().setStatusTip("Actions here only affect the view, they don't modify the image file")
        filters_menu.addAction(button_grayscale)
        filters_menu.addAction(button_autocontrast)
        filters_menu.addAction(button_equalize)

        edit_menu = self.menu.addMenu("&Edit")
        edit_menu.menuAction().setStatusTip("Actions here save an edited version of the file to disk")
        edit_menu.addAction(button_crop_bottom)
        edit_menu.addAction(button_crop_top)
        edit_menu.addAction(button_edit_rotate_cw)
        edit_menu.addAction(button_edit_rotate_180)
        edit_menu.addAction(button_edit_rotate_ccw)

        action_undo = QtWidgets.QAction("&Undo", self)
        action_undo.setShortcut(QtGui.QKeySequence.Undo)
        action_undo.triggered[bool].connect(self.on_undo)

        actions_menu = self.menu.addMenu("&Actions")
        edit_menu.menuAction().setStatusTip("Global actions")
        actions_menu.addAction(action_undo)

        self.cache = PictureCache(self.cache_size)
        self.cache.pic_loaded.connect(self.on_pic_loaded)
        self.cache.pic_load_failed.connect(self.on_pic_load_failed)
        self.cache.pic_load_skipped.connect(self.on_pic_load_skipped)

        self.viewer.scale_changed.connect(self.on_scale_changed)

        self.paths: List[Path] = []
        self._view_clear()

        if self.resolve_city_names and PictureWindow.rg is None:
            PictureWindow.rg = ReverseGeocode()

        self.user_events: Dict[QtCore.Qt.Key, Tuple[Callable, tuple]] = {}
        user_funcs_map = {
            "move_to_subdir": self.move_to_subdir,
        }

        for k, (funcname, args) in config.get("user_events", {}).items():
            try:
                key = getattr(QtCore.Qt.Key, k)
            except AttributeError:
                logging.error("Invalid Qt.Key %s", k)
                continue

            try:
                func = user_funcs_map[funcname]
            except KeyError:
                logging.error("Invalid function name %s", funcname)
                continue

            self.user_events[key] = (func, args)

        self.actions: Deque = deque(maxlen=UNDO_LIST_SIZE)

    def busy_add(self):
        if self.num_busy == 0:
            self.setCursor(QtCore.Qt.BusyCursor)
        self.num_busy += 1

    def busy_sub(self):
        self.num_busy -= 1
        if self.num_busy == 0:
            self.unsetCursor()

    def _view_clear(self) -> None:
        self.path_idx = -1
        self.loaded = None
        self.setWindowTitle(self.wm.app_name)
        self.statusbar_number.setText(None)
        self.statusbar_filename.setText(None)
        self.statusbar_filesize.setText(None)
        self.statusbar_resolution.setText(None)
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
        if idx >= 0 and idx < len(self.paths):
            self.cache.put(self.paths[idx])

    def load_pictures(self, paths: Sequence[Path]) -> None:
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
                raise ValueError(f"Path <{p}> is neither file nor directory")
        else:
            self.path_idx = 0
            self.paths = []
            for p in paths:
                if p.is_file():
                    self.paths.append(p)
                elif p.is_dir():
                    self.paths.extend(self._get_pic_paths(p))
                else:
                    logging.warning("Path <%s> is neither file nor directory", p)

        if self.paths:
            path = self.paths[self.path_idx]
            self.busy_add()
            self.cache.load(path)
            self.try_preload(self.path_idx + 1)
            self.try_preload(self.path_idx - 1)
        else:
            self._view_clear()

    def reload(self) -> None:
        if self.path_idx >= 0:
            self.busy_add()
            self.cache.load(self.paths[self.path_idx])

    def load_next(self) -> None:
        if self.path_idx < len(self.paths) - 1:
            self.path_idx += 1
            self.busy_add()
            self.cache.load(self.paths[self.path_idx])
            self.try_preload(self.path_idx + 1)
            self.last_action = "next"

    def load_prev(self) -> None:
        if self.path_idx > 0:
            self.path_idx -= 1
            self.busy_add()
            self.cache.load(self.paths[self.path_idx])
            self.try_preload(self.path_idx - 1)
            self.last_action = "prev"

    def load_first(self) -> None:
        if self.path_idx != 0:
            self.path_idx = 0
            self.busy_add()
            self.cache.load(self.paths[self.path_idx])
            self.try_preload(self.path_idx + 1)
            self.last_action = "first"

    def load_last(self) -> None:
        if self.path_idx != len(self.paths) - 1:
            self.path_idx = len(self.paths) - 1
            self.busy_add()
            self.cache.load(self.paths[self.path_idx])
            self.try_preload(self.path_idx - 1)
            self.last_action = "last"

    def set_fit_to_window(self, checked: bool) -> None:
        self.button_fit_to_window.setChecked(checked)
        self.viewer.fit_to_window = checked

    def _load_after_image_gone(self, idx: int) -> None:
        first_idx = 0
        last_idx = len(self.paths) - 1

        if self.last_action in ("none", "next", "last"):
            self.path_idx = min(idx, last_idx)
        elif self.last_action in ("prev", "first"):
            self.path_idx = max(first_idx, idx - 1)
        else:
            assert False, f"Unhandled last action: {self.last_action}"

        if self.paths:
            self.busy_add()
            self.cache.load(self.paths[self.path_idx])
        else:
            self._view_clear()

    def move_to_subdir(self, path, idx, subdir, mode: str, ask: bool) -> None:
        verb = {
            "move": ("move", "moved", "Move"),
            "delete": ("delete", "deleted", "Delete"),
        }[mode]

        if ask:
            ret = QtWidgets.QMessageBox.question(self, f"{verb[2]} file?", f"Do you want to {verb[0]} <{path}>?")
        else:
            ret = QtWidgets.QMessageBox.StandardButton.Yes

        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            target_dir = path.parent / subdir
            target = target_dir / path.name
            if target.exists():
                QtWidgets.QMessageBox.warning(
                    self,
                    f"Cannot {verb[0]} file",
                    f"<{path}> could not be {verb[1]} because <{target}> already exists.",
                )
            else:
                target_dir.mkdir(parents=False, exist_ok=True)
                try:
                    path.rename(target)
                except FileNotFoundError:
                    QtWidgets.QMessageBox.warning(
                        self,
                        f"Cannot {verb[0]} file",
                        f"<{path}> could not be {verb[1]} because it only existed in cache.",
                    )
                except Exception as e:
                    logging.exception(f"Failed to {verb[0]} file <%s>", path)
                    QtWidgets.QMessageBox.warning(
                        self, f"Cannot {verb[0]} file", f"<{path}> could not be {verb[1]}: {e}"
                    )
                    return
                else:
                    self.actions.append(("move-to-subdir", path, target))  # doesn't undo `mkdir`

                del_path = self.paths.pop(idx)
                assert del_path == path
                self._load_after_image_gone(idx)
        elif ret == QtWidgets.QMessageBox.StandardButton.No:
            pass
        else:
            assert False

    def delete_current(self) -> None:
        assert self.loaded is not None
        path: Path = self.loaded["path"]
        idx: int = self.loaded["idx"]

        if self.delete_mode == "move-to-subdir":
            self.move_to_subdir(path, idx, self.deleted_subdir, mode="delete", ask=True)
        else:
            raise RuntimeError("Not implemented yet")

    def refresh_folder(self) -> None:
        if self.loaded is not None:
            path: Path = self.loaded["path"]
            self.load_pictures([path])

    @classmethod
    @lru_cache(1000)
    def get_location(cls, lat: Sequence[Fraction], lon: Sequence[Fraction]) -> Optional[str]:
        assert cls.rg is not None
        coords, distance, city = cls.rg.lat_lon(gps_dms_to_dd(lat), gps_dms_to_dd(lon), "degrees")
        return city.name

    def make_cam_info_string(self, meta: Dict[str, Any]) -> str:
        if self.resolve_city_names and meta.get("gps-lat") and meta.get("gps-lon"):
            with MeasureTime() as stopwatch:
                meta["city"] = self.get_location(meta["gps-lat"], meta["gps-lon"])
                time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
            logging.debug("Resolving GPS coordinates took %s", time_delta)

        vals = {
            "FL (mm)": "focal-length",
            "F": "f-number",
            "E (s)": "exposure-time",
            "ISO": "iso",
            "A": "aperture-value",
            "City": "city",
        }

        return ", ".join(f"{k}: {nice_meta(meta[v])}" for k, v in vals.items() if meta.get(v))

    def handle_user_event(self, key: QtCore.Qt.Key) -> None:
        func, args = self.user_events[key]

        assert self.loaded is not None
        path: Path = self.loaded["path"]
        idx: int = self.loaded["idx"]

        try:
            func(path, idx, *args)
        except Exception:
            logging.exception("Failed to run user event %s", func, args)

    # signal handlers

    @QtCore.Slot(Path, QImageWithBuffer)
    def on_pic_loaded(self, path: Path, image: QImageWithBuffer) -> None:
        idx = self.paths.index(path)
        self.loaded = {"path": path, "idx": idx}
        self.setWindowTitle(f"{path.name} - {self.wm.app_name}")
        self.statusbar_number.setText(f"{idx + 1}/{len(self.paths)}")
        self.statusbar_filename.setText(path.name)
        self.statusbar_filesize.setText(str(path.stat().st_size))
        self.statusbar_resolution.setText(f"{image.width}x{image.height}")
        self.statusbar_make.setText(image.meta.get("make"))
        self.statusbar_model.setText(image.meta.get("model"))
        self.statusbar_info.setText(self.make_cam_info_string(image.meta))

        self.viewer.setPixmap(image.get_pixmap())
        self.busy_sub()

    @QtCore.Slot(Path, Exception)
    def on_pic_load_failed(self, path: Path, e: Exception) -> None:
        idx = self.paths.index(path)
        if isinstance(e, FileNotFoundError):
            del_path = self.paths.pop(idx)
            assert del_path == path
            self._load_after_image_gone(idx)
        elif isinstance(e, CancelledError):
            pass
        else:
            self.statusbar_number.setText(f"{idx + 1}/{len(self.paths)}")
            self.statusbar_filename.setText(None)
            self.statusbar_filesize.setText(None)
            self.statusbar_resolution.setText(None)
            self.statusbar_make.setText(None)
            self.statusbar_model.setText(None)
            self.statusbar_info.setText(None)
            self.viewer.clear()
            QtWidgets.QMessageBox.warning(
                self, "Loading picture failed", f"Loading <{path}> failed. {type(e).__name__}: {e}"
            )
        self.busy_sub()

    @QtCore.Slot(Path)
    def on_pic_load_skipped(self, path: Path) -> None:
        self.busy_sub()

    @QtCore.Slot()
    def on_file_open(self) -> None:
        # fixme: how to open files and/or directories?
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
    def on_view_rotate_cw(self) -> None:
        tr = QtGui.QTransform()
        tr.rotate(90)
        self.viewer.label.transform(tr, "rotate")

    @QtCore.Slot()
    def on_view_rotate_ccw(self) -> None:
        tr = QtGui.QTransform()
        tr.rotate(-90)
        self.viewer.label.transform(tr, "rotate")

    @QtCore.Slot(bool)
    def on_view_fullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
            self.statusbar.hide()
            # self.menu.hide()  # hiding the menu also disables the shortcut...
        else:
            self.showNormal()
            self.activateWindow()
            self.showMaximized()

            self.statusbar.show()
            # self.menu.show()

    @QtCore.Slot(bool)
    def on_filter_grayscale(self, checked: bool) -> None:
        if checked:
            self.cache.process.append(_grayscale)
        else:
            self.cache.process.remove(_grayscale)
        self.reload()

    @QtCore.Slot(bool)
    def on_filter_autocontrast(self, checked: bool) -> None:
        if checked:
            self.cache.process.append(ImageOps.autocontrast)
        else:
            self.cache.process.remove(ImageOps.autocontrast)
        self.reload()

    @QtCore.Slot(bool)
    def on_filter_equalize(self, checked: bool) -> None:
        if checked:
            self.cache.process.append(_equalize_hist_skimage)
        else:
            self.cache.process.remove(_equalize_hist_skimage)
        self.reload()

    @QtCore.Slot()
    def on_crop_bottom(self) -> None:
        assert self.loaded is not None
        pm = self.viewer.label.pm
        path = self.loaded["path"]
        if pm is not None:
            format = pm.meta["format"]
            assert format == "JPEG", format
            try:
                crop_half_save(path, "bottom")
            except (ImperfectTransform, FileExistsError) as e:
                QtWidgets.QMessageBox.warning(
                    self, "Cropping picture failed", f"Cropping <{path}> failed. {type(e).__name__}: {e}"
                )

    @QtCore.Slot()
    def on_undo(self) -> None:
        if not self.actions:
            QtWidgets.QMessageBox.information(self, "Undo", "Nothing to undo!")
            return

        actionname, *args = self.actions[-1]
        ret = QtWidgets.QMessageBox.question(self, f"Undo {actionname}?", f"Do you want to undo {actionname} {args}?")

        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            if actionname == "move-to-subdir":
                path, target = args
                target.rename(path)
                self.load_pictures([path])
            else:
                assert False

            self.actions.pop()
        elif ret == QtWidgets.QMessageBox.StandardButton.No:
            pass
        else:
            assert False

    @QtCore.Slot()
    def on_crop_top(self) -> None:
        assert self.loaded is not None
        pm = self.viewer.label.pm
        path = self.loaded["path"]
        if pm is not None:
            format = pm.meta["format"]
            assert format == "JPEG", format
            try:
                crop_half_save(path, "top")
            except (ImperfectTransform, FileExistsError) as e:
                QtWidgets.QMessageBox.warning(
                    self, "Cropping picture failed", f"Cropping <{path}> failed. {type(e).__name__}: {e}"
                )

    def _on_edit_rotate(self, target: str) -> None:
        assert self.loaded is not None
        pm = self.viewer.label.pm
        path = self.loaded["path"]
        if pm is not None:
            format = pm.meta["format"]
            assert format == "JPEG", format
            try:
                rotate_save(path, target)
            except (ImperfectTransform, FileExistsError) as e:
                QtWidgets.QMessageBox.warning(
                    self, "Rotating picture failed", f"Rotating <{path}> failed. {type(e).__name__}: {e}"
                )

    @QtCore.Slot()
    def on_edit_rotate_cw(self) -> None:
        self._on_edit_rotate("cw")

    @QtCore.Slot()
    def on_edit_rotate_180(self) -> None:
        self._on_edit_rotate("180")

    @QtCore.Slot()
    def on_edit_rotate_ccw(self) -> None:
        self._on_edit_rotate("ccw")

    @QtCore.Slot(bool)
    def on_fit_to_window(self, checked: bool) -> None:
        self.viewer.fit_to_window = checked

    @QtCore.Slot(float)
    def on_scale_changed(self, scale: float) -> None:
        self.statusbar_scale.setText(f"{scale:.03f}")
        self.statusbar_transforms.setText("->".join(self.viewer.label.transforms))

    # qt event handlers

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Delete and event.modifiers() == QtCore.Qt.NoModifier:
            event.accept()
            self.delete_current()
        elif event.matches(QtGui.QKeySequence.Refresh):
            event.accept()
            self.refresh_folder()
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
        elif event.key() in self.user_events and event.modifiers() == QtCore.Qt.NoModifier:
            self.handle_user_event(event.key())
            event.accept()
        else:
            event.ignore()

    def close(self) -> None:
        logging.debug("Closing window and emptying cache")
        self.cache.close()
        super().close()

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
            logging.info("Pipe server `%s` started", self.name)
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
