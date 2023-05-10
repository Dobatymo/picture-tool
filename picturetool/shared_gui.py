import logging
from copy import deepcopy
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import piexif
from genutility.pillow import NoActionNeeded, fix_orientation
from PIL import Image
from pillow_heif import register_heif_opener
from PySide2 import QtCore, QtGui, QtWidgets

register_heif_opener()

logger = logging.getLogger(__name__)


class QPixmapWithMeta:
    pixmap: QtGui.QPixmap
    meta: Dict[str, Any]

    def __init__(self, pixmap: QtGui.QPixmap, meta: Dict[str, Any]) -> None:
        self.pixmap = pixmap
        self.meta = meta

    def transformed(self, tr: QtGui.QTransform, name: str):
        pixmap = self.pixmap.transformed(tr)
        meta = deepcopy(self.meta)
        meta["transforms"].append(name)
        return QPixmapWithMeta(pixmap, meta)

    def size(self) -> QtCore.QSize:
        return self.pixmap.size()


class QImageWithBuffer:
    def __init__(self, image: QtGui.QImage, buffer: QtCore.QByteArray, meta: Dict[str, Any]) -> None:
        self.image = image
        self.buffer = buffer
        self.meta = meta

    def get_pixmap(self) -> QPixmapWithMeta:
        pixmap = QtGui.QPixmap.fromImage(self.image)
        return QPixmapWithMeta(pixmap, self.meta)


def piexif_get(d: Dict[str, Dict[int, Any]], idx1: str, idx2: int, dtype: str) -> Any:
    try:
        val = d[idx1][idx2]
    except KeyError:
        return None

    try:
        if dtype == "ascii":
            return val.decode("ascii")
        elif dtype == "utf-8":
            return val.decode("utf-8")
        elif dtype == "int":
            return int(val)
        elif dtype == "float":
            return float(val)
        elif dtype == "rational":
            return Fraction(*val)
        elif dtype == "tuple-of-rational":
            return tuple(Fraction(*v) for v in val)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
    except (ValueError, TypeError) as e:
        logger.warning("Invalid exif value. %s: %s", type(e).__name__, e)
        return None
    except ZeroDivisionError:
        if dtype in ("rational", "tuple-of-rational"):
            return None
        raise


ImageTransformT = Callable[[Image.Image], Image.Image]


def read_qt_image(
    path: str, rotate: bool = True, process: Optional[Iterable[ImageTransformT]] = None
) -> QImageWithBuffer:
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
        "I;16": QtGui.QImage.Format_Grayscale16,
        "RGB": QtGui.QImage.Format_RGB888,
        "BGR;24": QtGui.QImage.Format_BGR888,
        "RGBA": QtGui.QImage.Format_RGBA8888,
        "RGBX": QtGui.QImage.Format_RGBX8888,
        "RGBa": QtGui.QImage.Format_RGBA8888_Premultiplied,
    }

    channelmap = {
        "L": 1,
        "I;16": 2,
        "RGB": 3,
        "BGR;24": 3,
        "RGBA": 4,
        "RGBX": 4,
    }

    with Image.open(path) as img:
        meta: Dict[str, Any] = {"transforms": []}

        if "exif" in img.info:
            exif = piexif.load(img.info["exif"])

            try:
                img = fix_orientation(img, exif)
                meta["transforms"].append("rotate")
            except (NoActionNeeded, KeyError):
                pass

            meta.update(
                {
                    "make": piexif_get(exif, "0th", piexif.ImageIFD.Make, "ascii"),
                    "model": piexif_get(exif, "0th", piexif.ImageIFD.Model, "ascii"),
                    "exposure-time": piexif_get(exif, "Exif", piexif.ExifIFD.ExposureTime, "rational"),
                    "f-number": piexif_get(exif, "Exif", piexif.ExifIFD.FNumber, "rational"),
                    "iso-speed": piexif_get(exif, "Exif", piexif.ExifIFD.ISOSpeed, "int"),
                    "aperture-value": piexif_get(exif, "Exif", piexif.ExifIFD.ApertureValue, "rational"),
                    "focal-length": piexif_get(exif, "Exif", piexif.ExifIFD.FocalLength, "rational"),
                    "iso": piexif_get(exif, "Exif", piexif.ExifIFD.ISOSpeedRatings, "int"),
                    "gps-lat": piexif_get(exif, "GPS", piexif.GPSIFD.GPSLatitude, "tuple-of-rational"),
                    "gps-lon": piexif_get(exif, "GPS", piexif.GPSIFD.GPSLongitude, "tuple-of-rational"),
                }
            )

        if process:
            for func in process:
                try:
                    img = func(img)
                    meta["transforms"].append(func.__name__)
                except OSError:
                    logging.debug("Applying %s to <%s> [mode=%s] failed", func.__name__, path, img.mode)
                    raise

        if img.mode not in modemap or (img.width * channelmap[img.mode]) % 4 != 0:
            # Unsupported image mode or image scanlines not 32-bit aligned
            img = img.convert("RGBA")
            meta["transforms"].append("convert-color-to-rgba")

        # QImage simply references the QByteArray. So you need to keep it around.

        qimg_format = modemap[img.mode]
        b = QtCore.QByteArray(img.tobytes())

    qimg = QtGui.QImage(b, img.size[0], img.size[1], qimg_format)  # , img.close
    return QImageWithBuffer(qimg, b, meta)


def read_qt_pixmap(path: str) -> QtGui.QPixmap:
    return read_qt_image(path).get_pixmap()


class AspectRatioPixmapLabel(QtWidgets.QLabel):
    pm: Optional[QPixmapWithMeta]

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
        self.pm = None

        # set properties
        self.fit_to_widget = fit_to_widget
        self.fixed_size = fixed_size
        self._scale_factor = 1.0
        self._transforms: List[str] = []

    @property
    def transforms(self) -> List[str]:
        if self.pm is None:
            return self._transforms
        else:
            return self.pm.meta.get("transforms", []) + self._transforms

    def _scaled_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap:
        assert self.pm is not None

        new_size = size * self.scale_factor
        print(size, new_size, self.pm.size())
        if self.pm.size() == new_size:
            self._transforms = []
            return self.pm.pixmap
        else:
            self._transforms = ["scale"]
            return self.pm.pixmap.scaled(
                size * self.scale_factor, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )

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

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float) -> None:
        self._scale_factor = value
        if self.pm is not None:
            self.scale()

    def transform(self, tr: QtGui.QTransform, name: str) -> None:
        assert self.pm is not None
        self.setPixmap(self.pm.transformed(tr, name))

    # qt funcs

    def clear(self) -> None:
        super().clear()
        self.pm = None

    def setPixmap(self, pm: QPixmapWithMeta) -> None:
        self.pm = pm
        self.scale()

    # qt event handlers

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        # The label doesn't get `resizeEvent`s when `QScrollArea.widgetResizable()` is False.
        # `resizeEvent`s are however triggered by the labels `self.adjustSize()`,
        # so when setting a new pixmap, a resize event could still be triggered
        # even if `self.fit_to_widget` is False.

        # print("resizeEvent", self.fit_to_widget, self.fixed_size)

        if self.pm is not None and self.fit_to_widget and self.fixed_size is None:
            self.scale()
        super().resizeEvent(event)


class PixmapViewer(QtWidgets.QScrollArea):
    scale_changed = QtCore.Signal(float)

    arrow_keys = [QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]

    label: AspectRatioPixmapLabel
    fit_to_window: bool

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.label = AspectRatioPixmapLabel(parent=parent)
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setWidget(self.label)

        # set properties
        self._fixed_size: Optional[Tuple[int, int]] = None  # set first
        self.fit_to_window = True

    @property
    def fixed_size(self) -> Optional[Tuple[int, int]]:
        return self._fixed_size

    @fixed_size.setter
    def fixed_size(self, value: Optional[Tuple[int, int]]) -> None:
        self._fixed_size = value
        if value is None:
            self.setWidgetResizable(self.label.fit_to_widget)
            self.label.fixed_size = None
        else:
            self.setWidgetResizable(False)
            self.label.fixed_size = QtCore.QSize(*value)

    @property  # type: ignore[no-redef]
    def fit_to_window(self) -> bool:
        return self._fit_to_window

    @fit_to_window.setter
    def fit_to_window(self, value: bool) -> None:
        self._fit_to_window = value
        if self.fixed_size is None:
            self.setWidgetResizable(value)
            self.label.fit_to_widget = value
        else:
            self.label._fit_to_widget = value

    # pass-through

    def setPixmap(self, pm: QPixmapWithMeta) -> None:
        self.label.setPixmap(pm)
        self.scale_changed.emit(self.label.scale_factor)

    def clear(self) -> None:
        self.label.clear()

    # qt event handlers

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in self.arrow_keys and event.modifiers() == QtCore.Qt.NoModifier:
            event.ignore()
        elif event.key() == QtCore.Qt.Key_0 and event.modifiers() == QtCore.Qt.ControlModifier:
            new_scale = 1.0
            self.label.scale_factor = new_scale
            self.scale_changed.emit(new_scale)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        units = event.angleDelta().y()
        if event.modifiers() == QtCore.Qt.ControlModifier and units != 0:
            inc = 1 + (abs(units) / 120) * 0.2

            if units > 0:
                new_scale = self.label.scale_factor * inc
                if new_scale <= 8:
                    self.label.scale_factor = new_scale
                    self.scale_changed.emit(new_scale)
            else:
                new_scale = self.label.scale_factor / inc
                if new_scale >= 0.01:
                    self.label.scale_factor = new_scale
                    self.scale_changed.emit(new_scale)

            event.accept()
        else:
            event.ignore()


class QSystemTrayIconWithMenu(QtWidgets.QSystemTrayIcon):
    doubleclicked = QtCore.Signal()

    def __init__(self, icon: QtGui.QIcon, menu: QtWidgets.QMenu, parent: Optional[QtWidgets.QWidget] = None) -> None:
        assert icon
        super().__init__(icon, parent)
        assert menu

        self.menu = menu
        self.setContextMenu(menu)

        # self.menu.aboutToHide.connect(self.on_aboutToHide)
        # self.menu.aboutToShow.connect(self.on_aboutToShow)
        # self.menu.hovered.connect(self.on_hovered)
        self.menu.triggered.connect(self.on_triggered)

        self.activated.connect(self.on_activated)

    @QtCore.Slot(QtWidgets.QSystemTrayIcon.ActivationReason)
    def on_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        logging.debug("%s", self.menu)
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.Context:
            self.menu.show()
        elif reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self.doubleclicked.emit()

    """
    @QtCore.Slot()
    def on_aboutToHide(self):
        logging.debug("called")

    @QtCore.Slot()
    def on_aboutToShow(self):
        logging.debug("called")
    """

    """
    @QtCore.Slot(QtWidgets.QAction)
    def on_hovered(self, action):
        logging.debug("%s", action)
    """

    @QtCore.Slot(QtWidgets.QAction)
    def on_triggered(self, action):
        logging.debug("%s", action)


def _equalize_hist_cv2(img: Image.Image) -> Image.Image:
    """Histogram equalization using opencv.
    Only supports 8-bit grayscale images.
    Other grayscale images are converted to 8-bit first.
    fast: 0.05s
    """

    import cv2
    import numpy as np

    arr = np.array(img)
    if img.mode == "L":
        pass
    elif img.mode == "I;16":
        arr = (arr / 256).astype(np.uint8)
    else:
        raise ValueError("Only grayscale images can be equalized")
    arr = cv2.equalizeHist(arr)
    return Image.fromarray(arr, "L")


def _equalize_hist_skimage(img: Image.Image) -> Image.Image:
    """Histogram equalization using scikit-image.
    Supports all grayscale images.
    slow 8-bit: 0.36s, 16-bit: 0.67s
    """

    import numpy as np
    from skimage.exposure import equalize_hist

    arr = np.array(img)
    if len(arr.shape) != 2:
        raise ValueError("Only grayscale images can be equalized")
    arr = equalize_hist(arr, nbins=256) * 256
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr, "L")


def _grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")
