import logging
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import piexif
from genutility.pillow import NoActionNeeded, fix_orientation
from PIL import Image
from pillow_heif import register_heif_opener
from PySide2 import QtCore, QtGui, QtWidgets

register_heif_opener()

logger = logging.getLogger(__name__)


class QImageWithBuffer:
    def __init__(self, image: QtGui.QImage, buffer: QtCore.QByteArray, meta: dict) -> None:
        self.image = image
        self.buffer = buffer
        self.meta = meta

    def get_pixmap(self) -> QtGui.QPixmap:
        return QtGui.QPixmap.fromImage(self.image)


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
        print(type(e).__name__, e)
        return None


def read_qt_image(path: str, rotate: bool = True) -> QImageWithBuffer:
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
        if "exif" in img.info:
            exif = piexif.load(img.info["exif"])

            try:
                img = fix_orientation(img, exif)
            except (NoActionNeeded, KeyError):
                pass

            meta = {
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
        else:
            meta = {}

        if img.mode not in modemap or (img.width * channelmap[img.mode]) % 4 != 0:
            # Unsupported image mode or image scanlines not 32-bit aligned
            img = img.convert("RGBA")

        # QImage simply references the QByteArray. So you need to keep it around.

        qimg_format = modemap[img.mode]
        b = QtCore.QByteArray(img.tobytes())

    qimg = QtGui.QImage(b, img.size[0], img.size[1], qimg_format)  # , img.close
    return QImageWithBuffer(qimg, b, meta)


def read_qt_pixmap(path: str) -> QtGui.QPixmap:
    return read_qt_image(path).get_pixmap()


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
        self._scale_factor = 1.0

    def _scaled_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap:
        assert self.pm is not None
        return self.pm.scaled(size * self.scale_factor, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

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

    def transform(self, tr: QtGui.QTransform) -> None:
        assert self.pm is not None
        self.setPixmap(self.pm.transformed(tr))

    # qt funcs

    def clear(self) -> None:
        super().clear()
        self.pm = None

    def setPixmap(self, pm: QtGui.QPixmap) -> None:
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

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.label = AspectRatioPixmapLabel(parent=parent)
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setWidget(self.label)

        # set properties
        self._fixed_size = None  # set first
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

    @property
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

    def setPixmap(self, pm: QtGui.QPixmap) -> None:
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