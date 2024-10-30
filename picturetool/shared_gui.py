import logging
import os
from copy import deepcopy
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import piexif
import pillow_avif  # noqa: F401
import rawpy
import turbojpeg
from genutility.pillow import NoActionNeeded, fix_orientation
from piexif._load import _ExifReader
from PIL import Image
from pillow_heif import register_heif_opener
from typing_extensions import Self

from .utils import extensions_raw

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtGui import QAction
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets
    from PySide2.QtWidgets import QAction

register_heif_opener()

logger = logging.getLogger(__name__)


class ImperfectTransform(Exception):
    pass


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

    @property
    def width(self) -> int:
        return self.image.width()

    @property
    def height(self) -> int:
        return self.image.height()


def piexif_get(d: Dict[str, Dict[int, Any]], idx1: str, idx2: int, dtype: str) -> Any:
    try:
        val = d[idx1][idx2]
    except KeyError:
        return None

    try:
        if dtype == "ascii":
            return val.decode("ascii").rstrip("\0")
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

mode_bits_per_channel = {
    "1": 1,
    "L": 8,
    "P": 8,
    "RGB": 8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "LAB": 8,
    "HSV": 8,
    "LA": 8,
    "PA": 8,
    "RGBX": 8,
    "RGBa": 8,
    "I;16": 16,
    "BGR;24": 8,
}


def adjust_gamma(img: Image.Image, inv_gamma_in: float, inv_gamma_out: float = 1 / 2.2) -> None:
    import cv2

    if img.mode in ("L", "RGB", "BGR;24", "RGBX"):
        pass
    elif img.mode in ("RGBA", "RGBa"):  # alpha channel should be unaffected by gamma according to PNG specs
        raise ValueError(f"Unsupported image mode: {img.mode!r}")
    else:
        raise ValueError(f"Unsupported image mode: {img.mode!r}")

    max_int = 2 ** mode_bits_per_channel[img.mode]

    clut = (((np.arange(0, max_int) / (max_int - 1)) ** (inv_gamma_out / inv_gamma_in)) * (max_int - 1)).astype("uint8")
    image_data = Image.fromarray(cv2.LUT(np.array(img), clut), img.mode)
    img.paste(image_data)


def read_qt_image(path: Path, frame: int = 0, process: Optional[Iterable[ImageTransformT]] = None) -> QImageWithBuffer:
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
        "1": QtGui.QImage.Format_Mono,  # fixme: or Format_MonoLSB?
        "L": QtGui.QImage.Format_Grayscale8,
        "I;16": QtGui.QImage.Format_Grayscale16,
        "RGB": QtGui.QImage.Format_RGB888,
        "BGR;24": QtGui.QImage.Format_BGR888,
        "RGBA": QtGui.QImage.Format_RGBA8888,
        "RGBX": QtGui.QImage.Format_RGBX8888,
        "RGBa": QtGui.QImage.Format_RGBA8888_Premultiplied,
    }

    modebits = {
        "1": 1,
        "L": 8,
        "I;16": 16,
        "RGB": 24,
        "BGR;24": 24,
        "RGBA": 32,
        "RGBX": 32,
        "RGBa": 32,
    }

    _path = os.fspath(path)

    meta: Dict[str, Any] = {
        "transforms": [],
    }

    if path.suffix.lower() in extensions_raw:
        with rawpy.imread(_path) as raw:
            rgb = raw.postprocess()  # ndarray[h,w,c]
            meta["transforms"].append("raw-to-rgb")

            """with pyexiv2.Image(path) as metafile:
                exif = metafile.read_exif()
                xmp = metafile.read_xmp()
                icc = metafile.read_icc()"""

            img = Image.fromarray(rgb)
        try:
            r = _ExifReader(_path)
        except piexif.InvalidImageDataError as e:
            logger.debug("Failed to read %r using piexif: %s", _path, e)
        else:
            img.info["exif"] = r.tiftag

    else:
        img = Image.open(_path)

    with img:
        meta["format"] = img.format
        try:
            meta["n_frames"] = getattr(img, "n_frames", 1)
        except TypeError:
            logger.exception("Failed to load `n_frames` from %r", _path)
            meta["n_frames"] = 1

        if meta["n_frames"] > 1 and frame > 0:
            img.seek(frame)

        gamma = img.info.get("gamma", None)
        srgb = img.info.get("srgb", None)

        if srgb is None and gamma is not None:
            try:
                adjust_gamma(img, gamma)
            except ValueError as e:
                logger.warning("Adjusting gamma failed for %r [frame=%d]: %s", _path, frame, e)
            else:
                meta["transforms"].append("gamma")

        elif srgb is not None:
            meta["srgb-rendering-intent"] = srgb

        img.load()  # necessary for PNG exif data to be loaded if available

        if "exif" in img.info:
            exif = piexif.load(img.info["exif"])

            logger.warning("exif.gamma: %s", exif["Exif"].get(piexif.ExifIFD.Gamma, None))

            try:
                img = fix_orientation(img, exif)
            except (NoActionNeeded, KeyError):
                pass
            except ValueError as e:
                logger.warning("Could not fix orientation of <%s> [frame=%d]: %s", _path, frame, e)
            else:
                meta["transforms"].append("rotate")

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
                    logger.debug(
                        "Applying %s to <%s> [frame=%d, mode=%s] failed", func.__name__, _path, frame, img.mode
                    )
                    raise

        if img.mode not in modemap or (img.width * modebits[img.mode]) % 32 != 0:
            # Unsupported image mode or image scanlines not 32-bit aligned
            img = img.convert("RGBA")
            meta["transforms"].append("convert-color-to-rgba")

        # QImage simply references the QByteArray. So you need to keep it around.

        qimg_format = modemap[img.mode]
        b = QtCore.QByteArray(img.tobytes())

    qimg = QtGui.QImage(b, img.size[0], img.size[1], qimg_format)  # , img.close
    return QImageWithBuffer(qimg, b, meta)


def read_qt_pixmap(path: Path) -> QtGui.QPixmap:
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
        if self.pm.size() == new_size:
            self._transforms = []
            return self.pm.pixmap
        else:
            self._transforms = ["scale"]
            return self.pm.pixmap.scaled(
                size * self.scale_factor, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )

    def resize_pixmap(self, size: QtCore.QSize) -> None:
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

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
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

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
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
        logger.debug("%s", self.menu)
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.Context:
            self.menu.show()
        elif reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self.doubleclicked.emit()

    """
    @QtCore.Slot()
    def on_aboutToHide(self):
        logger.debug("called")

    @QtCore.Slot()
    def on_aboutToShow(self):
        logger.debug("called")
    """

    """
    @QtCore.Slot(QAction)
    def on_hovered(self, action):
        logger.debug("%s", action)
    """

    @QtCore.Slot(QAction)
    def on_triggered(self, action) -> None:
        logger.debug("%s", action)


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


class TranslateTjException:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(exc_value, RuntimeError):
            if exc_value.args[0] == "tj3Transform(): Transform is not perfect":
                raise ImperfectTransform("Perfectly lossless rotation not possible")


def _tj_fix_orientation(img: bytes, orientation: int, perfect: bool = True) -> bytes:
    with TranslateTjException():
        if orientation == 1:
            raise NoActionNeeded("File already properly rotated")
        elif orientation == 2:
            img = turbojpeg.transform(img, turbojpeg.OP.HFLIP, perfect=perfect)
        elif orientation == 3:
            img = turbojpeg.transform(img, turbojpeg.OP.ROT180, perfect=perfect)
        elif orientation == 4:
            img = turbojpeg.transform(img, turbojpeg.OP.VFLIP, perfect=perfect)
        elif orientation == 5:
            img = turbojpeg.transform(img, turbojpeg.OP.TRANSPOSE, perfect=perfect)
        elif orientation == 6:
            img = turbojpeg.transform(img, turbojpeg.OP.ROT90, perfect=perfect)
        elif orientation == 7:
            img = turbojpeg.transform(img, turbojpeg.OP.TRANSVERSE, perfect=perfect)
        elif orientation == 8:
            img = turbojpeg.transform(img, turbojpeg.OP.ROT270, perfect=perfect)
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")

    return img


def tj_fix_orientation(img: bytes, exif: dict, perfect: bool = True) -> bytes:
    orientation = exif["0th"][piexif.ImageIFD.Orientation]
    img = _tj_fix_orientation(img, orientation, perfect)
    exif["0th"][piexif.ImageIFD.Orientation] = 1

    return img


def _fix_thumbnail(
    data: bytes,
    exif: dict,
    *,
    orientation: Optional[int] = None,
    op: Optional[turbojpeg.OP] = None,
    perfect: bool = True,
) -> bytes:
    if (orientation is None) == (op is None):
        raise ValueError("Either orientation or op must be given")

    if exif.get("thumbnail") is not None:
        assert exif["thumbnail"], exif["thumbnail"]
        try:
            if orientation is not None:
                exif["thumbnail"] = _tj_fix_orientation(exif["thumbnail"], orientation, perfect)
            elif op is not None:
                with TranslateTjException():
                    exif["thumbnail"] = turbojpeg.transform(exif["thumbnail"], op, perfect)
        except ImperfectTransform:
            img = Image.fromarray(turbojpeg.decompress(data))
            img.thumbnail((160, 120), Image.Resampling.LANCZOS)
            exif["thumbnail"] = turbojpeg.compress(np.array(img), 90, turbojpeg.SAMP.Y420)

        with BytesIO() as out:
            piexif.insert(piexif.dump(exif), data, out)
            data = out.getvalue()

    return data


def jpeg_fix_orientation(data: bytes, perfect: bool = True) -> bytes:
    exif = piexif.load(data)
    try:
        orientation = exif["0th"][piexif.ImageIFD.Orientation]
        data = _tj_fix_orientation(data, orientation, perfect)
        exif["0th"][piexif.ImageIFD.Orientation] = 1

        data = _fix_thumbnail(data, exif, orientation=orientation)

    except NoActionNeeded:
        pass
    except KeyError:
        pass
    except ValueError:
        pass
    return data


def crop_half_save(path: Path, target: str) -> None:
    img = path.read_bytes()

    outpath = path.with_suffix(f".cropped{path.suffix}")
    if outpath.exists():
        raise FileExistsError(outpath)

    img = jpeg_fix_orientation(img)
    info = turbojpeg.decompress_header(img)

    if target == "top":
        x = 0
        y = info["height"] // 2
        y -= y % 16
        width = info["width"]
        height = info["height"] - y
    elif target == "bottom":
        x = 0
        y = 0
        width = info["width"]
        height = info["height"] // 2
    elif target == "left":
        x = info["width"] // 2
        x -= x % 16
        y = 0
        width = info["width"] - x
        height = info["height"]
    elif target == "right":
        x = 0
        y = 0
        width = info["width"] // 2
        height = info["height"]
    else:
        raise ValueError(f"Unsupported target: {target}")

    with TranslateTjException():
        img = turbojpeg.transform(img, crop=True, x=x, y=y, w=width, h=height)

    outpath.write_bytes(img)


def rotate_save(path: Path, target: str) -> None:
    img = path.read_bytes()

    outpath = path.with_suffix(f".rotated{path.suffix}")
    if outpath.exists():
        raise FileExistsError(outpath)

    img = jpeg_fix_orientation(img, perfect=True)
    exif = piexif.load(img)

    if target == "cw":
        op = turbojpeg.OP.ROT90
    elif target == "180":
        op = turbojpeg.OP.ROT180
    elif target == "ccw":
        op = turbojpeg.OP.ROT270
    else:
        raise ValueError(f"Unsupported target: {target}")

    with TranslateTjException():
        img = turbojpeg.transform(img, op, perfect=True)

    img = _fix_thumbnail(img, exif, op=op)

    outpath.write_bytes(img)
