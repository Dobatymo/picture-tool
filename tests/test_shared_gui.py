from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import piexif
from genutility.test import MyTestCase
from PIL import Image

from picturetool.shared_gui import adjust_gamma, jpeg_fix_orientation, read_qt_image


class SharedGuiTest(MyTestCase):
    def test_jpeg_orientation_without_thumbnail(self):
        with BytesIO() as source:
            Image.new("RGB", (32, 16), "red").save(
                source, format="JPEG", exif=piexif.dump({"0th": {piexif.ImageIFD.Orientation: 6}})
            )
            result = jpeg_fix_orientation(source.getvalue())
        with Image.open(BytesIO(result)) as image:
            self.assertEqual(image.size, (16, 32))
        self.assertEqual(piexif.load(result)["0th"][piexif.ImageIFD.Orientation], 1)
        self.assertEqual(jpeg_fix_orientation(result), result)

    def test_gps_hemisphere_metadata(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "gps.jpg"
            for lat_ref, lon_ref in [(b"N", b"E"), (b"S", b"W")]:
                with self.subTest(lat_ref=lat_ref, lon_ref=lon_ref):
                    exif = {
                        "GPS": {
                            piexif.GPSIFD.GPSLatitude: ((0, 1), (30, 1), (0, 1)),
                            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
                            piexif.GPSIFD.GPSLongitude: ((151, 1), (12, 1), (0, 1)),
                            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
                        }
                    }
                    Image.new("RGB", (32, 16)).save(source, exif=piexif.dump(exif))
                    image = read_qt_image(source)
                    self.assertEqual(image.meta["gps-lat-ref"], lat_ref.decode("ascii"))
                    self.assertEqual(image.meta["gps-lon-ref"], lon_ref.decode("ascii"))
                    self.assertEqual(image.meta["gps-lat"], (0, 30, 0))
                    self.assertEqual(image.meta["gps-lon"], (151, 12, 0))
