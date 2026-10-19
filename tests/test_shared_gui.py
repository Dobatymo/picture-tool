import struct
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import piexif
from genutility.test import MyTestCase
from piexif._common import split_into_segments
from PIL import Image

from picturetool.shared_gui import _update_jpeg_exif, jpeg_fix_orientation, read_qt_image, rotate_save


class SharedGuiTest(MyTestCase):
    def test_rotation_preserves_unrelated_metadata(self):
        for thumbnail in (False, True):
            for orientation in (1, 6):
                with self.subTest(thumbnail=thumbnail, orientation=orientation), TemporaryDirectory() as folder:
                    thumb = BytesIO()
                    Image.new("RGB", (32, 16), "blue").save(thumb, format="JPEG")
                    exif = {
                        "0th": {piexif.ImageIFD.Orientation: orientation, piexif.ImageIFD.Make: b"private-value"},
                        "Exif": {piexif.ExifIFD.MakerNote: b"opaque maker note data"},
                    }
                    if thumbnail:
                        exif.update({"1st": {}, "thumbnail": thumb.getvalue()})
                    raw = bytearray(piexif.dump(exif))
                    # Turn a known ASCII tag into an unknown tag without moving its value.
                    offset = 6 + struct.unpack_from(">I", raw, 10)[0]
                    count = struct.unpack_from(">H", raw, offset)[0]
                    for pos in range(offset + 2, offset + 2 + 12 * count, 12):
                        if struct.unpack_from(">H", raw, pos)[0] == piexif.ImageIFD.Make:
                            struct.pack_into(">H", raw, pos, 65000)
                    source = Path(folder) / "source.jpg"
                    Image.new("RGB", (32, 16), "red").save(
                        source, exif=bytes(raw), dpi=(300, 300), icc_profile=b"test ICC profile"
                    )
                    original = source.read_bytes()
                    rotate_save(source, "cw")
                    output = source.with_suffix(".rotated.jpg").read_bytes()
                    with Image.open(BytesIO(output)) as image:
                        self.assertEqual(image.info["dpi"], (300, 300))
                        self.assertEqual(image.info["icc_profile"], b"test ICC profile")
                        self.assertEqual(image.getexif()[65000], "private-value")
                        self.assertEqual(image.getexif()[274], 1)
                    result_exif = piexif.load(output)
                    self.assertEqual(result_exif["Exif"][piexif.ExifIFD.MakerNote], b"opaque maker note data")
                    if thumbnail:
                        with Image.open(BytesIO(result_exif["thumbnail"])) as image:
                            self.assertEqual(image.size, (16, 32) if orientation == 1 else (32, 16))

                    # All metadata segments except EXIF must remain byte-for-byte intact.
                    def other_metadata(data):
                        return [seg for seg in split_into_segments(data) if seg[:2] in (b"\xff\xe0", b"\xff\xe2")]

                    self.assertEqual(other_metadata(output), other_metadata(original))

    def test_rotation_without_exif_preserves_jfif(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.jpg"
            Image.new("RGB", (32, 16)).save(source, dpi=(300, 300))
            rotate_save(source, "cw")
            with Image.open(source.with_suffix(".rotated.jpg")) as image:
                self.assertEqual(image.size, (16, 32))
                self.assertEqual(image.info["dpi"], (300, 300))
                self.assertNotIn("exif", image.info)

    def test_exif_patch_preserves_offsets_and_byte_orders(self):
        for endian, magic in [("<", b"II"), (">", b"MM")]:
            with self.subTest(endian=endian):
                # One orientation entry; arbitrary opaque bytes follow the directory.
                raw = (
                    magic
                    + struct.pack(endian + "HIH", 42, 8, 1)
                    + struct.pack(endian + "HHI", 274, 3, 1)
                    + struct.pack(endian + "H", 6)
                    + b"\x00\x00"
                    + struct.pack(endian + "I", 0)
                    + b"opaque trailing bytes"
                )
                data = BytesIO()
                Image.new("RGB", (32, 16)).save(data, format="JPEG", exif=b"Exif\x00\x00" + raw)
                result = _update_jpeg_exif(data.getvalue(), 1, None)
                with Image.open(BytesIO(result)) as image:
                    updated = image.info["exif"][6:]
                expected = bytearray(raw)
                struct.pack_into(endian + "H", expected, 18, 1)
                self.assertEqual(updated, expected)

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
