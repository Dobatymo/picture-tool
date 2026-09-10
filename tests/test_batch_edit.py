from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import piexif
from PIL import Image

from scripts.batch_edit import dt_gps_from_exif, mod_image


class BatchEditTest(TestCase):
    def test_date_stamp_preserves_save_options(self):
        exif = {
            "0th": {piexif.ImageIFD.Orientation: 1},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:09:10 12:00:00"},
        }
        args = SimpleNamespace(
            resize=False,
            add_date=True,
            rotate=False,
            align="BR",
            fontsize=0.03,
            padding=0.01,
            fill="white",
            outline="black",
        )
        with TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.jpg", Path(folder) / "output.jpg"
            Image.new("RGB", (32, 16), "red").save(
                source, exif=piexif.dump(exif), icc_profile=b"test-profile", progressive=True
            )
            original_save = Image.Image.save
            options = {}

            def capture_save(image, path, **kwargs):
                options.update(kwargs)
                return original_save(image, path, **kwargs)

            # Font rendering is independent of the save-options regression.
            with patch("scripts.batch_edit.write_text"), patch.object(Image.Image, "save", capture_save):
                self.assertTrue(mod_image(source, output, args, quality=95))
            self.assertEqual(options["quality"], 95)
            self.assertTrue(options["optimize"])
            with Image.open(output) as image:
                self.assertEqual(image.info["icc_profile"], b"test-profile")
                self.assertTrue(image.info["progressive"])
                self.assertEqual(image.getexif()[piexif.ImageIFD.Orientation], 1)

    def test_gps_fractional_seconds(self):
        for seconds, expected in [((56, 1), 0), ((113, 2), 500000), ((561234, 10000), 123400)]:
            with self.subTest(seconds=seconds):
                exif = {
                    "GPS": {
                        piexif.GPSIFD.GPSDateStamp: b"2026:09:10",
                        piexif.GPSIFD.GPSTimeStamp: ((12, 1), (34, 1), seconds),
                    }
                }
                date = dt_gps_from_exif(exif)
                self.assertEqual((date.hour, date.minute, date.second, date.microsecond), (12, 34, 56, expected))

    def test_batch_rotate_without_exif(self):
        with TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.jpg", Path(folder) / "output.jpg"
            Image.new("RGB", (32, 16)).save(source)
            args = SimpleNamespace(resize=False, add_date=False, rotate=True, maxsize=(16, 16))
            self.assertFalse(mod_image(source, output, args))
            self.assertFalse(output.exists())
            args.resize = True
            self.assertTrue(mod_image(source, output, args))
            with Image.open(output) as image:
                self.assertEqual(image.size, (16, 8))
