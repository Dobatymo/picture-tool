from datetime import datetime, timezone, timedelta, time

from PIL import Image
import piexif

from genutility.pillow import fix_orientation, write_text, NoActionNeeded

def get_original_date(exif, offset=8):
	# type: (dict, int) -> datetime

	""" Returns the original picture date from exif date.
		Input:
			`exit`: piexif exit_dict
			`offset`: timezone offset in hours
	"""

	try:
		datestamp = exif["GPS"][piexif.GPSIFD.GPSDateStamp].decode("ascii")
		timestamp = exif["GPS"][piexif.GPSIFD.GPSTimeStamp]
		_date = datetime.strptime(datestamp, "%Y:%m:%d")
		(h, hd), (m, md), (s, sd) = timestamp
		assert hd == 1 and md == 1
		s, ms = divmod(s, sd)
		_time = time(h, m, s, ms*1000)
		dt = datetime.combine(_date, _time, timezone.utc)
		return dt.astimezone(timezone(timedelta(hours=offset)))

	except KeyError:
		pass

	original = exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("ascii")
	dt = datetime.strptime(original, "%Y:%m:%d %H:%M:%S") # local time
	return dt.astimezone(timezone(timedelta(hours=offset)))

def center_by_topleft(tl_1, size_1, size_2):
	center_1 = (tl_1[0] + size_1[0] / 2, tl_1[1] + size_1[1] / 2)
	tl_2 = (center_1[0] - size_2[0] / 2, center_1[1] - size_2[1] / 2)
	return tl_2

class NoDateFound(Exception):
	pass

def save_with_date(inpath, outpath, align="BR", fillcolor="white", outlinecolor="black", fontratio=0.03, quality=90):
	# type: (Path, Path, str, str, str, float, int) -> None

	assert not outpath.exists()

	image = Image.open(str(inpath))
	try:
		exif = piexif.load(image.info["exif"])
		try:
			dt = get_original_date(exif).date()
		except KeyError:
			raise NoDateFound()

		try:
			image, exif = fix_orientation(image, exif)
		except NoActionNeeded:
			pass

		write_text(image, dt.isoformat(), align, fillcolor, outlinecolor, fontratio, padding=(10, 10))
		kwargs = {
			"quality": quality,
			"optimize": True,
			"progressive": image.info.get("progression", False),
			"icc_profile": image.info.get("icc_profile"),
			"exif": piexif.dump(exif), 
		}
		image.save(str(outpath), **kwargs)

	finally:
		image.close()

if __name__ == "__main__":
	from pathlib import Path
	from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("path", help="directory with .jpg files")
	parser.add_argument("--align", choices=("TL", "TC", "TR", "BL", "BC", "BR"), default="BR", help="The corner alignment of the date string. TL is top left, BC is bottom center, and so on.")
	parser.add_argument("--quality", default=90, help="JPEG quality level")
	parser.add_argument("--fill", default="white", help="Font fill color")
	parser.add_argument("--outline", default="black", help="Font outline color")
	parser.add_argument("--ratio", default=0.03, type=float, help="Fontsize ratio in percent of the image height")
	args = parser.parse_args()

	for entry in Path(args.path).glob("*.jpg"):
		try:
			outpath = entry.with_suffix(".withdate.jpg")
			save_with_date(entry, outpath, align=args.align, quality=args.quality)
			print("Saved", outpath)
		except NoDateFound:
			print("No date found for", entry)
