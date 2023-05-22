import logging
from datetime import datetime, time, timezone
from os import fspath
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import piexif
from genutility.filesystem import scandir_ext
from genutility.pillow import NoActionNeeded, fix_orientation, write_text
from PIL import Image

from picturetool.utils import extensions_jpeg, get_exif_dates

logger = logging.getLogger(__name__)


class NoDateFound(Exception):
    pass


class RotateFailed(Exception):
    pass


def dt_gps_from_exif(exif: dict) -> datetime:
    try:
        datestamp = exif["GPS"][piexif.GPSIFD.GPSDateStamp].decode("ascii")
        timestamp = exif["GPS"][piexif.GPSIFD.GPSTimeStamp]
    except KeyError:
        raise NoDateFound()
    except UnicodeDecodeError:
        raise

    _date = datetime.strptime(datestamp, "%Y:%m:%d")
    (h, hd), (m, md), (s, sd) = timestamp
    assert hd == 1 and md == 1
    s, ms = divmod(s, sd)
    _time = time(h, m, s, ms * 1000)

    return datetime.combine(_date, _time, timezone.utc)


def get_original_date(
    exif: dict, aslocal: bool = True, sources: Iterable[str] = ("exif-original", "exif-digitized", "gps")
) -> datetime:
    """Returns the original picture date from exif date or raises `NoDateFound`.

    Input:
            `exif`: piexif exif info
            `offset`: timezone offset in hours
    """

    dates = get_exif_dates(exif)

    sourcemap = {
        "exif-original": dates.get("original"),
        "exif-digitized": dates.get("digitized"),
        "gps": dt_gps_from_exif,
    }

    for s in sources:
        val = sourcemap[s]
        if val is None:  # skip
            pass
        elif isinstance(val, datetime):
            dt = val
            break
        else:  # callable
            try:
                dt = val(exif)
                break
            except NoDateFound:
                pass
    else:
        raise NoDateFound()

    if aslocal:
        return dt.astimezone(None)
    else:
        return dt


def add_date(
    image: Image,
    align: str = "BR",
    fontsize: float = 0.03,
    padding: float = 0.01,
    fillcolor: str = "white",
    outlinecolor: str = "black",
) -> Tuple[Image, Dict[str, Any]]:
    try:
        exif = piexif.load(image.info["exif"])
    except KeyError as e:
        raise NoDateFound(e)

    dt = get_original_date(exif).date()

    try:
        image = fix_orientation(image, exif)
    except NoActionNeeded:
        pass

    write_text(image, dt.isoformat(), align, fillcolor, outlinecolor, fontsize, padding)
    kwargs = {
        "exif": piexif.dump(exif),
    }

    return image, kwargs


def mod_image(inpath: Path, outpath: Path, args: Any, quality: int = 90, move: Optional[str] = None) -> bool:
    if inpath.resolve() == outpath.resolve():
        raise ValueError("inpath cannot be equal to output")

    image = Image.open(fspath(inpath))
    kwargs = {
        "quality": quality,
        "optimize": True,
        "progressive": image.info.get("progression", False),
        "icc_profile": image.info.get("icc_profile"),
        "exif": image.info.get("exif"),
    }
    modified = False

    try:
        if args.resize:
            image.thumbnail(args.maxsize, reducing_gap=None)  # inplace
            modified = True

        if args.add_date:  # implies rotate
            image, kwargs = add_date(image, args.align, args.fontsize, args.padding, args.fill, args.outline)
            kwargs.update(kwargs)
            modified = True

        elif args.rotate:
            # warning: this is not lossless

            try:
                exif = piexif.load(kwargs["exif"])
            except KeyError as e:
                raise RotateFailed(e)

            try:
                image = fix_orientation(image, exif)
                kwargs.update(
                    {
                        "exif": piexif.dump(exif),
                    }
                )
                modified = True
            except ValueError:
                raise RotateFailed()
            except NoActionNeeded:
                pass

        if not modified:
            return False

        image.save(fspath(outpath), **kwargs)

    finally:
        image.close()

    if move:
        if outpath.exists():
            destdir = inpath.parent / move
            destfile = destdir / inpath.name
            if destfile.exists():
                logger.error("Cannot move %s to %s. Destination already exists.", inpath, destfile)
            else:
                destdir.mkdir(exist_ok=True)
                inpath.rename(destfile)
        else:
            logger.critical("Bad error! Save succeeded, but %s doesn't exist", outpath)

    return True


def main():
    from argparse import ArgumentDefaultsHelpFormatter

    from genutility.args import is_dir
    from gooey import GooeyParser

    DEFAULT_QUALITY = 90

    parser = GooeyParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=is_dir, help="Directory with image files", widget="DirChooser")

    parser.add_argument("--extensions", nargs="+", default=extensions_jpeg)
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directory recursively.")
    parser.add_argument("-q", "--quality", type=int, default=DEFAULT_QUALITY, help="JPEG quality level.")
    parser.add_argument(
        "--move", type=str, default=None, help="Move original files to this subdirectory after processing."
    )

    # actions
    parser.add_argument("--resize", action="store_true", help="Downsize image.")
    parser.add_argument("--add-date", action="store_true", help="Add date string to image. Implies --rotate.")
    parser.add_argument("--rotate", action="store_true", help="Rotate image according on exif info.")

    ALIGN_DEFAULT = "BR"
    FONTSIZE_DEFAULT = 0.03
    PADDING_DEFAULT = 0.01
    FILL_DEFAULT = "white"
    OUTLINE_DEFAULT = "black"

    parser.add_argument(
        "-a",
        "--align",
        choices=("TL", "TC", "TR", "BL", "BC", "BR"),
        default=ALIGN_DEFAULT,
        help="The corner alignment of the date string. TL is top left, BC is bottom center, and so on.",
    )
    parser.add_argument(
        "-p", "--fontsize", type=float, default=FONTSIZE_DEFAULT, help="Fontsize ratio relative to the image height"
    )
    parser.add_argument(
        "--padding", type=float, default=PADDING_DEFAULT, help="Padding ratio relative to the image size"
    )
    parser.add_argument("--fill", default=FILL_DEFAULT, help="Font fill color")
    parser.add_argument("--outline", default=OUTLINE_DEFAULT, help="Font outline color")
    # parser.add_argument("--maxsize", metavar=("W", "H"), nargs=2, type=int, help="Downsize so the images dimensions don't exceed W x H")  # fails with Gooey
    parser.add_argument(
        "--maxsize", metavar="W H", nargs=2, type=int, help="Downsize so the images dimensions don't exceed W x H"
    )
    args = parser.parse_args()

    if args.maxsize and not args.resize:
        parser.error("--maxsize needs --resize")

    if (
        args.align != ALIGN_DEFAULT
        or args.fontsize != FONTSIZE_DEFAULT
        or args.padding != PADDING_DEFAULT
        or args.fill != FILL_DEFAULT
        or args.outline != OUTLINE_DEFAULT
    ) and not args.add_date:
        parser.error("--add-date needed")

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    it = scandir_ext(args.path, args.extensions, rec=args.recursive)

    suffix = ".pp"

    for entry in it:
        path = Path(entry)
        new_suffix = suffix + path.suffix
        outpath = path.with_suffix(new_suffix)

        if outpath.exists() or path.suffix == new_suffix:
            logger.warning("Skipping %s (already exists)", entry)
        else:
            try:
                if mod_image(path, outpath, args, args.quality, args.move):
                    logger.info("Saved %s", outpath)
                else:
                    logger.info("Unmodified %s", outpath)

            except NoDateFound:
                logger.warning("No date found for %s", entry)
            except RotateFailed:
                logger.warning("Could not auto-rotate %s", entry)
            except Exception:
                logger.exception("Processing %s failed", entry)


if __name__ == "__main__":
    main()
