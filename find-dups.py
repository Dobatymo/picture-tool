import csv
import hashlib
import logging
import multiprocessing
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import timedelta
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Callable, Collection, Container, DefaultDict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import humanize
import imagehash
import msgpack
import numpy as np
import piexif
from appdirs import user_data_dir
from genutility.args import is_dir, suffix_lower
from genutility.datetime import datetime_from_utc_timestamp_ns
from genutility.file import StdoutFile
from genutility.filesdb import FileDbSimple, NoResult
from genutility.filesystem import entrysuffix, scandir_rec
from genutility.hash import hash_file
from genutility.image import normalize_image_rotation
from genutility.iter import progress
from genutility.numpy import hamming_dist_packed
from genutility.time import MeasureTime
from genutility.typing import CsvWriter
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from PIL.IptcImagePlugin import getiptcinfo

from npmp import ChunkedParallel, SharedNdarray

HEIF_EXTENSIONS = (".heic", ".heif")
JPEG_EXTENSIONS = (".jpg", ".jpeg")

Shape = Tuple[int, ...]


class HashDB(FileDbSimple):
    @classmethod
    def derived(cls):
        return [
            ("file_sha256", "BLOB", "?"),
            ("image_sha256", "BLOB", "?"),
            ("phash", "BLOB", "?"),
            ("exif", "BLOB", "?"),
            ("icc_profile", "BLOB", "?"),
            ("iptc", "BLOB", "?"),
            ("photoshop", "BLOB", "?"),
        ]

    def __init__(self, path: str) -> None:
        FileDbSimple.__init__(self, path, "picture-hashes")


def hash_file_hash(path: str) -> bytes:
    return hash_file(path, hashlib.sha256).digest()  # 32bytes


class hash_image_hash:
    def __init__(self, normalize: Container[str], resolution: Tuple[int, int]) -> None:
        self.normalize = normalize
        self.resolution = resolution

    def __call__(self, path: str) -> bytes:
        with Image.open(path, "r") as img:
            if "orientation" in self.normalize:
                img_gray = ImageOps.grayscale(img)
                img = Image.fromarray(normalize_image_rotation(np.asarray(img), np.asarray(img_gray)))
            if "resolution" in self.normalize:
                img = img.resize(self.resolution, resample=Image.Resampling.LANCZOS)
            if "colors" in self.normalize:
                img = img.filter(ImageFilter.SMOOTH)
            img_bytes = np.asarray(img).tobytes()
        m = hashlib.sha256()
        m.update(img_bytes)
        return m.digest()  # 32bytes


def hash_phash(path: str) -> bytes:
    with Image.open(path, "r") as img:
        return bytes.fromhex(str(imagehash.phash(img, hash_size=16)))  # 32bytes


class ImageError(Exception):
    pass


class wrap:
    def __init__(self, func: Callable[[str], bytes], colname: str) -> None:
        self.func = func
        self.colname = colname

    def get_cols(self) -> Tuple[str, ...]:
        return (self.colname, "width", "height", "exif")

    def _get_file_meta(self, path: Path) -> dict:
        try:
            img_hash = self.func(os.fspath(path))
        except UnidentifiedImageError:
            raise  # inherits from OSError, so must be re-raised explicitly
        except OSError as e:
            raise ImageError(path, e)

        with Image.open(path) as img:
            width, height = img.size
            exif = img.info.get("exif", None)
            icc_profile = img.info.get("icc_profile", None)
            iptc = getiptcinfo(img)
            photoshop = img.info.get("photoshop", None)

        iptc_ = msgpack.packb(iptc, use_bin_type=True)
        photoshop_ = msgpack.packb(photoshop, use_bin_type=True)

        assert isinstance(img_hash, bytes)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(exif, (bytes, type(None)))
        assert isinstance(icc_profile, (bytes, type(None)))
        assert isinstance(iptc_, (bytes, type(None)))
        assert isinstance(photoshop_, (bytes, type(None)))

        return {
            self.colname: img_hash,
            "width": width,
            "height": height,
            "exif": exif,
            "icc_profile": icc_profile,
            "iptc": iptc_,
            "photoshop": photoshop_,
        }


class wrap_without_db(wrap):
    def __call__(self, path: Path) -> Tuple[bool, Path, dict]:
        cached = False
        meta = self._get_file_meta(path)
        return cached, path, meta


class wrap_with_db(wrap):
    def __init__(self, func: Callable[[str], bytes], colname: str, db: HashDB, overwrite: bool = False) -> None:
        wrap.__init__(self, func, colname)
        self.only = self.get_cols()
        self.db = db
        self.overwrite = overwrite

    def __call__(self, path: Path) -> Tuple[bool, Path, dict]:
        try:
            if self.overwrite:
                raise NoResult
            values = self.db.get(path, only=self.only)
            meta = dict(zip(self.only, values))
            if meta[self.colname] is None:
                raise NoResult
            cached = True
        except NoResult:
            meta = self._get_file_meta(path)
            cached = False

        return cached, path, meta


hash_cols = {"file-hash": "file_sha256", "image-hash": "image_sha256", "phash": "phash"}
hash_metrics = {"file-hash": "equivalence", "image-hash": "equivalence", "phash": "hamming"}


def scandir_error_log_warning(entry: os.DirEntry, exception) -> None:
    logging.warning("<%s> %s: %s", entry.path, type(exception).__name__, exception)


def initializer_worker(extensions: Collection[str]) -> None:
    if set(HEIF_EXTENSIONS) & set(extensions):
        from pillow_heif import register_heif_opener

        register_heif_opener()


def hamming_duplicates_chunk(
    a_arr: np.ndarray, b_arr: np.ndarray, coords: Shape, axis: int, hamming_threshold: int
) -> np.ndarray:
    dists = hamming_dist_packed(a_arr, b_arr, axis)
    return np.argwhere(np.triu(dists <= hamming_threshold, 1)) + np.array(coords)


def hamming_duplicates(sharr: SharedNdarray, chunkshape: Shape, hamming_threshold: int) -> Collection[np.ndarray]:
    if len(sharr.shape) != 2 or sharr.dtype != np.uint8:
        raise ValueError("Input must be a list of packed hashes (2-dimensional byte array)")

    a_arr = sharr.reshape((1, sharr.shape[0], sharr.shape[1]))
    b_arr = sharr.reshape((sharr.shape[0], 1, sharr.shape[1]))

    return ChunkedParallel(
        hamming_duplicates_chunk, a_arr, b_arr, chunkshape, axis=-1, hamming_threshold=hamming_threshold
    )


def buffer_fill(it: Iterable[bytes], buffer: memoryview) -> None:
    pos = 0
    for buf in it:
        size = len(buf)
        buffer[pos : pos + size] = buf
        pos += size


def maybe_decode(s: Optional[bytes], encoding: str = "ascii") -> Optional[str]:
    if s is None:
        return None
    else:
        return s.decode(encoding)


def main() -> None:

    APP_NAME = "picture-tool"
    APP_AUTHOR = "Dobatymo"
    DEFAULT_APPDATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))

    DEFAULT_EXTENSIONS = JPEG_EXTENSIONS + HEIF_EXTENSIONS + (".png",)
    DEFAULT_HASHDB = DEFAULT_APPDATA_DIR / "hashes.sqlite"
    DEFAULT_NORMALIZATION_OPS = ("orientation", "resolution", "colors")
    DEFAULT_NORMALIZED_RESOLUTION = (256, 256)
    DEFAULT_PARALLEL_READ = multiprocessing.cpu_count()
    DESCRIPTION = "Find exact image duplicates"

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description=DESCRIPTION)
    parser.add_argument("directories", metavar="DIR", nargs="+", type=is_dir, help="Input directories")
    parser.add_argument(
        "--extensions",
        metavar=".EXT",
        type=suffix_lower,
        default=DEFAULT_EXTENSIONS,
        nargs="+",
        help="Image file extensions",
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="Read directories recursively")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--mode",
        choices=["file-hash", "image-hash", "phash"],
        default="image-hash",
        help="Hashing mode. `file-hash` simply hashes the whole file. `image-hash` hashes the uncompressed image data of the file and normalizes the rotation. `phash` calculates the perceptual hash of the image.",
    )
    parser.add_argument(
        "--hashdb",
        metavar="PATH",
        type=Path,
        default=DEFAULT_HASHDB,
        help="Path to sqlite database file to store hashes.",
    )
    parser.add_argument(
        "--normalize",
        metavar="OP",
        default=DEFAULT_NORMALIZATION_OPS,
        nargs="+",
        help="Normalization operations. Ie. when orientation is normalized, files with different orientations can be detected as duplicates",
    )
    parser.add_argument(
        "--resolution-normalized",
        default=DEFAULT_NORMALIZED_RESOLUTION,
        nargs=2,
        type=int,
        help="All pictures will be resized to this resolution prior to comparison. It should be smaller than the smallest picture in one duplicate group. If it's smaller, more differences in image details will be ignored.",
    )
    parser.add_argument(
        "--parallel-read",
        default=DEFAULT_PARALLEL_READ,
        type=int,
        help="Default read concurrency",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2000,
        help="Specifies the number of hashes to compare at the same the time. Larger chunksizes require more memory.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write results to file. Otherwise they are written to stdout."
    )
    parser.add_argument("--overwrite-cache", action="store_true", help="Update cached values")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    def pathiter(directories: Iterable[Path], recursive: bool) -> Iterator[os.DirEntry]:
        for directory in directories:
            yield from scandir_rec(
                directory, files=True, dirs=False, rec=recursive, errorfunc=scandir_error_log_warning
            )

    if args.hashdb:
        args.hashdb.parent.mkdir(parents=True, exist_ok=True)
        db = HashDB(args.hashdb)
        db.connection.execute("PRAGMA journal_mode=WAL;")
    else:
        db = None

    hash_funcs = {
        "file-hash": hash_file_hash,
        "image-hash": hash_image_hash(args.normalize, args.resolution_normalized),
        "phash": hash_phash,
    }
    metric = hash_metrics[args.mode]

    if metric == "equivalence":
        hashes: DefaultDict[bytes, Set[str]] = defaultdict(set)
    elif metric == "hamming":
        hashes: List[bytes] = []
        paths: List[str] = []

    hash_func = hash_funcs[args.mode]
    colname = hash_cols[args.mode]
    if db is not None:
        hash_func = wrap_with_db(hash_func, colname, db, args.overwrite_cache)
    else:
        hash_func = wrap_without_db(hash_func, colname)

    with ProcessPoolExecutor(
        args.parallel_read, initializer=initializer_worker, initargs=(args.extensions,)
    ) as executor, MeasureTime() as stopwatch:
        futures: List[Future] = []
        cached: bool
        path: Path
        img_hash: bytes

        for entry in progress(
            pathiter(args.directories, args.recursive), extra_info_callback=lambda total, length: "Finding files"
        ):
            if entrysuffix(entry).lower() not in args.extensions:
                continue

            futures.append(executor.submit(hash_func, Path(entry)))

        logging.info("Analyzing %d files", len(futures))

        num_cached = 0
        num_fresh = 0
        num_error = 0
        for future in progress(
            as_completed(futures), length=len(futures), extra_info_callback=lambda total, length: "Computing hashes"
        ):
            try:
                cached, path, meta = future.result()
            except ImageError as e:
                _path = e.args[0]
                _e = e.args[1]
                logging.warning("%s <%s>: %s", type(_e).__name__, _path, _e)
                num_error += 1
                continue
            except UnidentifiedImageError as e:
                logging.warning("%s: %s", type(e).__name__, e)
                num_error += 1
                continue
            except Exception:
                logging.exception("Failed to hash image file")
                num_error += 1
                continue

            img_hash = meta[colname]
            rawpath = os.fspath(path)

            if db is not None and not cached:
                db.add(path, derived=meta, commit=False, replace=False)

            if metric == "equivalence":
                hashes[img_hash].add(rawpath)
            elif metric == "hamming":
                paths.append(rawpath)
                hashes.append(img_hash)

            if cached:
                num_cached += 1
            else:
                num_fresh += 1

        time_delta = humanize.naturaldelta(timedelta(seconds=stopwatch.get()))
        logging.info(
            "Loaded %d hashes from cache, computed %d fresh ones and failed to read %d in %s",
            num_cached,
            num_fresh,
            num_error,
            time_delta,
        )

    if db is not None:
        db.commit()

    def write_dup(fw: CsvWriter, i: int, path: Union[str, Path]) -> None:
        if db is not None:
            if isinstance(path, Path):
                _path = path
            else:
                _path = Path(path)

            filesize, mod_date, exif = db.get(_path, only=("filesize", "mod_date", "exif"))
            dt = datetime_from_utc_timestamp_ns(mod_date, aslocal=True)

            if exif is None:
                exif_date = None
                exif_maker = None
                exif_model = None
            else:
                d = piexif.load(exif)
                exif_date = maybe_decode(d["Exif"].get(36867, None))
                exif_maker = maybe_decode(d["0th"].get(271, None))
                exif_model = maybe_decode(d["0th"].get(272, None))

            fw.writerow([i, path, filesize, dt.isoformat(), exif_date, exif_maker, exif_model])
        else:
            fw.writerow([i, path])

    if not hashes:
        return

    if metric == "equivalence":
        hashes = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}

        with StdoutFile(args.out, "wt", newline="") as fw:
            writer = csv.writer(fw)
            for i, (k, paths) in enumerate(hashes.items(), 1):
                for path in paths:
                    write_dup(writer, i, path)

    elif metric == "hamming":
        hamming_threshold = 1

        sharr = SharedNdarray.create((len(hashes), len(hashes[0])), np.uint8)
        buffer_fill(hashes, sharr.getbuffer())
        chunkshape = (args.chunksize, args.chunksize)
        dups = np.concatenate(
            list(
                progress(
                    hamming_duplicates(sharr, chunkshape, hamming_threshold),
                    extra_info_callback=lambda total, length: "Matching hashes",
                )
            )
        )

        idx = np.argsort(dups[:, 0])
        dups = dups[idx]

        with StdoutFile(args.out, "wt", newline="") as fw:
            writer = csv.writer(fw)
            for i, (key, group) in enumerate(groupby(dups, key=itemgetter(0)), 1):
                write_dup(writer, i, paths[key])
                for _, second in group:
                    write_dup(writer, i, paths[second])


if __name__ == "__main__":
    main()
