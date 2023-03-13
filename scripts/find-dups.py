import csv
import hashlib
import logging
import multiprocessing
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Container,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import humanize
import imagehash
import msgpack
import numpy as np
import piexif
from genutility.args import is_dir, suffix_lower
from genutility.datetime import datetime_from_utc_timestamp_ns
from genutility.file import StdoutFile
from genutility.filesdb import NoResult
from genutility.filesystem import entrysuffix, scandir_rec
from genutility.hash import hash_file
from genutility.image import normalize_image_rotation
from genutility.iter import progress
from genutility.json import read_json
from genutility.time import MeasureTime
from genutility.typing import CsvWriter, SizedIterable
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from PIL.IptcImagePlugin import getiptcinfo

from picturetool.npmp import ChunkedParallel, SharedNdarray
from picturetool.utils import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_APPDATA_DIR,
    DEFAULT_HASHDB,
    HashDB,
    get_exif_dates,
    hamming_duplicates_chunk,
    make_groups,
    npmp_to_pairs,
)

HEIF_EXTENSIONS = (".heic", ".heif")
JPEG_EXTENSIONS = (".jpg", ".jpeg")
Shape = Tuple[int, ...]


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
        except ValueError as e:
            raise ImageError(path, e)

        with Image.open(path, "r") as img:
            width, height = img.size
            exif = img.info.get("exif", None)
            icc_profile = img.info.get("icc_profile", None)

            try:
                iptc_ = getiptcinfo(img)
            except SyntaxError as e:
                logging.warning("%s <%s>: %s", type(e).__name__, path, e)  # fixme: is this shown?
                iptc = None
            else:
                iptc = msgpack.packb(iptc_, use_bin_type=True)

            try:
                photoshop_ = img.info["photoshop"]
            except KeyError:
                photoshop = None
            else:
                photoshop = msgpack.packb(photoshop_, use_bin_type=True)

        assert isinstance(img_hash, bytes)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(exif, (bytes, type(None)))
        assert isinstance(icc_profile, (bytes, type(None)))
        assert isinstance(iptc, (bytes, type(None)))
        assert isinstance(photoshop, (bytes, type(None)))

        return {
            self.colname: img_hash,
            "width": width,
            "height": height,
            "exif": exif,
            "icc_profile": icc_profile,
            "iptc": iptc,
            "photoshop": photoshop,
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


def hamming_duplicates(sharr: SharedNdarray, chunkshape: Shape, hamming_threshold: int) -> SizedIterable[np.ndarray]:
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


def maybe_decode(
    s: Optional[bytes], encoding: str = "ascii", context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    if s is None:
        return None
    else:
        s = s.rstrip(b"\0")
        try:
            return s.decode(encoding)
        except UnicodeDecodeError as e:
            logging.warning("%s <%s>: %s", type(e).__name__, context, e)
            return s.decode("latin1")  # should never fail


def notify(topic: str, message: str):
    import requests

    requests.post(
        "https://ntfy.sh/",
        json={
            "topic": topic,
            "message": message,
            "title": APP_NAME,
        },
        timeout=60,
    )


def write_dup(db: Optional[HashDB], fw: CsvWriter, i: int, path: Union[str, Path]) -> None:
    if db is not None:
        if isinstance(path, Path):
            _path = path
        else:
            _path = Path(path)

        file_id, device_id, filesize, mod_date, width, height, exif = db.get(
            _path, only=("file_id", "device_id", "filesize", "mod_date", "width", "height", "exif")
        )
        dt = datetime_from_utc_timestamp_ns(mod_date, aslocal=True)

        if exif is None:
            exif_date_modified = None
            exif_date_taken = None
            exif_date_created = None
            exif_maker = None
            exif_model = None
        else:
            d = piexif.load(exif)
            try:
                dates = get_exif_dates(d)
            except ValueError as e:
                logging.warning("Invalid date format <%s>: %s", path, e)
                dates = {}
            exif_date_modified = dates.get("modified")
            exif_date_taken = dates.get("original")
            exif_date_created = dates.get("digitized")

            exif_maker = maybe_decode(d["0th"].get(271, None), context={"path": path, "0th": 271})
            exif_model = maybe_decode(d["0th"].get(272, None), context={"path": path, "0th": 272})

        fw.writerow(
            [
                i,
                path,
                file_id,
                device_id,
                filesize,
                dt.isoformat(),
                width,
                height,
                exif_date_modified.isoformat() if exif_date_modified else None,
                exif_date_taken.isoformat() if exif_date_taken else None,
                exif_date_created.isoformat() if exif_date_created else None,
                exif_maker,
                exif_model,
            ]
        )
    else:
        fw.writerow([i, path])


def pathiter(directories: Iterable[Path], recursive: bool) -> Iterator[os.DirEntry]:
    for directory in directories:
        yield from scandir_rec(directory, files=True, dirs=False, rec=recursive, errorfunc=scandir_error_log_warning)


def get_hash_func(mode: str, db: Optional[HashDB], overwrite_cache: bool, normalize, resolution_normalized) -> wrap:
    hash_funcs = {
        "file-hash": hash_file_hash,
        "image-hash": hash_image_hash(normalize, resolution_normalized),
        "phash": hash_phash,
    }

    hash_func = hash_funcs[mode]
    colname = hash_cols[mode]
    if db is not None:
        hash_func = wrap_with_db(hash_func, colname, db, overwrite_cache)
    else:
        hash_func = wrap_without_db(hash_func, colname)

    return hash_func


HashResultT = Union[Dict[bytes, Set[str]], Tuple[List[bytes], List[str]]]


def get_hashes(
    directories: Iterable[Path],
    recursive: bool,
    extensions,
    hash_func: wrap,
    metric: str,
    db: Optional[HashDB] = None,
    parallel_read: Optional[int] = None,
) -> HashResultT:
    if metric == "equivalence":
        hash2paths: DefaultDict[bytes, Set[str]] = defaultdict(set)
    elif metric == "hamming":
        hashes: List[bytes] = []
        paths: List[str] = []
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    with ProcessPoolExecutor(
        parallel_read, initializer=initializer_worker, initargs=(extensions,)
    ) as executor, MeasureTime() as stopwatch:
        futures: List[Future] = []
        cached: bool
        path: Path
        img_hash: bytes
        inodes: Dict[Tuple[int, int], Set[Path]] = {}

        for entry in progress(
            pathiter(directories, recursive), extra_info_callback=lambda total, length: "Finding files"
        ):
            if entrysuffix(entry).lower() not in extensions:
                continue

            path = Path(entry)
            stats = path.stat()  # `entry.stat()` does not populate all fields on windows

            assert stats.st_nlink > 0

            if stats.st_nlink == 1:  # no other links
                futures.append(executor.submit(hash_func, path))
            else:
                file_id = (stats.st_dev, stats.st_ino)
                try:
                    inodes[file_id].add(path)
                except KeyError:
                    # only scan if this inode group was encountered the first time
                    inodes[file_id] = {path}
                    futures.append(executor.submit(hash_func, path))

        logging.info("Analyzing %d files", len(futures))

        inodes = {k: paths for k, paths in inodes.items() if len(paths) > 1}

        if inodes:
            n_paths = sum(map(len, inodes.values()))
            logging.warning(
                "File collection resulted in %d groups of inodes which are referenced more than once, with a total of %d paths. Only the first encountered path will be considered for duplicate matching.",
                len(inodes),
                n_paths,
            )
        del inodes

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
            except (UnidentifiedImageError, FileNotFoundError) as e:
                logging.warning("%s: %s", type(e).__name__, e)
                num_error += 1
                continue
            except MemoryError as e:
                logging.error("%s: %s", type(e).__name__, e)
                num_error += 1
                continue
            except Exception:
                logging.exception("Failed to hash image file")
                num_error += 1
                continue

            img_hash = meta[hash_func.colname]
            rawpath = os.fspath(path)

            if db is not None and not cached:
                db.add(path, derived=meta, commit=False, replace=False)

            if metric == "equivalence":
                hash2paths[img_hash].add(rawpath)
            elif metric == "hamming":
                paths.append(rawpath)
                hashes.append(img_hash)

            if cached:
                num_cached += 1
            else:
                num_fresh += 1

        time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
        logging.info(
            "Loaded %d hashes from cache, computed %d fresh ones and failed to read %d in %s",
            num_cached,
            num_fresh,
            num_error,
            time_delta,
        )

    if db is not None:
        db.commit()

    if metric == "equivalence":
        return dict(hash2paths)
    elif metric == "hamming":
        return paths, hashes


def get_dupe_groups(
    metric: str, hash_result: HashResultT, chunksize, ntfy_topic: Optional[str] = None
) -> List[List[str]]:
    dupgroups: List[List[str]]

    with MeasureTime() as stopwatch:
        if metric == "equivalence":
            hashes = hash_result
            dupgroups = [list(paths) for digest, paths in hashes.items() if len(paths) > 1]

        elif metric == "hamming":
            paths, hashes = hash_result
            del hash_result
            assert len(paths) == len(hashes)
            if hashes:
                hamming_threshold = 1

                sharr = SharedNdarray.create((len(hashes), len(hashes[0])), np.uint8)
                buffer_fill(hashes, sharr.getbuffer())
                del hashes

                chunkshape = (chunksize, chunksize)
                it = progress(
                    hamming_duplicates(sharr, chunkshape, hamming_threshold),
                    extra_info_callback=lambda total, length: "Matching hashes",
                )
                dups = npmp_to_pairs(it)
                del sharr
                dupgroups = [[paths[idx] for idx in indices] for indices in make_groups(dups)]
                del dups
            else:
                dupgroups = []
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
        logging.info("Found %s duplicate groups in %s", len(dupgroups), time_delta)
        if ntfy_topic:
            notify(ntfy_topic, f"Found {len(dupgroups)} duplicate groups in {time_delta}")

    return dupgroups


def main() -> None:
    try:
        config_path = DEFAULT_APPDATA_DIR / "config.json"
        default_config = {k.replace("-", "_"): v for k, v in read_json(config_path).items()}
        logging.debug("Loaded default arguments from %s", config_path)
    except FileNotFoundError:
        default_config = {}

    DEFAULT_EXTENSIONS = JPEG_EXTENSIONS + HEIF_EXTENSIONS + (".png", ".webp")
    DEFAULT_NORMALIZATION_OPS = ("orientation", "resolution", "colors")
    DEFAULT_NORMALIZED_RESOLUTION = (256, 256)
    DEFAULT_PARALLEL_READ = multiprocessing.cpu_count()
    DESCRIPTION = "Find picture duplicates"

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description=DESCRIPTION)
    parser.add_argument("directories", metavar="DIR", nargs="+", type=is_dir, help="Input directories")
    parser.add_argument(
        "--extensions",
        metavar=".EXT",
        nargs="+",
        type=suffix_lower,
        default=DEFAULT_EXTENSIONS,
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
        nargs="+",
        default=DEFAULT_NORMALIZATION_OPS,
        help="Normalization operations. Ie. when orientation is normalized, files with different orientations can be detected as duplicates",
    )
    parser.add_argument(
        "--resolution-normalized",
        metavar="N",
        nargs=2,
        type=int,
        default=DEFAULT_NORMALIZED_RESOLUTION,
        help="All pictures will be resized to this resolution prior to comparison. It should be smaller than the smallest picture in one duplicate group. If it's smaller, more differences in image details will be ignored.",
    )
    parser.add_argument(
        "--parallel-read",
        metavar="N",
        type=int,
        default=DEFAULT_PARALLEL_READ,
        help="Default read concurrency",
    )
    parser.add_argument(
        "--chunksize",
        metavar="N",
        type=int,
        default=2000,
        help="Specifies the number of hashes to compare at the same the time. Larger chunksizes require more memory.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default=None,
        help="Write results to file. Otherwise they are written to stdout.",
    )
    parser.add_argument(
        "--ntfy-topic",
        type=str,
        default=None,
        help="Get notifications using *ntfy* topics. Useful for long-running scans.",
    )
    parser.add_argument("--overwrite-cache", action="store_true", help="Update cached values")
    parser.add_argument("--version", action="version", version=APP_VERSION)
    parser.set_defaults(**default_config)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.hashdb:
        args.hashdb.parent.mkdir(parents=True, exist_ok=True)
        db = HashDB(args.hashdb)
        db.connection.execute("PRAGMA journal_mode=WAL;")
    else:
        db = None

    hash_func = get_hash_func(args.mode, db, args.overwrite_cache, args.normalize, args.resolution_normalized)
    metric = hash_metrics[args.mode]
    hash_result = get_hashes(
        args.directories, args.recursive, args.extensions, hash_func, metric, db, args.parallel_read
    )
    dupgroups = get_dupe_groups(metric, hash_result, args.chunksize, args.ntfy_topic)

    with StdoutFile(args.out, "wt", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(
            [
                "group",
                "path",
                "file_id",
                "device_id",
                "filesize",
                "mod_date",
                "width",
                "height",
                "exif_date_modified",
                "exif_date_taken",
                "exif_date_created",
                "maker",
                "model",
            ]
        )
        for i, paths in enumerate(dupgroups, 1):
            for path in paths:
                try:
                    write_dup(db, writer, i, path)
                except Exception:
                    logging.exception("Failed to obtain meta info for <%s>")


if __name__ == "__main__":
    main()
