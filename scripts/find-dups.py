import csv
import hashlib
import logging
import logging.handlers
import multiprocessing
import os
import queue
import sys
import threading
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Collection,
    Container,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
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
import requests
from genutility.args import is_dir, suffix_lower
from genutility.datetime import datetime_from_utc_timestamp_ns
from genutility.file import StdoutFile
from genutility.filesdb import NoResult
from genutility.filesystem import entrysuffix, scandir_rec
from genutility.hash import hash_file
from genutility.image import normalize_image_rotation
from genutility.json import read_json
from genutility.rich import Progress, get_double_format_columns
from genutility.time import DeltaTime, MeasureTime
from genutility.typing import CsvWriter
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from PIL.IptcImagePlugin import getiptcinfo
from rich.progress import Progress as RichProgress
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from picturetool.utils import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_APPDATA_DIR,
    DEFAULT_HASHDB,
    HashDB,
    MultiprocessingProcess,
    QueueListenerContext,
    ThreadRichHandler,
    extensions_heif,
    extensions_images,
    get_exif_dates,
    make_groups,
    npmp_duplicates_threshold_pairs,
    rich_sys_excepthook,
    rich_sys_unraisablehook,
    rich_threading_excepthook,
)

COMMIT_PERIOD_SECONDS = 120.0


class ImageError(Exception):
    pass


class MetaProvider:
    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        raise NotImplementedError

    def get_cols(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def get_non_optional_cols(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def get_meta(self, path: Path) -> Dict[str, Any]:
        raise NotImplementedError


class HashProvider:
    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        raise NotImplementedError

    def get_metric(self) -> str:
        raise NotImplementedError

    def get_col(self) -> str:
        raise NotImplementedError

    def get_hash(self, path: Path) -> bytes:
        raise NotImplementedError


class NoMetaProvider(MetaProvider):
    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        pass

    def get_cols(self) -> Tuple[str, ...]:
        return ()

    def get_non_optional_cols(self) -> Tuple[str, ...]:
        return ()

    def get_meta(self, path: Path) -> Dict[str, Any]:
        return {}


class ImageMetaProvider(MetaProvider):
    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        if extensions is None or set(extensions_heif) & set(extensions):
            from pillow_heif import register_heif_opener

            register_heif_opener()

    def get_cols(self) -> Tuple[str, ...]:
        return ("width", "height", "exif", "icc_profile")

    def get_non_optional_cols(self) -> Tuple[str, ...]:
        return ("width", "height")

    def get_meta(self, path: Path) -> Dict[str, Any]:
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

        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(exif, (bytes, type(None)))
        assert isinstance(icc_profile, (bytes, type(None)))
        assert isinstance(iptc, (bytes, type(None)))
        assert isinstance(photoshop, (bytes, type(None)))

        return {
            "width": width,
            "height": height,
            "exif": exif,
            "icc_profile": icc_profile,
            "iptc": iptc,
            "photoshop": photoshop,
        }


class FileHashProvider(HashProvider):
    def __init__(self, **kwargs) -> None:
        pass

    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        pass

    def get_metric(self) -> str:
        return "equivalence"

    def get_col(self) -> str:
        return "file_sha256"

    def get_hash(self, path: Path) -> bytes:
        return hash_file(path, hashlib.sha256).digest()


class ImageHashProvider(HashProvider):
    def __init__(self, *, normalize: Container[str], resolution: Tuple[int, int], **kwargs) -> None:
        self.normalize = normalize
        self.resolution = resolution

    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        if extensions is None or set(extensions_heif) & set(extensions):
            from pillow_heif import register_heif_opener

            register_heif_opener()

    def get_metric(self) -> str:
        return "equivalence"

    def get_col(self) -> str:
        return "image_sha256"

    def get_hash(self, path: Path) -> bytes:
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

        return m.digest()


class PhashProvider(HashProvider):
    def __init__(self, **kwargs) -> None:
        pass

    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        pass

    def get_metric(self) -> str:
        return "hamming"

    def get_col(self) -> str:
        return "phash"

    def get_hash(self, path: Path) -> bytes:
        with Image.open(path, "r") as img:
            phash = bytes.fromhex(str(imagehash.phash(img, hash_size=16)))  # 32bytes

        return phash


class wrap:
    def __init__(self, hash: HashProvider, meta: MetaProvider) -> None:
        self.hash = hash
        self.meta = meta

    def initializer(self, extensions: Optional[Collection[str]]) -> None:
        self.hash.initializer(extensions)
        self.meta.initializer(extensions)

    def _get_cols(self) -> Tuple[str, ...]:
        return (self.hash.get_col(),) + self.meta.get_cols()

    def _get_non_optional_cols(self) -> Tuple[str, ...]:
        return (self.hash.get_col(),) + self.meta.get_non_optional_cols()

    def _get_file_meta(self, path: Path) -> Dict[str, Any]:
        try:
            hash_bytes = self.hash.get_hash(path)
        except UnidentifiedImageError:
            raise  # inherits from OSError, so must be re-raised explicitly
        except OSError as e:
            raise ImageError(path, e)
        except ValueError as e:
            raise ImageError(path, e)

        out = {
            self.hash.get_col(): hash_bytes,
        }
        out.update(self.meta.get_meta(path))
        return out


class wrap_without_db(wrap):
    def __call__(self, path: Path) -> Tuple[bool, Path, Dict[str, Any]]:
        cached = False
        meta = self._get_file_meta(path)
        return cached, path, meta


class wrap_with_db(wrap):
    def __init__(self, hash: HashProvider, meta: MetaProvider, db: HashDB, overwrite: bool = False) -> None:
        wrap.__init__(self, hash, meta)
        self.only = self._get_cols()
        self.db = db
        self.overwrite = overwrite

    def __call__(self, path: Path) -> Tuple[bool, Path, Dict[str, Any]]:
        try:
            if self.overwrite:
                raise NoResult
            try:
                values = self.db.get(path, only=self.only)
            except OverflowError as e:
                logging.error("Failed to query hash cache database for <%s>. OverflowError: %s", path, e)
                raise NoResult
            meta = dict(zip(self.only, values))
            if any(meta[col] is None for col in self._get_non_optional_cols()):
                raise NoResult
            cached = True
        except NoResult:
            meta = self._get_file_meta(path)
            cached = False

        return cached, path, meta


hash_providers = {"file-hash": FileHashProvider, "image-hash": ImageHashProvider, "phash": PhashProvider}
meta_providers = {"file-hash": NoMetaProvider, "image-hash": ImageMetaProvider, "phash": ImageMetaProvider}


def scandir_error_log_warning(entry: os.DirEntry, exception) -> None:
    logging.warning("<%s> %s: %s", entry.path, type(exception).__name__, exception)


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


def notify(topic: str, message: str) -> None:
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
        try:
            dt = datetime_from_utc_timestamp_ns(mod_date, aslocal=True)
        except OSError:
            # can happen for bad modification dates
            dt = None

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
                dt.isoformat() if dt else None,
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


def get_hash_func(
    mode: str, db: Optional[HashDB], overwrite_cache: bool, hash_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any]
) -> wrap:
    hashprovider = hash_providers[mode](**hash_kwargs)
    metaprovider = meta_providers[mode](**meta_kwargs)

    if db is not None:
        hash_func: wrap = wrap_with_db(hashprovider, metaprovider, db, overwrite_cache)
    else:
        hash_func = wrap_without_db(hashprovider, metaprovider)

    return hash_func


class HashResult(NamedTuple):
    paths: List[str]
    hashes: List[bytes]


def log_basic_config(level, handler) -> None:
    logging.basicConfig(level=level, format="%(message)s", handlers=[handler])
    logging.captureWarnings(True)
    if level == logging.DEBUG:
        logging.getLogger("PIL").setLevel(logging.INFO)
        logging.getLogger("genutility").setLevel(logging.INFO)

    threading.excepthook = rich_threading_excepthook
    sys.excepthook = rich_sys_excepthook
    sys.unraisablehook = rich_sys_unraisablehook
    # there are still unstyled exceptions logged by `multiprocessing.process.BaseProgress._bootstrap`


def initializer(
    queue: Optional[queue.Queue], hash_func: wrap, extensions: Optional[Collection[str]], level: int = logging.NOTSET
) -> None:
    name = multiprocessing.current_process().name
    if queue is not None:
        handler = logging.handlers.QueueHandler(queue)
        log_basic_config(level, handler)
    hash_func.initializer(extensions)
    logging.debug("Initialized worker %s", name)


def get_hashes(
    directories: Iterable[Path],
    recursive: bool,
    hash_func: wrap,
    extensions: Optional[Collection[str]] = None,
    db: Optional[HashDB] = None,
    parallel_read: Optional[int] = None,
    queue: Optional[queue.Queue] = None,
) -> HashResult:
    paths: List[str] = []
    hashes: List[bytes] = []

    with MeasureTime() as stopwatch, RichProgress(*get_double_format_columns()) as progress:
        futures: List[Future] = []
        cached: bool
        path: Path
        img_hash: bytes
        inodes: Dict[Tuple[int, int], Set[Path]] = {}
        p = Progress(progress)

        mp_context = multiprocessing.get_context()
        mp_context.Process = MultiprocessingProcess
        executor = ProcessPoolExecutor(
            parallel_read,
            mp_context,
            initializer=initializer,
            initargs=(queue, hash_func, extensions, logging.root.level),
        )

        logging.debug("Start reading files using %d workers", parallel_read)

        num_cached = 0
        num_fresh = 0
        num_error = 0

        try:
            for entry in p.track(pathiter(directories, recursive), description="Found {task.completed} files"):
                if extensions is not None and entrysuffix(entry).lower() not in extensions:
                    continue

                path = Path(entry)
                stats = path.stat()  # `entry.stat()` does not populate all fields on windows

                assert stats.st_nlink > 0, path

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

            delta = DeltaTime()
            for future in p.track(
                as_completed(futures), total=len(futures), description="Computed {task.completed}/{task.total} hashes"
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

                if db is not None and not cached:
                    if delta.get() > COMMIT_PERIOD_SECONDS:
                        commit = True
                        logging.debug("Committing hashes to database...")
                    else:
                        commit = False
                    db.add(path, derived=meta, commit=commit, replace=False)
                    if commit:
                        delta.reset()

                img_hash = meta[hash_func.hash.get_col()]
                rawpath = os.fspath(path)

                paths.append(rawpath)
                hashes.append(img_hash)

                if cached:
                    num_cached += 1
                else:
                    num_fresh += 1
        except KeyboardInterrupt:
            logging.warning("Interrupted. Cancelling outstanding tasks and waiting for current tasks to finish.")
            for future in futures:
                future.cancel()
            sys.exit(1)
        except Exception:
            logging.error("Error. Cancelling outstanding tasks and waiting for current tasks to finish.")
            for future in futures:
                future.cancel()
            raise
        finally:
            time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
            logging.info(
                "Loaded %d hashes from cache, computed %d fresh ones and failed to read %d in %s",
                num_cached,
                num_fresh,
                num_error,
                time_delta,
            )
            try:
                if db is not None:
                    db.commit()
            finally:
                logging.debug("Shutting down worker processes...")
                executor.shutdown()
                logging.debug("Shutting down worker processes done")

    return HashResult(paths, hashes)


def get_dupe_groups(
    metric: str, hash_result: HashResult, metric_kwargs: Dict[str, Any], chunksize, ntfy_topic: Optional[str] = None
) -> List[List[str]]:
    dupgroups: List[List[str]]

    if len(hash_result.paths) != len(hash_result.hashes):
        raise ValueError(
            f"paths ({len(hash_result.paths)}) and hashes ({len(hash_result.hashes)}) of hash_result are of different lengths"
        )

    with MeasureTime() as stopwatch, RichProgress() as progress:
        p = Progress(progress)

        if hash_result.paths:
            if metric == "equivalence":
                hash2paths: DefaultDict[bytes, Set[str]] = defaultdict(set)
                for rawpath, img_hash in zip(hash_result.paths, hash_result.hashes):
                    hash2paths[img_hash].add(rawpath)
                dupgroups = [list(paths) for digest, paths in hash2paths.items() if len(paths) > 1]
            elif metric == "hamming":
                hamming_threshold = metric_kwargs["hamming_threshold"]
                dups = npmp_duplicates_threshold_pairs("hamming", hash_result.hashes, hamming_threshold, chunksize, p)
                dupgroups = [[hash_result.paths[idx] for idx in indices] for indices in make_groups(dups)]
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        else:
            dupgroups = []

    time_delta = humanize.precisedelta(timedelta(seconds=stopwatch.get()))
    logging.info("Found %s duplicate groups in %s", len(dupgroups), time_delta)
    if ntfy_topic:
        notify(ntfy_topic, f"Found {len(dupgroups)} duplicate groups in {time_delta}")

    return dupgroups


extensions_mode = {
    "file-hash": None,
    "image-hash": extensions_images,
    "phash": extensions_images,
}


def main() -> None:
    try:
        config_path = DEFAULT_APPDATA_DIR / "config.json"
        default_config = {k.replace("-", "_"): v for k, v in read_json(config_path).items()}
    except FileNotFoundError:
        default_config = {}

    DEFAULT_NORMALIZATION_OPS = ("orientation", "resolution", "colors")
    DEFAULT_NORMALIZED_RESOLUTION = (256, 256)
    DEFAULT_PARALLEL_READ = max(multiprocessing.cpu_count() - 1, 1)
    DESCRIPTION = "Find picture duplicates"
    DEFAULT_HAMMING_THRESHOLD = 1

    parser = ArgumentParser(formatter_class=ArgumentDefaultsRichHelpFormatter, description=DESCRIPTION)
    parser.add_argument("directories", metavar="DIR", nargs="+", type=is_dir, help="Input directories")
    parser.add_argument(
        "--extensions",
        metavar=".EXT",
        nargs="+",
        type=suffix_lower,
        default=None,
        help=f"File extensions to process. The default depends on `--mode`. For `file-hash` it's all extensions, for `image-hash` and `phash` it's {extensions_mode['image-hash']}",
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
        "--hamming-threshold",
        metavar="N",
        type=int,
        default=DEFAULT_HAMMING_THRESHOLD,
        help="Maximum distance of semantic hashes which use the hamming metric",
    )
    parser.add_argument(
        "--parallel-read", metavar="N", type=int, default=DEFAULT_PARALLEL_READ, help="Default read concurrency"
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

    handler = ThreadRichHandler(verbose=args.verbose, log_time_format="%Y-%m-%d %H-%M-%S%Z")
    level = logging.DEBUG if args.verbose else logging.INFO
    log_basic_config(level, handler)  # don't use logging above this line

    queue = multiprocessing.Manager().Queue(-1)
    queuelistener = QueueListenerContext(queue, handler)

    if args.hashdb:
        args.hashdb.parent.mkdir(parents=True, exist_ok=True)
        db = HashDB(args.hashdb)
        db.connection.execute("PRAGMA journal_mode=WAL;")
    else:
        db = None

    extensions = extensions_mode[args.mode]
    hash_kwargs: Dict[str, Any] = {"normalize": args.normalize, "resolution": args.resolution_normalized}
    meta_kwargs: Dict[str, Any] = {}
    hash_func = get_hash_func(args.mode, db, args.overwrite_cache, hash_kwargs, meta_kwargs)
    metric = hash_func.hash.get_metric()

    with queuelistener:
        hash_result = get_hashes(args.directories, args.recursive, hash_func, extensions, db, args.parallel_read, queue)
    metric_kwargs: Dict[str, Any] = {"hamming_threshold": args.hamming_threshold}
    dupgroups = get_dupe_groups(metric, hash_result, metric_kwargs, args.chunksize, args.ntfy_topic)

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
                    logging.exception("Failed to obtain meta info for <%s>", path)


if __name__ == "__main__":
    import colorama

    colorama.init()

    main()
