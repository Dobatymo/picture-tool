import hashlib
import logging
import multiprocessing
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Callable, Container, DefaultDict, Iterable, Iterator, List, Set, Tuple

import imagehash
import numpy as np
from genutility.args import is_dir, suffix_lower
from genutility.filesdb import FileDbSimple, NoResult
from genutility.filesystem import entrysuffix, scandir_rec
from genutility.hash import hash_file
from genutility.image import normalize_image_rotation
from genutility.iter import progress
from genutility.numpy import hamming_dist_packed
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


class HashDB(FileDbSimple):
    @classmethod
    def derived(cls):
        return [
            ("file_sha256", "BLOB", "?"),
            ("image_sha256", "BLOB", "?"),
            ("phash", "BLOB", "?"),
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


class wrap_with_db:
    def __init__(self, func: Callable[[str], bytes], db: HashDB, colname: str) -> None:
        self.func = func
        self.db = db
        self.colname = colname

    def __call__(self, path: Path) -> Tuple[bool, Path, bytes]:
        try:
            (img_hash,) = self.db.get(path, only={self.colname})
            if img_hash is None:
                raise NoResult
            cached = True
        except NoResult:
            img_hash = self.func(os.fspath(path))
            cached = False

        return cached, path, img_hash


class wrap_without_db:
    def __init__(self, func: Callable[[str], bytes]) -> None:
        self.func = func

    def __call__(self, path: Path) -> Tuple[bool, Path, bytes]:
        cached = False
        img_hash = self.func(os.fspath(path))
        return cached, path, img_hash


hash_cols = {"file-hash": "file_sha256", "image-hash": "image_sha256", "phash": "phash"}
hash_metrics = {"file-hash": "equivalence", "image-hash": "equivalence", "phash": "hamming"}


def scandir_error_log_warning(entry: os.DirEntry, exception) -> None:
    logging.warning("<%s> %s: %s", entry.path, type(exception).__name__, exception)


def main():

    HEIF_EXTENSIONS = (".heic", ".heif")
    DEFAULT_EXTENSIONS = (".jpg", ".jpeg") + HEIF_EXTENSIONS
    DEFAULT_HASHDB = "hashes.sqlite"
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
        "--hashdb", metavar="PATH", default=DEFAULT_HASHDB, help="Path to sqlite database file to store hashes."
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
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if set(HEIF_EXTENSIONS) & set(args.extensions):
        from pillow_heif import register_heif_opener

        register_heif_opener()

    def pathiter(directories: Iterable[Path], recursive: bool) -> Iterator[os.DirEntry]:
        for directory in directories:
            yield from scandir_rec(
                directory, files=True, dirs=False, rec=recursive, errorfunc=scandir_error_log_warning
            )

    if args.hashdb:
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
    if db:
        hash_func = wrap_with_db(hash_func, db, colname)
    else:
        hash_func = wrap_without_db(hash_func)

    with ProcessPoolExecutor(args.parallel_read) as executor:
        futures: List[Future] = []
        cached: bool
        path: Path
        img_hash: bytes

        for entry in progress(pathiter(args.directories, args.recursive)):
            if entrysuffix(entry) not in args.extensions:
                continue

            futures.append(executor.submit(hash_func, Path(entry)))

        logging.info("Analyzing %d files", len(futures))

        num_cached = 0
        num_fresh = 0
        num_error = 0
        for future in progress(as_completed(futures), length=len(futures)):
            try:
                cached, path, img_hash = future.result()
            except (OSError, UnidentifiedImageError) as e:
                logging.warning("%s: %s", type(e).__name__, e)
                num_error += 1
                continue
            except Exception:
                logging.exception("Failed to hash image file")
                num_error += 1
                continue

            rawpath = os.fspath(path)

            if db and not cached:
                db.add(path, derived={colname: img_hash}, commit=False)

            if metric == "equivalence":
                hashes[img_hash].add(rawpath)
            elif metric == "hamming":
                paths.append(rawpath)
                hashes.append(img_hash)

            if cached:
                num_cached += 1
            else:
                num_fresh += 1

        logging.info(
            "Loaded %d hashes from cache, computed %d fresh ones and failed to read %d",
            num_cached,
            num_fresh,
            num_error,
        )

    db.commit()

    if metric == "equivalence":
        hashes = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}

        for i, (k, paths) in enumerate(hashes.items(), 1):
            for path in paths:
                print(f"{i:09} {path}")
    elif metric == "hamming":
        arr = np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(len(hashes), -1)
        hamming_threshold = 1

        dists = hamming_dist_packed(arr[None, :], arr[:, None])
        dups = np.argwhere(np.triu(dists <= hamming_threshold, 1))

        for i, (key, group) in enumerate(groupby(dups, key=itemgetter(0))):
            print(f"{i:09} {paths[key]}")
            for _, second in group:
                print(f"{i:09} {paths[second]}")


if __name__ == "__main__":
    main()
