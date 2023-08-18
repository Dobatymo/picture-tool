import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import cv2
import dlib
import face_recognition
import numpy as np
from genutility.concurrency import parallel_map
from genutility.filesystem import scandir_ext
from genutility.pickle import read_pickle, write_pickle

from picturetool.utils import extensions_images

PathType = Union[Path, os.DirEntry]
RawImage = np.ndarray

Encoding = np.ndarray
Location = Tuple[int, int, int, int]

Encodings = List[Encoding]
Locations = List[Location]


class PictureWindow:
    def __init__(self, name: str = "picture") -> None:
        self.name = name
        cv2.startWindowThread()  # doesn't really do anything
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        cv2.destroyAllWindows()

    def show(self, im: RawImage, size: Optional[Tuple[int, int]] = None, from_: str = "rgb") -> None:
        if size:
            im = cv2.resize(im, size)

        from_ = from_.lower()

        if from_ == "rgb":
            pass
        elif from_ in ("bgr", "pillow"):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Invalid from_ argument: {from_}")

        cv2.imshow(self.name, im)
        self.tick()

    def tick(self) -> None:
        cv2.waitKey(1)

    def showfile(self, path: str) -> None:
        im = cv2.imread(path)
        self.show(im)


def crop(img: RawImage, loc: Location, context: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> RawImage:
    top, right, bottom, left = loc  # face_recognition style
    to, ri, bo, le = context

    height, width, channels = img.shape

    top = max(0, top - to)
    right = min(width, right + ri)
    bottom = min(height, bottom + bo)
    left = max(0, left - le)

    return img[top:bottom, left:right]


class FaceStorage:
    def __init__(self, strict_bound: float, suggest_bound: float) -> None:
        if suggest_bound <= strict_bound:
            raise ValueError("strict_bound must be smaller than suggest_bound")

        self.strict_bound = strict_bound
        self.suggest_bound = suggest_bound

        self._encodings: List[Encoding] = []
        self._unknown: List[Encoding] = []
        self._skipped: List[Encoding] = []
        self._names: List[str] = []

        self._versions = self._get_versions()

    @staticmethod
    def _get_versions() -> Dict[str, str]:
        return {
            "dlib": dlib.__version__,
            "face_recognition": face_recognition.__version__,
        }

    def add(self, encoding: Encoding, name: str) -> None:
        if not name:
            raise ValueError("name cannot be empty")

        self._encodings.append(encoding)
        self._names.append(name)

    def unknown(self, encoding: Encoding) -> None:
        self._unknown.append(encoding)

    def skipped(self, encoding: Encoding) -> None:
        self._skipped.append(encoding)

    def _closest(self, encodings: Encodings, encoding: Encoding) -> Tuple[List[int], List[int]]:
        dists = face_recognition.face_distance(encodings, encoding)

        suggest = [i for i, d in enumerate(dists) if d <= self.suggest_bound and d > self.strict_bound]
        sure = [i for i, d in enumerate(dists) if d <= self.strict_bound]

        return sure, suggest

    def closest(self, encoding: Encoding) -> Tuple[List[int], List[int]]:
        return self._closest(self._encodings, encoding)

    def is_unknown(self, encoding: Encoding) -> bool:
        a, b = self._closest(self._unknown, encoding)
        return bool(a)

    def is_skipped(self, encoding: Encoding) -> bool:
        a, b = self._closest(self._skipped, encoding)
        return bool(a)

    def get_names(self, indices: Iterable[int]) -> List[str]:
        return [self._names[i] for i in indices]

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        if state["_versions"] != self._get_versions():
            logging.warning("dlib and face_recognition versions of pickled object and current environment don't match")

        self.__dict__.update(state)


def get_features(entry: PathType) -> Tuple[PathType, Locations, Encodings]:
    img = face_recognition.load_image_file(os.fspath(entry))

    locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, locations, num_jitters=1)
    # print(f"Found {len(locations)} face(s) in {os.fspath(entry)}")

    assert len(locations) == len(encodings)

    return entry, locations, encodings


def get_faces(paths: Iterable[PathType]) -> Iterator[Tuple[PathType, Location, Encoding]]:
    try:
        for entry, locations, encodings in parallel_map(get_features, paths, bufsize=100):
            for loc, enc in zip(locations, encodings):
                yield entry, loc, enc
    except KeyboardInterrupt:
        print("Face analysis interrupted")
        raise


class VectorStorage:
    def __init__(self) -> None:
        self.d: DefaultDict[str, Dict[Location, Encoding]] = defaultdict(dict)

        self._versions = self._get_versions()

    @staticmethod
    def _get_versions() -> Dict[str, str]:
        return {
            "dlib": dlib.__version__,
            "face_recognition": face_recognition.__version__,
        }

    def add(self, path: str, location: Location, encoding: Encoding) -> None:
        if not isinstance(path, str):
            raise TypeError("path must be str")

        self.d[path][location] = encoding

    def hasfile(self, path: str) -> bool:
        return path in self.d

    def encodings(self, path: str) -> Iterator[Tuple[Location, Encoding]]:
        yield from self.d[path].items()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        if state["_versions"] != self._get_versions():
            logging.warning("dlib and face_recognition versions of pickled object and current environment don't match")

        self.__dict__.update(state)


def label(paths: Iterable[PathType], fdb: FaceStorage, vdb: VectorStorage) -> None:
    with PictureWindow() as win:

        def uncached_paths(paths: PathType) -> PathType:
            for p in paths:
                # this also skips partially labelled files
                if not vdb.hasfile(os.fspath(p)):
                    yield p

        for entry, loc, enc in get_faces(uncached_paths(paths)):
            vdb.add(os.fspath(entry), loc, enc)

            if fdb.is_skipped(enc):
                print(f"Skipping {entry.name}")
                continue
            if fdb.is_unknown(enc):
                print(f"Skipping unknown {entry.name}")
                continue

            sure, suggest = fdb.closest(enc)

            foundnames = set(fdb.get_names(sure))
            if len(foundnames) == 1:
                print(f"{entry.name} - Found: {foundnames.pop()}")
                continue
            elif len(foundnames) > 1:
                pass
            else:
                foundnames = set(fdb.get_names(suggest))

            if foundnames:
                print(f"{entry.name} - Matches: {', '.join(foundnames)}")

            img = face_recognition.load_image_file(os.fspath(entry))
            img = crop(img, loc, (20, 20, 20, 20))

            win.show(img, from_="BGR")
            name = input(f"{entry.name} - Name: ")

            if name:
                fdb.add(enc, name)
            else:
                fdb.skipped(enc)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from genutility.args import is_dir

    DEFAULT_STRICT_BOUND = 0.1
    DEFAULT_SUGGEST_BOUND = 0.5
    DEFAULT_FACES_DB = "faces.p"
    DEFAULT_VECTOR_DB = "encodings.p"

    parser = ArgumentParser()
    parser.add_argument("path", type=is_dir, help="Input path to scan for images")
    parser.add_argument("--strict", type=float, default=DEFAULT_STRICT_BOUND)
    parser.add_argument("--extensions", nargs="+", default=extensions_images, help="List of file extensions to scan")
    parser.add_argument("--suggest", type=float, default=DEFAULT_SUGGEST_BOUND)
    parser.add_argument("--faces-db", type=Path, default=DEFAULT_FACES_DB)
    parser.add_argument("--vector-db", type=Path, default=DEFAULT_VECTOR_DB)
    parser.add_argument("-r", "--recursive", action="store_true", help="Scan directory recursively")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    it = scandir_ext(args.path, args.extensions, rec=args.recursive)

    try:
        fdb = read_pickle(args.faces_db)
    except FileNotFoundError:
        fdb = FaceStorage(args.strict, args.suggest)

    try:
        vdb = read_pickle(args.vector_db)
    except FileNotFoundError:
        vdb = VectorStorage()

    try:
        label(it, fdb, vdb)
    except KeyboardInterrupt:
        print("Interrupted labeling")

    write_pickle(fdb, args.faces_db, safe=True)
    write_pickle(vdb, args.vector_db, safe=True)
