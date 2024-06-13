import csv
import logging
import os
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path
from typing import Dict

import cv2
import kornia.filters
import numpy as np
import piq
import pyiqa
import skvideo.measure
from concurrex.thread import ThreadedIterator
from genutility.args import is_dir
from genutility.filesystem import scandir_ext
from genutility.rich import Progress
from imquality import brisque
from PIL import Image
from rich.logging import RichHandler
from rich.progress import Progress as RichProgress
from torchvision.transforms import functional as f
from typing_extensions import Self

from picturetool.utils import extensions_images

LOG_STR = "Failed to run %r on %r"


class AutoThreadedIterator(ThreadedIterator):
    def __len__(self):
        return self.processed()


class LogException:
    def __init__(self, s: str, *args) -> None:
        self.s = s
        self.args = args

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            if isinstance(exc_value, Exception):
                logging.exception(self.s, *self.args)  # , exc_info=
                return True


def np_total_variation(x: np.ndarray, norm_type: str = "l2") -> np.ndarray:
    """x: [..., h, w, c]"""

    if x.dtype == np.uint8:
        x = x.astype(np.uint)

    if norm_type == "l1":
        w_variance = np.sum(np.abs(x[..., :, 1:, :] - x[..., :, :-1, :]), axis=(-1, -2, -3))
        h_variance = np.sum(np.abs(x[..., 1:, :, :] - x[..., :-1, :, :]), axis=(-1, -2, -3))
        score = h_variance + w_variance
    elif norm_type == "l2":
        d_w = x[..., :-1, 1:, :] - x[..., :-1, :-1, :]
        d_h = x[..., 1:, :-1, :] - x[..., :-1, :-1, :]
        score = np.sum(np.sqrt(np.square(d_w) + np.square(d_h)), axis=(-1, -2, -3))
    elif norm_type == "l2_squared":
        d_w = x[..., :-1, 1:, :] - x[..., :-1, :-1, :]
        d_h = x[..., 1:, :-1, :] - x[..., :-1, :-1, :]
        score = np.sum(np.square(d_w) + np.square(d_h), axis=(-1, -2, -3))
    else:
        raise ValueError("Incorrect norm type, should be one of {'l1', 'l2', 'l2_squared'}")

    return score


def cv2_iqa_score(path: str) -> Dict[str, float]:
    ret = {}

    img = cv2.imread(path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with LogException(LOG_STR, "cv2.Laplacian", path):
        ret["blur-cv"] = cv2.Laplacian(grey, cv2.CV_64F).var(ddof=0)
    with LogException(LOG_STR, "cv2.quality.QualityBRISQUE_compute", path):
        ret["brisque-cv"] = cv2.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml")[
            0
        ]
    with LogException(LOG_STR, "np_total_variation", path):
        ret["tv-np"] = np_total_variation(img).item()

    return ret


niqe_metric = pyiqa.create_metric("niqe")
brisque_metric = pyiqa.create_metric("brisque")
nima_metric = pyiqa.create_metric("nima")


def torch_iqa_score(path: str) -> Dict[str, float]:
    ret = {}

    with Image.open(path) as img:
        x = f.to_tensor(img).unsqueeze(0)
        grey = f.rgb_to_grayscale(x)

        with LogException(LOG_STR, "kornia.filters.laplacian", path):
            ret["blur-kornia"] = kornia.filters.laplacian(grey, 3).var(unbiased=False).item()
        with LogException(LOG_STR, "piq.brisque", path):
            ret["brisque-piq"] = piq.brisque(x, data_range=1.0).item()
        with LogException(LOG_STR, "piq.total_variation", path):
            ret["tv-piq"] = piq.total_variation(x, reduction="none").item()
        with LogException(LOG_STR, "pyiqa.create_metric('niqe')", path):
            ret["niqe-pyiqa"] = niqe_metric(x).item()
        with LogException(LOG_STR, "pyiqa.create_metric('brisque')", path):
            ret["brisque-pyiqa"] = brisque_metric(x).item()
        with LogException(LOG_STR, "pyiqa.create_metric('nima')", path):
            ret["nima-pyiqa"] = nima_metric(x).item()

    return ret


def other_iqa_scires(path: str) -> Dict[str, float]:
    ret = {}
    with Image.open(path) as img:
        with LogException(LOG_STR, "brisque.score", path):
            ret["brisque-imquality"] = brisque.score(img)
        with LogException(LOG_STR, "skvideo.measure.niqe", path):
            video = np.array(img.convert("YCbCr"))[None, :, :, 1]
            ret["niqe-skvideo"] = skvideo.measure.niqe(video)

    return ret


def main():
    parser = ArgumentParser()
    parser.add_argument("path", type=is_dir, help="Input directory")
    parser.add_argument("--extensions", nargs="+", default=extensions_images)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directory recursively.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    handler = RichHandler(log_time_format="%Y-%m-%d %H-%M-%S%Z")
    FORMAT = "%(message)s"

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=[handler])

    with AutoThreadedIterator(scandir_ext(args.path, args.extensions, rec=args.recursive), maxsize=0) as it:
        with open(args.out, "w", encoding="utf-8", newline="") as csvfile, RichProgress() as progress:
            p = Progress(progress)
            fieldnames = [
                "path",
                "filesize",
                "mod_time",
                "blur-cv",
                "brisque-cv",
                "tv-np",
                "blur-kornia",
                "brisque-piq",
                "tv-piq",
                "niqe-pyiqa",
                "brisque-pyiqa",
                "nima-pyiqa",
                "brisque-imquality",
                "niqe-skvideo",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entity in islice(p.track_auto(it), args.limit):
                path = os.fspath(entity)
                filesize = entity.stat().st_size
                mod_time = entity.stat().st_mtime_ns
                with LogException("Failed to process '%s'", path):
                    row = {"path": path, "filesize": filesize, "mod_time": mod_time}
                    for func in [cv2_iqa_score, torch_iqa_score, other_iqa_scires]:
                        d = func(path)
                        row.update(d)
                    writer.writerow(row)


if __name__ == "__main__":
    main()
