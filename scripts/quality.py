import os
from argparse import ArgumentParser
from typing import Dict

import cv2  # pip install opencv-contrib-python
import kornia.filters
import numpy as np
import piq
from genutility.args import is_dir
from genutility.filesystem import scandir_ext

# from imquality import brisque  # pip install image-quality
from PIL import Image
from torchvision.transforms import functional as f

from picturetool.utils import extensions_images


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
    img = cv2.imread(path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(grey, cv2.CV_64F).var(ddof=0)
    brisque_score = cv2.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml")
    tv_score = np_total_variation(img).item()

    return {"blur_score": blur_score, "brisque_score": brisque_score[0], "tv_score": tv_score}


def torch_iqa_score(path: str) -> Dict[str, float]:
    with Image.open(path) as img:
        x = f.to_tensor(img).unsqueeze(0)
    grey = f.rgb_to_grayscale(x)

    blur_score = kornia.filters.laplacian(grey, 3).var(unbiased=False).item()
    brisque_score = piq.brisque(x, data_range=1.0).item()
    tv_score = piq.total_variation(x, reduction="none").item()

    return {"blur_score": blur_score, "brisque_score": brisque_score, "tv_score": tv_score}


def main():
    parser = ArgumentParser()
    parser.add_argument("directory", type=is_dir)
    parser.add_argument("--extensions", nargs="+", default=extensions_images)
    args = parser.parse_args()

    it = scandir_ext(args.directory, args.extensions)

    for path in it:
        d = cv2_iqa_score(os.fspath(path))
        print("cv2", path.name, d)
        d = torch_iqa_score(os.fspath(path))
        print("torch", path.name, d)
        # with Image.open(path) as img:
        #    score = brisque.score(img)
        # print(path, d['brisque_score'], score)


if __name__ == "__main__":
    main()
