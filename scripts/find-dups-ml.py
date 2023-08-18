import csv
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import torch
import transformers
from genutility.args import is_dir
from genutility.file import StdoutFile
from genutility.filesystem import scandir_ext
from genutility.iter import batch
from genutility.rich import Progress
from PIL import Image
from rich.progress import Progress as RichProgress
from torchvision.transforms import functional as F
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor
from transformers.models.vit.modeling_vit import ViTModel

from picturetool.ml_utils import faiss_duplicates_threshold_pairs
from picturetool.utils import CollectingIterable, ThreadedIterator, extensions_images

DEFAULT_VIT_MODEL = "nateraw/vit-base-beans"


def load_images(paths: Iterable[Path]) -> Iterator[torch.Tensor]:
    for path in paths:
        try:
            with Image.open(path, "r") as img:
                if img.mode in ("L",):
                    logging.debug("Unsupported image mode for %s: %s", path, img.mode)
                    continue
                if img.mode not in ("RGB",):
                    logging.info("Unknown image mode for %s: %s", path, img.mode)

                yield F.pil_to_tensor(img)
        except Exception as e:
            logging.warning("Skipping %s: %s", path, e)


def extract_embeddings(
    model: torch.nn.Module, extractor: ImageFeatureExtractionMixin, images: List[torch.Tensor]
) -> np.ndarray:
    assert isinstance(images, list)
    assert isinstance(images[0], torch.Tensor)

    with torch.inference_mode():
        image_pp: transformers.BatchFeature = extractor(images, return_tensors="pt")
        assert image_pp["pixel_values"].shape[1:] == (3, 224, 224)

        features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
        assert features.shape[1:] == (768,)

    return features


def generate_embeddings(model, extractor, pictures, batchsize, progress: Progress) -> Iterator[np.ndarray]:
    for images in batch(progress.track(pictures), batchsize, list):
        yield extract_embeddings(model, extractor, images)


def find_dups_ml(
    path: Path, vit_model: str, batchsize: int = 100, verbose: bool = False
) -> Tuple[List[Path], np.ndarray]:
    paths = CollectingIterable(map(Path, scandir_ext(path, extensions_images)))

    threshold = 1.0
    num_threads = (os.cpu_count() or 2) - 1
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    logging.info("Using %i intra-op and %i inter-op threads", torch.get_num_threads(), torch.get_num_interop_threads())

    extractor: ViTFeatureExtractor = transformers.AutoFeatureExtractor.from_pretrained(vit_model)
    model: ViTModel = transformers.AutoModel.from_pretrained(vit_model)
    model.eval()

    pictures = ThreadedIterator(load_images(paths), batchsize)
    with RichProgress(disable=not verbose) as progress:
        p = Progress(progress)
        it = generate_embeddings(model, extractor, pictures, batchsize, p)
        pairs = faiss_duplicates_threshold_pairs("l2-squared", it, threshold, 1000, p)
    assert paths.exhausted

    return paths.collection, pairs


def main():
    parser = ArgumentParser()
    parser.add_argument("path", type=is_dir)
    parser.add_argument("--vision-transformer-model", default=DEFAULT_VIT_MODEL)
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default=None,
        help="Write results to file. Otherwise they are written to stdout.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("PIL").setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.INFO)
        transformers.logging.enable_progress_bar()
    else:
        logging.basicConfig(level=logging.INFO)
        transformers.logging.set_verbosity_warning()

    paths, pairs = find_dups_ml(args.path, args.vision_transformer_model, args.batchsize, args.verbose)

    with StdoutFile(args.out, "wt", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["path-a", "path-b"])
        for a, b in pairs:
            writer.writerow([paths[a], paths[b]])


if __name__ == "__main__":
    main()
