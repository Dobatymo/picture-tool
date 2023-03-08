# %pip install numpy torch transformers faiss-cpu pillow tqdm
import csv
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import faiss
import numpy as np
import torch
import transformers
from genutility.args import is_dir
from genutility.file import StdoutFile
from genutility.iter import batch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor
from transformers.models.vit.modeling_vit import ViTModel

from utils import CollectingIterable, ThreadedIterator, slice_idx

DEFAULT_VIT_MODEL = "nateraw/vit-base-beans"


def load_images(paths: Iterable[Path]) -> Iterator[torch.Tensor]:
    for path in paths:
        try:
            with Image.open(path) as img:
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

    image_pp: transformers.BatchFeature = extractor(images, return_tensors="pt")
    assert image_pp["pixel_values"].shape[1:] == (3, 224, 224)

    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    assert features.shape[1:] == (768,)

    return features


def get_similar_top_k(
    index: faiss.IndexFlat, batchsize: int = 1000, top_k: int = 5, verbose: bool = False
) -> Iterator[Tuple[int, int, float]]:
    with tqdm(total=index.ntotal, disable=not verbose) as pbar:
        for i0, ni in slice_idx(index.ntotal, batchsize):
            query = index.reconstruct_n(i0, ni)
            distances, indices = index.search(query, top_k)

            rindices = range(i0, i0 + ni)
            for q_indices, q_idx, q_distances in zip(indices, rindices, distances):
                for idx, dist in zip(q_indices, q_distances):
                    if idx == -1:
                        break
                    if idx != q_idx:
                        yield idx, q_idx, dist

            pbar.update(ni)


def get_similar_treshold(
    index, batchsize: int = 1000, threshold: float = 1.0, verbose: bool = False
) -> Iterator[Tuple[int, int, float]]:
    with tqdm(total=index.ntotal, disable=not verbose) as pbar:
        for i0, ni in slice_idx(index.ntotal, batchsize):
            query = index.reconstruct_n(i0, ni)

            lims, distances, indices = index.range_search(query, threshold)
            for i in range(ni):
                q_idx = i0 + i
                q_indices = indices[lims[i] : lims[i + 1]]
                q_distances = distances[lims[i] : lims[i + 1]]
                for idx, dist in zip(q_indices, q_distances):
                    if idx != q_idx:
                        yield idx, q_idx, dist

            pbar.update(ni)


def find_dups_ml(
    path: Path, vit_model: str, batchsize: int = 100, verbose: bool = False
) -> Tuple[List[Path], np.ndarray]:
    paths = CollectingIterable(path.rglob("*.jpg"))

    extractor: ViTFeatureExtractor = transformers.AutoFeatureExtractor.from_pretrained(vit_model)
    model: ViTModel = transformers.AutoModel.from_pretrained(vit_model)

    hidden_dim = model.config.hidden_size

    pictures = ThreadedIterator(load_images(paths), batchsize)

    index = faiss.IndexFlatL2(hidden_dim)
    for images in batch(tqdm(pictures, disable=not verbose), batchsize, list):
        embeddings = extract_embeddings(model, extractor, images)
        index.add(embeddings)

    assert paths.exhausted

    pairs_threshold = np.array([(a, b) for a, b, dist in get_similar_treshold(index)])

    # print('top-k')
    # pairs_top_k = np.array([(a, b) for a, b, dist in get_similar_top_k(index)])
    # print(pairs_top_k)

    return paths.collection, pairs_threshold


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
        writer.writerow(["a", "b"])
        for a, b in pairs:
            writer.writerow([paths[a], paths[b]])


if __name__ == "__main__":
    main()
