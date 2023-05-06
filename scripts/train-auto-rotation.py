import logging
import os
from pathlib import Path
from random import randrange
from typing import Any, Callable, List, Optional, Tuple, Dict

import lightning.pytorch as pl
import piexif
import torch
from genutility.cache import cache
from genutility.exceptions import NoActionNeeded
from genutility.filesystem import scandir_ext
from genutility.pillow import _fix_orientation
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification import multiclass_accuracy
from torchvision import models
from torchvision import transforms as T
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


def preview(dataset: Dataset) -> None:
    for i, (img, label) in enumerate(dataset):
        orientation = [1, 6, 3, 8][label]
        # print(i, label, orientation)
        try:
            img = _fix_orientation(img, orientation)
        except NoActionNeeded:
            pass

        img.show()


class NoOrientationInfoError(Exception):
    pass


def get_orientation(img: Image) -> int:
    exif = img.info.get("exif", None)

    if exif is None:
        raise NoOrientationInfoError("No exif info")

    try:
        d = piexif.load(exif)
    except piexif._exceptions.InvalidImageDataError:
        raise NoOrientationInfoError("Bad exif info")

    try:
        return d["0th"][piexif.ImageIFD.Orientation]
    except KeyError:
        raise NoOrientationInfoError("No orientation info in exif")


class AugmentedRotationDataset(Dataset):
    rot_map = {
        1: 0,
        3: 2,
        6: 1,
        8: 3,
    }
    rot_actions = {
        1: Image.Transpose.ROTATE_90,
        2: Image.Transpose.ROTATE_180,
        3: Image.Transpose.ROTATE_270,
    }

    def __init__(self, paths: List[Path], labels: List[int], transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def get_num_classes(self) -> int:
        return 4

    def with_transform(self, transform: Optional[Callable] = None) -> "AugmentedRotationDataset":
        return AugmentedRotationDataset(self.paths, self.labels, transform)

    @classmethod
    def get_label(cls, path: str) -> int:
        with open(path, "rb") as fr:
            img = Image.open(fr)

        try:
            orientation = get_orientation(img)
        except NoOrientationInfoError:
            orientation = 1

        try:
            label = cls.rot_map[orientation]
        except KeyError:
            raise ValueError(f"Unsupported orientation {orientation} in {path}")

        return label

    @classmethod
    def make(cls, basepath: Path, transform: Optional[Callable] = None) -> "AugmentedRotationDataset":
        paths: List[Path] = []
        labels: List[int] = []

        for path in scandir_ext(basepath, {".jpg", ".jpeg"}):
            _path = os.fspath(path)
            try:
                label = cls.get_label(_path)
            except Exception as e:
                logging.error("%s: %r", _path, e)
                continue

            paths.append(_path)
            labels.append(label)

        return cls(paths, labels, transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.paths[index]
        label = self.labels[index]

        with open(path, "rb") as fr:
            img = Image.open(fr).convert("RGB")

        dy = randrange(0, 4)

        if dy != 0:
            img = img.transpose(self.rot_actions[dy])
            label = (label + dy) % 4

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.paths)


class LightningSqueezeNet(pl.LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = models.SqueezeNet("1_1", num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx) -> float:
        imgs, labels = batch
        outputs = self.model(imgs)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        imgs, labels = batch

        outputs = self.model(imgs)
        loss = self.loss(outputs, labels)
        classes = outputs.argmax(dim=-1)
        acc = multiclass_accuracy(classes, labels, num_classes=4)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from genutility.args import is_dir

    DEFAULT_NUM_THREADS = (os.cpu_count() or 3) - 2
    DEFAULT_NUM_WORKERS = 2

    parser = ArgumentParser()
    parser.add_argument("action", choices=("train", "test", "preview"))
    parser.add_argument("--modelpath", default=Path("the-model"), type=is_dir)
    parser.add_argument("--data-path", type=is_dir, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--ckpt-path", default="best")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)

    dataset = cache(Path("cache"))(AugmentedRotationDataset.make)(args.data_path)
    callbacks = [ModelCheckpoint(), EarlyStopping(monitor="train_loss", mode="min")]
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=None, callbacks=callbacks)
    transform = T.Compose([T.PILToTensor(), T.Resize((224, 224), antialias=True), T.ConvertImageDtype(torch.float)])

    if args.action == "train":
        dataset = dataset.with_transform(transform)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=args.num_workers)
        model = LightningSqueezeNet(num_classes=dataset.get_num_classes())

        trainer.fit(model=model, train_dataloaders=data_loader)
        trainer.test(dataloaders=data_loader)

    elif args.action == "test":
        dataset = dataset.with_transform(transform)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)
        #model = LightningSqueezeNet.load_from_checkpoint(args.ckpt_path)
        model = LightningSqueezeNet(num_classes=dataset.get_num_classes())
        model.load_state_dict(torch.load(args.ckpt_path)["state_dict"])

        trainer.test(model=model, dataloaders=data_loader)

    elif args.action == "preview":
        preview(dataset)

    else:
        parser.error("invalid action")
