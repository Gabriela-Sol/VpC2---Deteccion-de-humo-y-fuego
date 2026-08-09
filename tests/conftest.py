"""Fixtures compartidas: dataset YOLO sintético en disco."""

from pathlib import Path

import pytest
from PIL import Image

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 80

TRAIN_LABELS = {
    "img_0": "0 0.5 0.5 0.2 0.4",
    "img_1": "1 0.25 0.25 0.2 0.2\n0 0.75 0.75 0.2 0.2",
    "img_2": "",
    "img_3": "0 0.9 0.5 0.4 0.4",
    "img_4": "1 0.5 0.5 0.004 0.004",
}

VAL_LABELS = {
    "val_0": "0 0.5 0.5 0.2 0.4",
    "val_1": "",
}


def _write_split(split_dir: Path, labels: dict) -> None:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for index, (stem, content) in enumerate(sorted(labels.items())):
        color = (10 * index % 256, 20 * index % 256, 30 * index % 256)
        image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color)
        image.save(images_dir / f"{stem}.jpg", quality=90)
        (labels_dir / f"{stem}.txt").write_text(content, encoding="utf-8")


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dfire_sintetico")
    _write_split(root / "train", TRAIN_LABELS)
    _write_split(root / "val", VAL_LABELS)
    return root
