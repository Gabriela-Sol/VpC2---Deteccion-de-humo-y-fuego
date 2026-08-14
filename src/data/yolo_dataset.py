"""Dataset en formato YOLO adaptado a la API de detección de torchvision.

torchvision espera cajas en xyxy absoluto y reserva la clase 0 para el fondo,
mientras que el dataset D-Fire usa cxcywh normalizado con clases 0 (smoke) y
1 (fire). Este módulo hace esa traducción y absorbe las irregularidades del
dataset: coordenadas fuera de rango, cajas degeneradas e imágenes sin objetos.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# torchvision usa el índice 0 para el fondo, así que las clases del dataset
# se desplazan en uno.
YOLO_TO_TORCHVISION_LABEL = {0: 1, 1: 2}
LABEL_ORDER = [1, 2]
CLASS_NAMES = {1: "smoke", 2: "fire"}

# Ancho o alto mínimo, en píxeles, para que una caja se considere válida.
MIN_BOX_SIZE_PX = 1.0


def _parse_label_line(line: str):
    """Devuelve `(yolo_class, cx, cy, w, h)` o `None` si la línea es inválida.

    Una sola definición de qué se considera una línea válida, para que el
    conteo de descartes del `__init__` y la lectura de `_load_boxes` no puedan
    divergir.
    """
    parts = line.split()
    if len(parts) != 5:
        return None

    try:
        yolo_class = int(float(parts[0]))
        center_x, center_y, box_w, box_h = (float(value) for value in parts[1:])
    except ValueError:
        return None

    if yolo_class not in YOLO_TO_TORCHVISION_LABEL:
        return None

    return yolo_class, center_x, center_y, box_w, box_h


class YoloDetectionDataset(Dataset):
    """Lee un split con estructura `<split_dir>/images` y `<split_dir>/labels`."""

    def __init__(
        self,
        split_dir: str | Path,
        train: bool = False,
        hflip_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"No existe el directorio de imágenes: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"No existe el directorio de etiquetas: {self.labels_dir}")

        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No se encontraron imágenes en {self.images_dir}")

        self.train = train
        self.hflip_prob = hflip_prob
        self._rng = random.Random(seed)

        self.malformed_label_lines = self._count_malformed_label_lines()
        if self.malformed_label_lines:
            # Un aviso por split, nunca uno por línea: en el dataset real serían
            # miles de mensajes. Descartar una caja real no solo pierde una
            # anotación: convierte una detección correcta en un falso positivo y
            # baja la precisión, así que conviene que se vea.
            warnings.warn(
                f"{self.split_dir}: se descartaron {self.malformed_label_lines} "
                "líneas de etiqueta mal formadas (campos faltantes, valores no "
                "numéricos o clase desconocida).",
                stacklevel=2,
            )

    def _count_malformed_label_lines(self) -> int:
        """Cuenta las líneas de etiqueta que `_load_boxes` va a descartar."""
        total = 0
        for image_path in self.image_paths:
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and _parse_label_line(line) is None:
                    total += 1
        return total

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes, labels = self._load_boxes(
            self.labels_dir / f"{image_path.stem}.txt", width, height
        )

        if self.train and self._rng.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return TF.to_tensor(image), target

    def _load_boxes(self, label_path: Path, width: int, height: int):
        """Devuelve (boxes xyxy float32 [N,4], labels int64 [N]).

        Las imágenes negativas y los archivos ausentes producen tensores vacíos
        con la forma que torchvision necesita para no fallar.
        """
        boxes: list[list[float]] = []
        labels: list[int] = []

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parseada = _parse_label_line(line)
                if parseada is None:
                    continue
                yolo_class, center_x, center_y, box_w, box_h = parseada

                x1 = (center_x - box_w / 2) * width
                y1 = (center_y - box_h / 2) * height
                x2 = (center_x + box_w / 2) * width
                y2 = (center_y + box_h / 2) * height

                # El EDA encontró cajas con coordenadas fuera de [0, 1]; se
                # recortan al marco en lugar de descartar la imagen entera.
                x1 = min(max(x1, 0.0), float(width))
                y1 = min(max(y1, 0.0), float(height))
                x2 = min(max(x2, 0.0), float(width))
                y2 = min(max(y2, 0.0), float(height))

                if x2 - x1 < MIN_BOX_SIZE_PX or y2 - y1 < MIN_BOX_SIZE_PX:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(YOLO_TO_TORCHVISION_LABEL[yolo_class])

        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.int64),
            )
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )


def collate_fn(batch):
    """Agrupa muestras sin apilarlas: las imágenes pueden tener distinto tamaño."""
    images, targets = zip(*batch)
    return list(images), list(targets)
