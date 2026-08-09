# Modelos adicionales de detección (Faster R-CNN + RT-DETR) — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar al repositorio un experimento de Faster R-CNN (torchvision) y uno de RT-DETR (Ultralytics) que produzcan los mismos artefactos de métricas y figuras que el baseline YOLOv8n ya existente, más una capa de comparación entre los tres modelos.

**Architecture:** El código reusable de entrenamiento, evaluación y reporte vive en el paquete `src/`, probado localmente en CPU con datasets sintéticos. Los notebooks de Colab quedan finos: arman la configuración, ejecutan el bucle de épocas y delegan todo el reporte en `src/reporting/experiment_report.py`. Un archivo `metrics_summary.csv` de esquema idéntico por experimento hace posible la comparación entre modelos de librerías distintas.

**Tech Stack:** Python 3.12, PyTorch 2.9, torchvision 0.24, torchmetrics 1.9, pycocotools, Ultralytics, pandas, matplotlib, pytest 8.4.

**Spec:** `docs/superpowers/specs/2026-08-09-modelos-adicionales-deteccion-design.md`

## Global Constraints

- El paquete de modelos se llama `src/modeling/`, **nunca** `src/models/`: el `.gitignore` línea 21 tiene la regla `models/`, que excluye cualquier directorio con ese nombre en cualquier nivel. Verificable con `git check-ignore -v src/models/x.py`.
- Etiquetas: el dataset YOLO usa `0: smoke`, `1: fire`. torchvision reserva el índice 0 para el fondo, así que dentro de `src/` las clases son `1: smoke`, `2: fire` y `num_classes=3`.
- Las imágenes sin cajas (6458 de 14122 en train) deben producir `boxes` de forma `(0, 4)` dtype `float32` y `labels` de forma `(0,)` dtype `int64`. Verificado: torchvision entrena y hace backward correctamente con esos targets.
- Todas las métricas comparativas se calculan sobre el split de **validación**, con protocolo COCO.
- `precision`, `recall` y `f1` en `metrics_summary.csv` se reportan en el umbral de confianza que maximiza F1.
- La matriz de confusión usa IoU 0.45 a confianza 0.25, con emparejamiento **agnóstico de clase** (igual que Ultralytics), para que los errores de clasificación aparezcan fuera de la diagonal.
- Las curvas P/R/F1/PR usan IoU 0.5.
- Faster R-CNN: 12 épocas. RT-DETR-l: 20 épocas.
- Los tests corren en CPU, sin GPU, sin red y sin descargar pesos preentrenados (`pretrained=False`).
- Entorno local: `.venv/` (ya creado con `--system-site-packages`, está en `.gitignore`). Todos los comandos de test usan `.venv/bin/python -m pytest`.
- Los pesos (`*.pt`, `*.pth`) no se versionan; viven en Google Drive.

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `requirements.txt` | modificar: agregar `torchmetrics`, `pycocotools` |
| `pytest.ini` | crear: `pythonpath = .` para que `import src...` funcione |
| `src/data/yolo_dataset.py` | leer el dataset YOLO y entregar targets de torchvision |
| `src/modeling/detectors.py` | construir Faster R-CNN y contar parámetros |
| `src/engine/matching.py` | primitivas puras: emparejar detecciones con GT, matriz de confusión |
| `src/engine/metrics.py` | mAP con torchmetrics, curvas de confianza, FPS |
| `src/engine/trainer.py` | optimizador, scheduler, una época, checkpoints |
| `src/reporting/summary.py` | esquema y escritura/lectura de `metrics_summary.csv` |
| `src/reporting/plots.py` | todas las figuras PNG |
| `src/reporting/experiment_report.py` | orquestador: de modelo entrenado a carpeta de resultados completa |
| `configs/experiments/fasterrcnn_r50fpn.yaml` | configuración del experimento Faster R-CNN |
| `configs/experiments/rtdetr_l.yaml` | configuración del experimento RT-DETR |
| `notebooks/03_entrenamiento_FasterRCNN.ipynb` | entrenamiento en Colab de Faster R-CNN |
| `notebooks/04_entrenamiento_RTDETR.ipynb` | entrenamiento en Colab de RT-DETR |
| `notebooks/05_comparacion_modelos.ipynb` | comparación de los tres modelos, sin GPU |
| `notebooks/02_entrenamiento_YOLO.ipynb` | modificar: celda que exporta el `metrics_summary.csv` del baseline |
| `tests/` | tests unitarios y smoke tests en CPU |

`matching.py` se separa de `metrics.py` a propósito: son funciones puras sobre arrays, sin dependencia de torchmetrics ni del modelo, y son la parte con más riesgo de tener un error silencioso. Aisladas se testean con cajas escritas a mano y resultados calculados a mano.

---

### Task 1: Scaffolding del paquete y arnés de tests

**Files:**
- Create: `pytest.ini`, `src/__init__.py`, `src/data/__init__.py`, `src/modeling/__init__.py`, `src/engine/__init__.py`, `src/reporting/__init__.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Consumes: nada.
- Produces: fixture `synthetic_dataset` (pytest, scope `session`) que devuelve un `pathlib.Path` a un directorio con la estructura `<root>/train/{images,labels}` y `<root>/val/{images,labels}`. Contiene 5 imágenes de 100×80 px (ancho×alto) en train, descritas en la tabla del Step 3, y 2 en val (`val_0` copia de `img_0`, `val_1` copia de `img_2`).

- [ ] **Step 1: Verificar la trampa del `.gitignore` antes de crear nada**

```bash
git check-ignore -v src/models/detectors.py   # DEBE imprimir ".gitignore:21:models/"
git check-ignore -v src/modeling/detectors.py # DEBE salir con código 1 y sin salida
```

Si la primera línea no coincide, detenerse y revisar el `.gitignore` antes de continuar. Todo el plan asume `src/modeling/`.

- [ ] **Step 2: Crear `pytest.ini` y los `__init__.py`**

`pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
addopts = -q
filterwarnings =
    ignore::UserWarning
```

Los seis `__init__.py` van vacíos:

```bash
mkdir -p src/data src/modeling src/engine src/reporting tests
touch src/__init__.py src/data/__init__.py src/modeling/__init__.py \
      src/engine/__init__.py src/reporting/__init__.py
```

- [ ] **Step 3: Escribir `tests/conftest.py`**

Las cinco imágenes de train cubren cada caso borde que el dataset debe manejar:

| Imagen | Contenido del `.txt` | Qué ejercita |
|---|---|---|
| `img_0` | `0 0.5 0.5 0.2 0.4` | una caja smoke normal |
| `img_1` | `1 0.25 0.25 0.2 0.2` y `0 0.75 0.75 0.2 0.2` | dos cajas, dos clases |
| `img_2` | archivo vacío | imagen negativa |
| `img_3` | `0 0.9 0.5 0.4 0.4` | caja que se sale del borde derecho, requiere clampeo |
| `img_4` | `1 0.5 0.5 0.004 0.004` | caja degenerada (0.4 px de ancho), se descarta |

```python
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
```

- [ ] **Step 4: Escribir el test del arnés**

`tests/test_scaffolding.py`:

```python
"""Verifica que el paquete es importable y que la fixture arma el dataset."""

from pathlib import Path


def test_paquetes_importables():
    import src.data
    import src.engine
    import src.modeling
    import src.reporting


def test_dataset_sintetico_tiene_la_estructura_esperada(synthetic_dataset: Path):
    for split, cantidad in [("train", 5), ("val", 2)]:
        images = sorted((synthetic_dataset / split / "images").glob("*.jpg"))
        labels = sorted((synthetic_dataset / split / "labels").glob("*.txt"))
        assert len(images) == cantidad
        assert len(labels) == cantidad

    vacio = synthetic_dataset / "train" / "labels" / "img_2.txt"
    assert vacio.read_text() == ""
```

- [ ] **Step 5: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_scaffolding.py -v`
Expected: 2 passed.

- [ ] **Step 6: Agregar las dependencias a `requirements.txt`**

Agregar al final del archivo, después de `tqdm`:

```
torchmetrics
pycocotools
```

- [ ] **Step 7: Commit**

```bash
git add pytest.ini requirements.txt src tests
git commit -m "chore: scaffolding del paquete src y arnes de tests"
```

---

### Task 2: Dataset YOLO para torchvision

**Files:**
- Create: `src/data/yolo_dataset.py`
- Test: `tests/test_yolo_dataset.py`

**Interfaces:**
- Consumes: fixture `synthetic_dataset` de Task 1.
- Produces:
  - `LABEL_ORDER: list[int] = [1, 2]`
  - `CLASS_NAMES: dict[int, str] = {1: "smoke", 2: "fire"}`
  - `YoloDetectionDataset(split_dir: str | Path, train: bool = False, hflip_prob: float = 0.5, seed: int | None = None)`, cuyo `__getitem__` devuelve `(image: Tensor[3,H,W] float32 en [0,1], target: dict)` con claves `boxes` (`float32 [N,4]` xyxy absoluto), `labels` (`int64 [N]`) e `image_id` (`int64` escalar).
  - `collate_fn(batch) -> tuple[list[Tensor], list[dict]]`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_yolo_dataset.py`:

```python
"""Tests del dataset YOLO -> torchvision."""

from pathlib import Path

import torch

from src.data.yolo_dataset import (
    CLASS_NAMES,
    LABEL_ORDER,
    YoloDetectionDataset,
    collate_fn,
)

# La fixture crea imágenes de 100 px de ancho por 80 de alto.
ANCHO = 100
ALTO = 80


def _por_nombre(dataset: YoloDetectionDataset, stem: str):
    for indice in range(len(dataset)):
        if dataset.image_paths[indice].stem == stem:
            return dataset[indice]
    raise AssertionError(f"No se encontró {stem} en el dataset")


def test_constantes_de_clases():
    assert LABEL_ORDER == [1, 2]
    assert CLASS_NAMES == {1: "smoke", 2: "fire"}


def test_convierte_cxcywh_normalizado_a_xyxy_absoluto(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    imagen, target = _por_nombre(dataset, "img_0")

    assert imagen.shape == (3, ALTO, ANCHO)
    assert imagen.dtype == torch.float32
    assert 0.0 <= float(imagen.min()) and float(imagen.max()) <= 1.0

    # 0 0.5 0.5 0.2 0.4 -> cx=50, cy=40, w=20, h=32 -> (40, 24, 60, 56)
    assert target["boxes"].shape == (1, 4)
    assert torch.allclose(
        target["boxes"][0], torch.tensor([40.0, 24.0, 60.0, 56.0]), atol=1e-4
    )


def test_remapea_las_clases_yolo_a_torchvision(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    _, target = _por_nombre(dataset, "img_1")

    # El archivo trae "1 ..." (fire) y luego "0 ..." (smoke), en ese orden.
    assert target["labels"].tolist() == [2, 1]
    assert target["labels"].dtype == torch.int64


def test_imagen_negativa_devuelve_tensores_vacios_bien_formados(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    _, target = _por_nombre(dataset, "img_2")

    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].shape == (0,)
    assert target["labels"].dtype == torch.int64


def test_clampea_cajas_que_se_salen_de_la_imagen(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    _, target = _por_nombre(dataset, "img_3")

    # 0 0.9 0.5 0.4 0.4 -> x va de 70 a 110; 110 excede el ancho y se recorta a 100.
    caja = target["boxes"][0]
    assert torch.allclose(caja, torch.tensor([70.0, 24.0, 100.0, 56.0]), atol=1e-4)
    assert float(caja[2]) <= ANCHO
    assert float(caja[3]) <= ALTO


def test_descarta_cajas_degeneradas(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    _, target = _por_nombre(dataset, "img_4")

    # 0.004 * 100 = 0.4 px de ancho, por debajo del mínimo de 1 px.
    assert target["boxes"].shape == (0, 4)


def test_volteo_horizontal_refleja_las_cajas(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(
        synthetic_dataset / "train", train=True, hflip_prob=1.0, seed=0
    )
    _, target = _por_nombre(dataset, "img_3")

    # Original (70, 24, 100, 56) reflejado en x -> (100-100, ..., 100-70) = (0, 24, 30, 56)
    assert torch.allclose(
        target["boxes"][0], torch.tensor([0.0, 24.0, 30.0, 56.0]), atol=1e-4
    )


def test_hflip_prob_cero_no_modifica_las_cajas(synthetic_dataset: Path):
    sin_flip = YoloDetectionDataset(
        synthetic_dataset / "train", train=True, hflip_prob=0.0, seed=0
    )
    _, target = _por_nombre(sin_flip, "img_0")
    assert torch.allclose(
        target["boxes"][0], torch.tensor([40.0, 24.0, 60.0, 56.0]), atol=1e-4
    )


def test_collate_fn_devuelve_listas_paralelas(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    imagenes, targets = collate_fn([dataset[0], dataset[1]])

    assert isinstance(imagenes, list) and isinstance(targets, list)
    assert len(imagenes) == len(targets) == 2
    assert imagenes[0].shape[0] == 3


def test_image_id_es_unico_por_muestra(synthetic_dataset: Path):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    ids = [int(dataset[i][1]["image_id"]) for i in range(len(dataset))]
    assert sorted(ids) == list(range(len(dataset)))
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_yolo_dataset.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.yolo_dataset'`.

- [ ] **Step 3: Implementar `src/data/yolo_dataset.py`**

```python
"""Dataset en formato YOLO adaptado a la API de detección de torchvision.

torchvision espera cajas en xyxy absoluto y reserva la clase 0 para el fondo,
mientras que el dataset D-Fire usa cxcywh normalizado con clases 0 (smoke) y
1 (fire). Este módulo hace esa traducción y absorbe las irregularidades del
dataset: coordenadas fuera de rango, cajas degeneradas e imágenes sin objetos.
"""

from __future__ import annotations

import random
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
                parts = line.split()
                if len(parts) != 5:
                    continue

                try:
                    yolo_class = int(float(parts[0]))
                    center_x, center_y, box_w, box_h = (float(value) for value in parts[1:])
                except ValueError:
                    continue

                if yolo_class not in YOLO_TO_TORCHVISION_LABEL:
                    continue

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
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_yolo_dataset.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/yolo_dataset.py tests/test_yolo_dataset.py
git commit -m "feat: dataset YOLO adaptado a torchvision con manejo de negativos"
```

---

### Task 3: Constructor de Faster R-CNN

**Files:**
- Create: `src/modeling/detectors.py`
- Test: `tests/test_detectors.py`

**Interfaces:**
- Consumes: nada de tareas anteriores.
- Produces:
  - `AVAILABLE_BACKBONES: dict[str, tuple]` con las claves `"resnet50_fpn_v2"` y `"mobilenet_v3_large_fpn"`.
  - `build_fasterrcnn(num_classes: int = 3, backbone: str = "resnet50_fpn_v2", trainable_backbone_layers: int = 3, min_size: int = 640, max_size: int = 1024, pretrained: bool = True) -> FasterRCNN`
  - `count_parameters(model) -> int` (solo entrenables)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_detectors.py`:

```python
"""Tests del constructor de Faster R-CNN. Corren sin red: pretrained=False."""

import pytest
import torch

from src.modeling.detectors import (
    AVAILABLE_BACKBONES,
    build_fasterrcnn,
    count_parameters,
)


def test_backbones_disponibles():
    assert set(AVAILABLE_BACKBONES) == {"resnet50_fpn_v2", "mobilenet_v3_large_fpn"}


def test_cabezal_tiene_la_cantidad_de_clases_pedida():
    model = build_fasterrcnn(
        num_classes=3, backbone="mobilenet_v3_large_fpn", pretrained=False
    )
    assert model.roi_heads.box_predictor.cls_score.out_features == 3
    # 4 coordenadas por clase.
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 12


def test_min_size_y_max_size_llegan_al_transform():
    model = build_fasterrcnn(
        backbone="mobilenet_v3_large_fpn", min_size=640, max_size=1024, pretrained=False
    )
    assert model.transform.min_size == (640,)
    assert model.transform.max_size == 1024


def test_backbone_desconocido_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="resnet50_fpn_v2"):
        build_fasterrcnn(backbone="inexistente", pretrained=False)


def test_count_parameters_cuenta_solo_entrenables():
    model = build_fasterrcnn(backbone="mobilenet_v3_large_fpn", pretrained=False)
    total = count_parameters(model)
    assert total > 1_000_000

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert count_parameters(model) == 0


def test_forward_de_entrenamiento_acepta_un_target_vacio():
    model = build_fasterrcnn(
        num_classes=3, backbone="mobilenet_v3_large_fpn", min_size=64, max_size=128,
        pretrained=False,
    )
    model.train()

    images = [torch.rand(3, 80, 100), torch.rand(3, 80, 100)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 70.0]]),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    ]

    losses = model(images, targets)
    assert set(losses) == {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total.detach())


def test_forward_de_inferencia_devuelve_boxes_scores_labels():
    model = build_fasterrcnn(
        backbone="mobilenet_v3_large_fpn", min_size=64, max_size=128, pretrained=False
    )
    model.eval()

    with torch.no_grad():
        salidas = model([torch.rand(3, 80, 100)])

    assert sorted(salidas[0]) == ["boxes", "labels", "scores"]
    assert salidas[0]["boxes"].shape[1] == 4
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_detectors.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.modeling.detectors'`.

- [ ] **Step 3: Implementar `src/modeling/detectors.py`**

```python
"""Construcción de detectores Faster R-CNN sobre torchvision."""

from __future__ import annotations

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FastRCNNPredictor,
)

# nombre -> (constructor, enum de pesos COCO)
AVAILABLE_BACKBONES = {
    "resnet50_fpn_v2": (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    ),
    "mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    ),
}


def build_fasterrcnn(
    num_classes: int = 3,
    backbone: str = "resnet50_fpn_v2",
    trainable_backbone_layers: int = 3,
    min_size: int = 640,
    max_size: int = 1024,
    pretrained: bool = True,
) -> FasterRCNN:
    """Arma un Faster R-CNN con el cabezal ajustado a `num_classes`.

    `num_classes` incluye el fondo: para smoke y fire el valor es 3.
    Con `pretrained=False` no se descarga ningún peso, que es lo que necesitan
    los tests para correr sin red.
    """
    if backbone not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Backbone desconocido: {backbone!r}. "
            f"Opciones válidas: {sorted(AVAILABLE_BACKBONES)}"
        )

    builder, weights_enum = AVAILABLE_BACKBONES[backbone]

    model = builder(
        weights=weights_enum.DEFAULT if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_parameters(model) -> int:
    """Cantidad de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_detectors.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/modeling/detectors.py tests/test_detectors.py
git commit -m "feat: constructor de Faster R-CNN con backbone configurable"
```

---

### Task 4: Primitivas de emparejamiento y matriz de confusión

**Files:**
- Create: `src/engine/matching.py`
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes: nada de tareas anteriores.
- Produces:
  - `MatchResult` (dataclass) con campos `scores: np.ndarray`, `true_positive: np.ndarray` (bool), `labels: np.ndarray`, `n_ground_truth: dict[int, int]`.
  - `match_dataset(predictions: list[dict], targets: list[dict], iou_threshold: float = 0.5) -> MatchResult`
  - `confusion_matrix(predictions, targets, label_order: list[int], conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> np.ndarray` de forma `(len(label_order) + 1, len(label_order) + 1)`, indexada `[predicho, real]`, con el último índice reservado al fondo.

`predictions` es una lista de dicts con `boxes`, `scores` y `labels` (tensores). `targets` es una lista de dicts con `boxes` y `labels`. Ambas listas están alineadas por imagen.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_matching.py`:

```python
"""Tests de las primitivas de emparejamiento por IoU.

Las cajas están elegidas para que los IoU sean fáciles de verificar a mano.
"""

import numpy as np
import pytest
import torch

from src.engine.matching import confusion_matrix, match_dataset


def _pred(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _gt(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_deteccion_perfecta_es_verdadero_positivo():
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
    )
    assert resultado.true_positive.tolist() == [True]
    assert resultado.n_ground_truth == {1: 1}


def test_iou_por_debajo_del_umbral_es_falso_positivo():
    # Solapamiento de 2x10 sobre una unión de 18x10 -> IoU = 0.111
    resultado = match_dataset(
        [_pred([[8, 0, 18, 10]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        iou_threshold=0.5,
    )
    assert resultado.true_positive.tolist() == [False]


def test_clase_distinta_no_empareja():
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10]], [0.9], [2])],
        [_gt([[0, 0, 10, 10]], [1])],
    )
    assert resultado.true_positive.tolist() == [False]
    assert resultado.n_ground_truth == {1: 1}


def test_solo_la_deteccion_de_mayor_score_se_queda_con_el_gt():
    # Dos detecciones sobre un único objeto: la segunda es un duplicado.
    # Los scores son 0.5 y 0.75 porque ambos son exactos en binario; con 0.6 y
    # 0.9 la comparación por igualdad fallaría contra el redondeo de float32.
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10], [0, 0, 10, 10]], [0.5, 0.75], [1, 1])],
        [_gt([[0, 0, 10, 10]], [1])],
    )
    # El resultado viene ordenado por score descendente.
    assert resultado.scores.tolist() == [0.75, 0.5]
    assert resultado.true_positive.tolist() == [True, False]


def test_imagen_negativa_sin_detecciones_no_aporta_nada():
    resultado = match_dataset(
        [_pred(np.zeros((0, 4)), [], [])],
        [_gt(np.zeros((0, 4)), [])],
    )
    assert resultado.scores.size == 0
    assert resultado.n_ground_truth == {}


def test_deteccion_sobre_imagen_negativa_es_falso_positivo():
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10]], [0.8], [1])],
        [_gt(np.zeros((0, 4)), [])],
    )
    assert resultado.true_positive.tolist() == [False]
    assert resultado.n_ground_truth == {}


def test_cuenta_ground_truth_por_clase_en_varias_imagenes():
    resultado = match_dataset(
        [
            _pred(np.zeros((0, 4)), [], []),
            _pred(np.zeros((0, 4)), [], []),
        ],
        [
            _gt([[0, 0, 10, 10], [20, 20, 30, 30]], [1, 2]),
            _gt([[0, 0, 10, 10]], [1]),
        ],
    )
    assert resultado.n_ground_truth == {1: 2, 2: 1}


def test_matriz_de_confusion_acierto_va_a_la_diagonal():
    matriz = confusion_matrix(
        [_pred([[0, 0, 10, 10]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
    )
    assert matriz.shape == (3, 3)
    assert matriz[0, 0] == 1
    assert matriz.sum() == 1


def test_matriz_de_confusion_registra_confusion_entre_clases():
    # Emparejamiento agnóstico de clase: se detecta fire donde había smoke.
    matriz = confusion_matrix(
        [_pred([[0, 0, 10, 10]], [0.9], [2])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
    )
    # fila = predicho (fire -> 1), columna = real (smoke -> 0)
    assert matriz[1, 0] == 1
    assert matriz.sum() == 1


def test_matriz_de_confusion_falso_positivo_va_a_la_columna_de_fondo():
    matriz = confusion_matrix(
        [_pred([[100, 100, 110, 110]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
    )
    assert matriz[0, 2] == 1  # predicho smoke, no había nada
    assert matriz[2, 0] == 1  # había smoke, no se detectó
    assert matriz.sum() == 2


def test_matriz_de_confusion_ignora_detecciones_bajo_el_umbral_de_confianza():
    matriz = confusion_matrix(
        [_pred([[0, 0, 10, 10]], [0.1], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
        conf_threshold=0.25,
    )
    # La detección se descarta, así que el objeto queda sin detectar.
    assert matriz[2, 0] == 1
    assert matriz.sum() == 1


def test_matriz_de_confusion_solo_la_de_mayor_score_consume_el_gt():
    # Sin esta prueba, borrar la línea que marca el GT como consumido no
    # rompería ningún test, y los duplicados dejarían de penalizarse.
    matriz = confusion_matrix(
        [_pred([[0, 0, 10, 10], [0, 0, 10, 10]], [0.5, 0.75], [1, 1])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
    )
    assert matriz[0, 0] == 1  # la de score 0.75 acierta
    assert matriz[0, 2] == 1  # la de score 0.5 queda como falso positivo
    assert matriz.sum() == 2


def test_match_dataset_exige_listas_alineadas():
    with pytest.raises(ValueError, match="alineados"):
        match_dataset([_pred([[0, 0, 10, 10]], [0.9], [1])], [])


def test_confusion_matrix_exige_listas_alineadas():
    # Sin esta validación, zip() truncaría en silencio y las métricas
    # derivadas quedarían mal sin que nada fallara.
    with pytest.raises(ValueError, match="alineados"):
        confusion_matrix(
            [_pred([[0, 0, 10, 10]], [0.9], [1])], [], label_order=[1, 2]
        )
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.engine.matching'`.

- [ ] **Step 3: Implementar `src/engine/matching.py`**

```python
"""Emparejamiento de detecciones con ground truth por IoU.

Son funciones puras sobre listas de predicciones y targets. De acá salen tanto
las curvas de precisión y recall como la matriz de confusión, así que el
criterio de emparejamiento queda definido en un solo lugar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision.ops import box_iou


@dataclass
class MatchResult:
    """Detecciones de todo el dataset ordenadas por score descendente."""

    scores: np.ndarray
    true_positive: np.ndarray
    labels: np.ndarray
    n_ground_truth: dict[int, int] = field(default_factory=dict)


def _match_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Empareja de forma golosa, por score descendente, exigiendo misma clase.

    Devuelve un tensor booleano alineado con las detecciones ya ordenadas.
    Cada objeto real puede consumir una sola detección: las siguientes que lo
    solapen cuentan como falsos positivos, que es lo que penaliza los duplicados.
    """
    true_positive = torch.zeros(len(pred_scores), dtype=torch.bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return true_positive

    ious = box_iou(pred_boxes, gt_boxes)
    gt_taken = torch.zeros(len(gt_boxes), dtype=torch.bool)

    for index in range(len(pred_boxes)):
        elegibles = (gt_labels == pred_labels[index]) & (~gt_taken)
        if not bool(elegibles.any()):
            continue

        candidatos = ious[index].clone()
        candidatos[~elegibles] = -1.0
        mejor = int(torch.argmax(candidatos))

        if float(candidatos[mejor]) >= iou_threshold:
            true_positive[index] = True
            gt_taken[mejor] = True

    return true_positive


def match_dataset(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Empareja todas las imágenes y concatena el resultado."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions y targets deben estar alineados: "
            f"{len(predictions)} != {len(targets)}"
        )

    scores_por_imagen: list[torch.Tensor] = []
    tp_por_imagen: list[torch.Tensor] = []
    labels_por_imagen: list[torch.Tensor] = []
    n_ground_truth: dict[int, int] = {}

    for prediction, target in zip(predictions, targets):
        gt_labels = target["labels"].detach().cpu()
        for label in gt_labels.tolist():
            n_ground_truth[label] = n_ground_truth.get(label, 0) + 1

        pred_scores = prediction["scores"].detach().cpu()
        orden = torch.argsort(pred_scores, descending=True)

        pred_boxes = prediction["boxes"].detach().cpu()[orden]
        pred_scores = pred_scores[orden]
        pred_labels = prediction["labels"].detach().cpu()[orden]

        tp_por_imagen.append(
            _match_image(
                pred_boxes,
                pred_scores,
                pred_labels,
                target["boxes"].detach().cpu(),
                gt_labels,
                iou_threshold,
            )
        )
        scores_por_imagen.append(pred_scores)
        labels_por_imagen.append(pred_labels)

    scores = torch.cat(scores_por_imagen) if scores_por_imagen else torch.zeros(0)
    true_positive = torch.cat(tp_por_imagen) if tp_por_imagen else torch.zeros(0, dtype=torch.bool)
    labels = torch.cat(labels_por_imagen) if labels_por_imagen else torch.zeros(0, dtype=torch.int64)

    orden_global = torch.argsort(scores, descending=True)

    return MatchResult(
        scores=scores[orden_global].numpy(),
        true_positive=true_positive[orden_global].numpy(),
        labels=labels[orden_global].numpy(),
        n_ground_truth=n_ground_truth,
    )


def confusion_matrix(
    predictions: list[dict],
    targets: list[dict],
    label_order: list[int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """Matriz `[predicho, real]` con una fila y columna extra para el fondo.

    El emparejamiento es agnóstico de clase, igual que en Ultralytics: así una
    detección con la caja correcta pero la clase equivocada cae fuera de la
    diagonal en vez de contarse como un falso positivo más un falso negativo.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions y targets deben estar alineados: "
            f"{len(predictions)} != {len(targets)}"
        )

    n_clases = len(label_order)
    fondo = n_clases
    indice_de_label = {label: posicion for posicion, label in enumerate(label_order)}
    matriz = np.zeros((n_clases + 1, n_clases + 1), dtype=np.int64)

    for prediction, target in zip(predictions, targets):
        scores = prediction["scores"].detach().cpu()
        conservar = scores >= conf_threshold

        pred_boxes = prediction["boxes"].detach().cpu()[conservar]
        pred_labels = prediction["labels"].detach().cpu()[conservar]
        scores = scores[conservar]

        orden = torch.argsort(scores, descending=True)
        pred_boxes, pred_labels = pred_boxes[orden], pred_labels[orden]

        gt_boxes = target["boxes"].detach().cpu()
        gt_labels = target["labels"].detach().cpu()

        gt_taken = torch.zeros(len(gt_boxes), dtype=torch.bool)
        pred_taken = torch.zeros(len(pred_boxes), dtype=torch.bool)

        if len(pred_boxes) and len(gt_boxes):
            ious = box_iou(pred_boxes, gt_boxes)
            for index in range(len(pred_boxes)):
                candidatos = ious[index].clone()
                candidatos[gt_taken] = -1.0
                mejor = int(torch.argmax(candidatos))
                if float(candidatos[mejor]) >= iou_threshold:
                    matriz[
                        indice_de_label[int(pred_labels[index])],
                        indice_de_label[int(gt_labels[mejor])],
                    ] += 1
                    gt_taken[mejor] = True
                    pred_taken[index] = True

        for index in range(len(pred_boxes)):
            if not bool(pred_taken[index]):
                matriz[indice_de_label[int(pred_labels[index])], fondo] += 1

        for index in range(len(gt_boxes)):
            if not bool(gt_taken[index]):
                matriz[fondo, indice_de_label[int(gt_labels[index])]] += 1

    return matriz
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/matching.py tests/test_matching.py
git commit -m "feat: emparejamiento por IoU y matriz de confusion"
```

---

### Task 5: Métricas — mAP, curvas de confianza y FPS

**Files:**
- Create: `src/engine/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `match_dataset`, `MatchResult` de `src/engine/matching.py`; `LABEL_ORDER`, `CLASS_NAMES`, `collate_fn` de `src/data/yolo_dataset.py`.
- Produces:
  - `collect_predictions(model, loader, device) -> tuple[list[dict], list[dict]]` (predicciones y targets, en CPU).
  - `compute_map(predictions, targets, label_order=LABEL_ORDER) -> dict` con claves `map50`, `map50_95`, `map50_per_class`, `map50_95_per_class` (estas dos, dicts `label -> float`).
  - `compute_curves(predictions, targets, label_order=LABEL_ORDER, iou_threshold=0.5, n_thresholds=101) -> dict` con claves `confidence`, `precision`, `recall`, `f1` (arrays de largo `n_thresholds`), `best_confidence`, `best_f1`, `best_precision`, `best_recall`, y `pr_per_class` (`label -> {"recall": array, "precision": array, "ap": float}`).
  - `measure_inference_fps(model, dataset, device, num_images=50, warmup=5) -> float`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_metrics.py`:

```python
"""Tests de métricas: mAP, curvas de confianza y FPS."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.yolo_dataset import LABEL_ORDER, YoloDetectionDataset, collate_fn
from src.engine.metrics import (
    collect_predictions,
    compute_curves,
    compute_map,
    measure_inference_fps,
)
from src.modeling.detectors import build_fasterrcnn


def _pred(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _gt(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_compute_map_con_deteccion_perfecta_da_uno():
    predicciones = [_pred([[0, 0, 10, 10], [20, 20, 40, 40]], [0.99, 0.98], [1, 2])]
    targets = [_gt([[0, 0, 10, 10], [20, 20, 40, 40]], [1, 2])]

    metricas = compute_map(predicciones, targets)

    assert metricas["map50"] == 1.0
    assert metricas["map50_95"] == 1.0
    assert metricas["map50_per_class"][1] == 1.0
    assert metricas["map50_per_class"][2] == 1.0


def test_compute_map_devuelve_una_entrada_por_clase_del_label_order():
    predicciones = [_pred([[0, 0, 10, 10]], [0.9], [1])]
    targets = [_gt([[0, 0, 10, 10]], [1])]

    metricas = compute_map(predicciones, targets)

    # La clase 2 no aparece en los datos; debe quedar como NaN, no ausente.
    assert set(metricas["map50_per_class"]) == set(LABEL_ORDER)
    assert np.isnan(metricas["map50_per_class"][2])


def test_compute_curves_tiene_las_formas_correctas():
    predicciones = [_pred([[0, 0, 10, 10]], [0.9], [1])]
    targets = [_gt([[0, 0, 10, 10]], [1])]

    curvas = compute_curves(predicciones, targets, n_thresholds=101)

    for clave in ["confidence", "precision", "recall", "f1"]:
        assert curvas[clave].shape == (101,)
    assert curvas["confidence"][0] == 0.0
    assert curvas["confidence"][-1] == 1.0


def test_curvas_de_deteccion_perfecta_valen_uno_bajo_el_score():
    # El score es 0.75 y no 0.9 a propósito: 0.75 es exacto en binario, mientras
    # que float32(0.9) vale 0.899999976 y quedaría por debajo del umbral 0.9.
    predicciones = [_pred([[0, 0, 10, 10]], [0.75], [1])]
    targets = [_gt([[0, 0, 10, 10]], [1])]

    curvas = compute_curves(predicciones, targets, n_thresholds=101)
    bajo_el_score = curvas["confidence"] <= 0.75

    assert np.allclose(curvas["precision"][bajo_el_score], 1.0)
    assert np.allclose(curvas["recall"][bajo_el_score], 1.0)
    assert curvas["best_f1"] == 1.0


def test_recall_cae_a_cero_por_encima_del_score_maximo():
    predicciones = [_pred([[0, 0, 10, 10]], [0.5], [1])]
    targets = [_gt([[0, 0, 10, 10]], [1])]

    curvas = compute_curves(predicciones, targets, n_thresholds=101)
    sobre_el_score = curvas["confidence"] > 0.5

    assert np.allclose(curvas["recall"][sobre_el_score], 0.0)


def test_best_confidence_maximiza_f1():
    # Un acierto con score alto y un falso positivo con score bajo:
    # el mejor F1 se logra descartando el falso positivo.
    predicciones = [_pred([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.2], [1, 1])]
    targets = [_gt([[0, 0, 10, 10]], [1])]

    curvas = compute_curves(predicciones, targets, n_thresholds=101)

    assert curvas["best_f1"] == 1.0
    assert 0.2 < curvas["best_confidence"] <= 0.9
    assert curvas["best_precision"] == 1.0
    assert curvas["best_recall"] == 1.0


def test_pr_per_class_incluye_average_precision():
    predicciones = [_pred([[0, 0, 10, 10]], [0.9], [1])]
    targets = [_gt([[0, 0, 10, 10]], [1])]

    curvas = compute_curves(predicciones, targets)

    assert set(curvas["pr_per_class"]) == set(LABEL_ORDER)
    assert curvas["pr_per_class"][1]["ap"] == 1.0
    assert len(curvas["pr_per_class"][1]["recall"]) == len(
        curvas["pr_per_class"][1]["precision"]
    )


def test_ap_ignora_los_falsos_positivos_de_menor_score():
    # Con la detección de mayor score acertando, la envolvente de precisión se
    # mantiene en 1.0 hasta el recall máximo, así que el AP es exactamente 1.0
    # por más falsos positivos de menor score que vengan atrás. Interpolar
    # linealmente en lugar de usar el escalón de COCO daría 0.9921.
    predicciones = [
        _pred(
            [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210],
             [300, 300, 310, 310], [400, 400, 410, 410]],
            [0.75, 0.5, 0.25, 0.125, 0.0625],
            [1, 1, 1, 1, 1],
        )
    ]
    targets = [_gt([[0, 0, 10, 10]], [1])]

    curvas = compute_curves(predicciones, targets)

    assert curvas["pr_per_class"][1]["ap"] == pytest.approx(1.0)


def test_sin_ground_truth_las_curvas_no_explotan():
    predicciones = [_pred([[0, 0, 10, 10]], [0.9], [1])]
    targets = [_gt(np.zeros((0, 4)), [])]

    curvas = compute_curves(predicciones, targets)

    assert np.allclose(curvas["recall"], 0.0)
    assert curvas["best_f1"] == 0.0


def test_collect_predictions_recorre_el_loader(synthetic_dataset):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = build_fasterrcnn(
        backbone="mobilenet_v3_large_fpn", min_size=64, max_size=128, pretrained=False
    )

    predicciones, targets = collect_predictions(model, loader, torch.device("cpu"))

    assert len(predicciones) == len(targets) == len(dataset)
    assert sorted(predicciones[0]) == ["boxes", "labels", "scores"]
    assert predicciones[0]["boxes"].device.type == "cpu"


def test_measure_inference_fps_es_positivo(synthetic_dataset):
    dataset = YoloDetectionDataset(synthetic_dataset / "train")
    model = build_fasterrcnn(
        backbone="mobilenet_v3_large_fpn", min_size=64, max_size=128, pretrained=False
    )

    fps = measure_inference_fps(
        model, dataset, torch.device("cpu"), num_images=2, warmup=1
    )

    assert fps > 0.0
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.engine.metrics'`.

- [ ] **Step 3: Implementar `src/engine/metrics.py`**

```python
"""Métricas de detección: mAP con protocolo COCO, curvas y velocidad.

El mAP sale de torchmetrics, que usa pycocotools por debajo, el mismo criterio
que aplica Ultralytics. Las curvas de precisión, recall y F1 en función de la
confianza se calculan acá a partir del emparejamiento de `matching.py`, porque
torchmetrics no las expone.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from src.data.yolo_dataset import LABEL_ORDER
from src.engine.matching import match_dataset

# Detecciones por debajo de este score no aportan a ninguna métrica y sí
# encarecen el cálculo.
MIN_SCORE = 1e-3


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Corre el modelo sobre todo el loader y devuelve predicciones y targets en CPU."""
    model.eval()
    model.to(device)

    predictions: list[dict] = []
    targets: list[dict] = []

    for images, batch_targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for output, target in zip(outputs, batch_targets):
            conservar = output["scores"].detach().cpu() >= MIN_SCORE
            predictions.append(
                {
                    "boxes": output["boxes"].detach().cpu()[conservar],
                    "scores": output["scores"].detach().cpu()[conservar],
                    "labels": output["labels"].detach().cpu()[conservar],
                }
            )
            targets.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

    return predictions, targets


def _per_class_dict(resultado, label_order: list[int]) -> dict[int, float]:
    """Convierte `map_per_class` de torchmetrics en un dict label -> valor.

    Las clases sin ningún objeto real quedan como NaN en lugar del -1 que
    devuelve pycocotools, para que promedios y gráficos las ignoren solos.
    """
    clases = np.atleast_1d(resultado["classes"].cpu().numpy())
    valores = np.atleast_1d(resultado["map_per_class"].cpu().numpy())
    presentes = {int(c): float(v) for c, v in zip(clases, valores)}

    salida: dict[int, float] = {}
    for label in label_order:
        valor = presentes.get(label, -1.0)
        salida[label] = float("nan") if valor < 0 else valor
    return salida


def compute_map(
    predictions: list[dict],
    targets: list[dict],
    label_order: list[int] = LABEL_ORDER,
) -> dict:
    """mAP50 y mAP50-95, globales y por clase.

    Se instancian dos métricas porque torchmetrics expone `map_per_class` solo
    para el promedio sobre los umbrales configurados: para obtener el mAP50 por
    clase hay que restringir `iou_thresholds` a [0.5].
    """
    metrica_completa = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True
    )
    metrica_completa.update(predictions, targets)
    resultado_completo = metrica_completa.compute()

    metrica_50 = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=True
    )
    metrica_50.update(predictions, targets)
    resultado_50 = metrica_50.compute()

    def _escalar(valor) -> float:
        numero = float(valor)
        return float("nan") if numero < 0 else numero

    return {
        "map50": _escalar(resultado_completo["map_50"]),
        "map50_95": _escalar(resultado_completo["map"]),
        "map50_per_class": _per_class_dict(resultado_50, label_order),
        "map50_95_per_class": _per_class_dict(resultado_completo, label_order),
    }


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP en 101 puntos con el criterio de COCO.

    COCO no interpola linealmente entre puntos de recall: para cada umbral toma
    la precisión envolvente del primer recall que lo alcanza, o sea una función
    escalonada. Interpolar linealmente suaviza las caídas de precisión y
    sobreestima el AP, y además daría un número que no coincide con el
    `map50_per_class` que devuelve torchmetrics para la misma clase.
    """
    if recall.size == 0:
        return 0.0

    # La precisión se vuelve monótona decreciente de derecha a izquierda.
    precision_envolvente = np.maximum.accumulate(precision[::-1])[::-1]
    puntos = np.linspace(0.0, 1.0, 101)

    indices = np.searchsorted(recall, puntos, side="left")
    valores = np.zeros(puntos.size)
    alcanzables = indices < precision_envolvente.size
    valores[alcanzables] = precision_envolvente[indices[alcanzables]]

    return float(valores.mean())


def compute_curves(
    predictions: list[dict],
    targets: list[dict],
    label_order: list[int] = LABEL_ORDER,
    iou_threshold: float = 0.5,
    n_thresholds: int = 101,
) -> dict:
    """Curvas P, R y F1 contra confianza, más la curva PR por clase."""
    match = match_dataset(predictions, targets, iou_threshold=iou_threshold)
    total_gt = sum(match.n_ground_truth.values())

    confidence = np.linspace(0.0, 1.0, n_thresholds)
    precision = np.zeros(n_thresholds)
    recall = np.zeros(n_thresholds)
    f1 = np.zeros(n_thresholds)

    for posicion, umbral in enumerate(confidence):
        seleccion = match.scores >= umbral
        verdaderos = int(match.true_positive[seleccion].sum())
        detectadas = int(seleccion.sum())

        p = verdaderos / detectadas if detectadas else 0.0
        r = verdaderos / total_gt if total_gt else 0.0

        precision[posicion] = p
        recall[posicion] = r
        f1[posicion] = 2 * p * r / (p + r) if (p + r) else 0.0

    mejor = int(np.argmax(f1))

    pr_per_class: dict[int, dict] = {}
    for label in label_order:
        de_la_clase = match.labels == label
        n_gt = match.n_ground_truth.get(label, 0)

        tp_acumulado = np.cumsum(match.true_positive[de_la_clase])
        cantidad = np.arange(1, tp_acumulado.size + 1)

        recall_clase = tp_acumulado / n_gt if n_gt else np.zeros_like(tp_acumulado, dtype=float)
        precision_clase = tp_acumulado / cantidad if cantidad.size else np.zeros(0)

        pr_per_class[label] = {
            "recall": np.asarray(recall_clase, dtype=float),
            "precision": np.asarray(precision_clase, dtype=float),
            "ap": _average_precision(
                np.asarray(recall_clase, dtype=float),
                np.asarray(precision_clase, dtype=float),
            )
            if n_gt
            else float("nan"),
        }

    return {
        "confidence": confidence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_confidence": float(confidence[mejor]),
        "best_f1": float(f1[mejor]),
        "best_precision": float(precision[mejor]),
        "best_recall": float(recall[mejor]),
        "pr_per_class": pr_per_class,
    }


@torch.no_grad()
def measure_inference_fps(model, dataset, device, num_images: int = 50, warmup: int = 5) -> float:
    """Imágenes por segundo en inferencia, de a una imagen por vez.

    Las primeras `warmup` pasadas se descartan: la primera inferencia paga la
    inicialización de kernels y no representa el régimen estacionario.
    """
    model.eval()
    model.to(device)

    total = min(num_images, len(dataset))
    if total == 0:
        return 0.0

    for indice in range(min(warmup, len(dataset))):
        model([dataset[indice][0].to(device)])

    if device.type == "cuda":
        torch.cuda.synchronize()

    inicio = time.perf_counter()
    for indice in range(total):
        model([dataset[indice][0].to(device)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    transcurrido = time.perf_counter() - inicio

    return total / transcurrido if transcurrido > 0 else 0.0
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/metrics.py tests/test_metrics.py
git commit -m "feat: metricas de deteccion con mAP COCO, curvas de confianza y FPS"
```

---

### Task 6: Bucle de entrenamiento y checkpoints

**Files:**
- Create: `src/engine/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `YoloDetectionDataset`, `collate_fn`; `build_fasterrcnn`.
- Produces:
  - `build_optimizer(model, training_config: dict) -> torch.optim.Optimizer`
  - `build_scheduler(optimizer, training_config: dict, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None`
  - `train_one_epoch(model, optimizer, loader, device, scaler=None, max_batches=None, log_every=50) -> dict[str, float]`
  - `save_checkpoint(path, model, optimizer, scheduler, epoch: int, history: list[dict]) -> None`
  - `load_checkpoint(path, model, optimizer=None, scheduler=None, device=None) -> tuple[int, list[dict]]` (devuelve la última época completada y el historial)

`training_config` acepta `optimizer` (`"sgd"` o `"adamw"`), `lr0`, `momentum`, `weight_decay` y `lr_scheduler` (`"cosine"`, `"step"` o `"none"`).

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_trainer.py`:

```python
"""Tests del bucle de entrenamiento y de la persistencia de checkpoints."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.yolo_dataset import YoloDetectionDataset, collate_fn
from src.engine.trainer import (
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from src.modeling.detectors import build_fasterrcnn

CONFIG_BASE = {
    "optimizer": "sgd",
    "lr0": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler": "cosine",
}


def _modelo():
    return build_fasterrcnn(
        num_classes=3, backbone="mobilenet_v3_large_fpn",
        min_size=64, max_size=128, pretrained=False,
    )


def test_build_optimizer_sgd_usa_los_hiperparametros():
    optimizer = build_optimizer(_modelo(), CONFIG_BASE)

    assert isinstance(optimizer, torch.optim.SGD)
    grupo = optimizer.param_groups[0]
    assert grupo["lr"] == 0.005
    assert grupo["momentum"] == 0.9
    assert grupo["weight_decay"] == 0.0005


def test_build_optimizer_adamw():
    optimizer = build_optimizer(_modelo(), {**CONFIG_BASE, "optimizer": "adamw"})
    assert isinstance(optimizer, torch.optim.AdamW)


def test_build_scheduler_cosine_baja_el_lr():
    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    scheduler = build_scheduler(optimizer, CONFIG_BASE, epochs=10)

    lr_inicial = optimizer.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] < lr_inicial


def test_build_scheduler_none_devuelve_none():
    optimizer = build_optimizer(_modelo(), CONFIG_BASE)
    assert build_scheduler(optimizer, {**CONFIG_BASE, "lr_scheduler": "none"}, 10) is None


def test_build_optimizer_rechaza_un_nombre_desconocido():
    with pytest.raises(ValueError, match="sgd"):
        build_optimizer(_modelo(), {**CONFIG_BASE, "optimizer": "inventado"})


def test_build_scheduler_rechaza_un_nombre_desconocido():
    optimizer = build_optimizer(_modelo(), CONFIG_BASE)
    with pytest.raises(ValueError, match="cosine"):
        build_scheduler(optimizer, {**CONFIG_BASE, "lr_scheduler": "inventado"}, 10)


def test_train_one_epoch_devuelve_las_perdidas_promedio(synthetic_dataset):
    dataset = YoloDetectionDataset(synthetic_dataset / "train", train=True, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)

    perdidas = train_one_epoch(
        modelo, optimizer, loader, torch.device("cpu"), max_batches=2
    )

    assert set(perdidas) == {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_total",
    }
    for valor in perdidas.values():
        assert isinstance(valor, float)
        assert valor == valor  # descarta NaN


def test_train_one_epoch_modifica_los_pesos(synthetic_dataset):
    dataset = YoloDetectionDataset(synthetic_dataset / "train", train=True, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    modelo = _modelo()
    antes = modelo.roi_heads.box_predictor.cls_score.weight.detach().clone()

    optimizer = build_optimizer(modelo, CONFIG_BASE)
    train_one_epoch(modelo, optimizer, loader, torch.device("cpu"), max_batches=2)

    despues = modelo.roi_heads.box_predictor.cls_score.weight.detach()
    assert not torch.allclose(antes, despues)


def test_checkpoint_ida_y_vuelta(tmp_path):
    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    scheduler = build_scheduler(optimizer, CONFIG_BASE, epochs=10)
    historial = [{"epoch": 1, "loss_total": 2.5}]

    ruta = tmp_path / "checkpoint.pth"
    save_checkpoint(ruta, modelo, optimizer, scheduler, epoch=1, history=historial)
    assert ruta.exists()

    modelo_nuevo = _modelo()
    optimizer_nuevo = build_optimizer(modelo_nuevo, CONFIG_BASE)
    scheduler_nuevo = build_scheduler(optimizer_nuevo, CONFIG_BASE, epochs=10)

    epoca, historial_recuperado = load_checkpoint(
        ruta, modelo_nuevo, optimizer_nuevo, scheduler_nuevo, torch.device("cpu")
    )

    assert epoca == 1
    assert historial_recuperado == historial
    assert torch.allclose(
        modelo.roi_heads.box_predictor.cls_score.weight,
        modelo_nuevo.roi_heads.box_predictor.cls_score.weight,
    )


def test_checkpoint_restaura_estado_ya_acumulado(synthetic_dataset, tmp_path):
    # El test anterior guarda un optimizador recién construido, cuyo estado está
    # vacío: pasaría igual si load_checkpoint no restaurara nada. Este entrena
    # primero para que haya momentum y el scheduler haya avanzado, que es lo que
    # de verdad tiene que sobrevivir cuando una corrida de 6 horas se reanuda.
    dataset = YoloDetectionDataset(synthetic_dataset / "train", train=True, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    scheduler = build_scheduler(optimizer, CONFIG_BASE, epochs=10)

    train_one_epoch(modelo, optimizer, loader, torch.device("cpu"), max_batches=2)
    scheduler.step()
    scheduler.step()

    estado_previo = optimizer.state_dict()["state"]
    assert estado_previo, "el optimizador debería haber acumulado momentum"
    lr_previo = optimizer.param_groups[0]["lr"]
    last_epoch_previo = scheduler.last_epoch

    ruta = tmp_path / "checkpoint.pth"
    save_checkpoint(ruta, modelo, optimizer, scheduler, epoch=1, history=[])

    modelo_nuevo = _modelo()
    optimizer_nuevo = build_optimizer(modelo_nuevo, CONFIG_BASE)
    scheduler_nuevo = build_scheduler(optimizer_nuevo, CONFIG_BASE, epochs=10)
    load_checkpoint(
        ruta, modelo_nuevo, optimizer_nuevo, scheduler_nuevo, torch.device("cpu")
    )

    assert scheduler_nuevo.last_epoch == last_epoch_previo
    assert optimizer_nuevo.param_groups[0]["lr"] == pytest.approx(lr_previo)

    estado_nuevo = optimizer_nuevo.state_dict()["state"]
    assert set(estado_nuevo) == set(estado_previo)
    for clave, valores in estado_previo.items():
        assert torch.allclose(
            valores["momentum_buffer"], estado_nuevo[clave]["momentum_buffer"]
        )


def test_load_checkpoint_sin_optimizador_tambien_funciona(tmp_path):
    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    ruta = tmp_path / "checkpoint.pth"
    save_checkpoint(ruta, modelo, optimizer, None, epoch=3, history=[])

    epoca, historial = load_checkpoint(ruta, _modelo())
    assert epoca == 3
    assert historial == []
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.engine.trainer'`.

- [ ] **Step 3: Implementar `src/engine/trainer.py`**

```python
"""Bucle de entrenamiento para detectores de torchvision.

Los checkpoints guardan modelo, optimizador, scheduler, época e historial. Un
entrenamiento de Faster R-CNN sobre D-Fire supera la duración de una sesión de
Colab, así que reanudar tiene que ser el caso normal, no la excepción.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


def build_optimizer(model, training_config: dict) -> torch.optim.Optimizer:
    nombre = str(training_config.get("optimizer", "sgd")).lower()
    parametros = [p for p in model.parameters() if p.requires_grad]

    if nombre == "sgd":
        return torch.optim.SGD(
            parametros,
            lr=training_config["lr0"],
            momentum=training_config.get("momentum", 0.9),
            weight_decay=training_config.get("weight_decay", 0.0005),
        )
    if nombre == "adamw":
        return torch.optim.AdamW(
            parametros,
            lr=training_config["lr0"],
            weight_decay=training_config.get("weight_decay", 0.0005),
        )
    raise ValueError(f"Optimizador desconocido: {nombre!r}. Usar 'sgd' o 'adamw'.")


def build_scheduler(optimizer, training_config: dict, epochs: int):
    nombre = str(training_config.get("lr_scheduler", "cosine")).lower()

    if nombre == "none":
        return None
    if nombre == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if nombre == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.1
        )
    raise ValueError(
        f"Scheduler desconocido: {nombre!r}. Usar 'cosine', 'step' o 'none'."
    )


def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    scaler=None,
    max_batches: int | None = None,
    log_every: int = 50,
) -> dict[str, float]:
    """Entrena una época y devuelve el promedio de cada componente de la pérdida."""
    model.train()
    model.to(device)

    acumulado = {clave: 0.0 for clave in LOSS_KEYS}
    acumulado["loss_total"] = 0.0
    batches = 0

    for indice, (images, targets) in enumerate(loader):
        if max_batches is not None and indice >= max_batches:
            break

        images = [image.to(device) for image in images]
        targets = [
            {clave: valor.to(device) for clave, valor in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=scaler is not None and device.type == "cuda"
        ):
            perdidas = model(images, targets)
            total = sum(perdidas.values())

        if not math.isfinite(float(total.detach())):
            # Un batch con pérdida infinita corrompe los pesos; se saltea.
            print(f"[aviso] batch {indice} con pérdida no finita, se omite")
            continue

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        for clave in LOSS_KEYS:
            acumulado[clave] += float(perdidas[clave].detach())
        acumulado["loss_total"] += float(total.detach())
        batches += 1

        if log_every and indice % log_every == 0:
            print(f"  batch {indice}: loss={float(total.detach()):.4f}")

    if batches == 0:
        return {clave: float("nan") for clave in acumulado}
    return {clave: valor / batches for clave, valor in acumulado.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, history: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    """Restaura el estado y devuelve `(ultima_epoca_completada, historial)`."""
    checkpoint = torch.load(
        Path(path), map_location=device or torch.device("cpu"), weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint["epoch"]), list(checkpoint.get("history", []))
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/trainer.py tests/test_trainer.py
git commit -m "feat: bucle de entrenamiento con checkpoints reanudables"
```

---

### Task 7: Esquema y escritura de `metrics_summary.csv`

**Files:**
- Create: `src/reporting/summary.py`
- Test: `tests/test_summary.py`

**Interfaces:**
- Consumes: nada de tareas anteriores.
- Produces:
  - `METRICS_SUMMARY_COLUMNS: list[str]` — exactamente las 20 columnas del spec, en ese orden.
  - `write_metrics_summary(path, metrics: dict) -> pandas.DataFrame` — falla con `ValueError` si sobra o falta alguna columna.
  - `load_metrics_summaries(results_root) -> pandas.DataFrame` — recorre `<results_root>/*/metrics_summary.csv`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_summary.py`:

```python
"""Tests del esquema común de métricas entre modelos."""

import pandas as pd
import pytest

from src.reporting.summary import (
    METRICS_SUMMARY_COLUMNS,
    load_metrics_summaries,
    write_metrics_summary,
)


def _metricas(nombre="fasterrcnn_r50fpn", map50=0.80):
    return {
        "experiment": nombre,
        "family": "Faster R-CNN",
        "model": "fasterrcnn_resnet50_fpn_v2",
        "params_M": 43.3,
        "epochs": 12,
        "imgsz": 640,
        "batch": 4,
        "train_time_min": 320.5,
        "mAP50": map50,
        "mAP50_95": 0.45,
        "precision": 0.78,
        "recall": 0.71,
        "f1": 0.74,
        "mAP50_smoke": 0.77,
        "mAP50_fire": 0.83,
        "mAP50_95_smoke": 0.41,
        "mAP50_95_fire": 0.49,
        "fps": 12.4,
        "device": "Tesla T4",
        "split": "val",
    }


def test_las_columnas_son_las_del_spec():
    assert METRICS_SUMMARY_COLUMNS == [
        "experiment", "family", "model", "params_M", "epochs", "imgsz", "batch",
        "train_time_min", "mAP50", "mAP50_95", "precision", "recall", "f1",
        "mAP50_smoke", "mAP50_fire", "mAP50_95_smoke", "mAP50_95_fire",
        "fps", "device", "split",
    ]


def test_escribe_una_sola_fila_con_las_columnas_en_orden(tmp_path):
    ruta = tmp_path / "metrics_summary.csv"
    write_metrics_summary(ruta, _metricas())

    df = pd.read_csv(ruta)
    assert len(df) == 1
    assert list(df.columns) == METRICS_SUMMARY_COLUMNS
    assert df.loc[0, "experiment"] == "fasterrcnn_r50fpn"
    assert df.loc[0, "mAP50"] == 0.80


def test_falla_si_falta_una_columna(tmp_path):
    metricas = _metricas()
    del metricas["fps"]

    with pytest.raises(ValueError, match="fps"):
        write_metrics_summary(tmp_path / "metrics_summary.csv", metricas)


def test_falla_si_sobra_una_columna(tmp_path):
    metricas = _metricas()
    metricas["columna_inventada"] = 1

    with pytest.raises(ValueError, match="columna_inventada"):
        write_metrics_summary(tmp_path / "metrics_summary.csv", metricas)


def test_load_metrics_summaries_junta_los_experimentos(tmp_path):
    for nombre, map50 in [("yolov8n_baseline", 0.75), ("fasterrcnn_r50fpn", 0.80)]:
        carpeta = tmp_path / nombre
        carpeta.mkdir()
        write_metrics_summary(carpeta / "metrics_summary.csv", _metricas(nombre, map50))

    df = load_metrics_summaries(tmp_path)

    assert len(df) == 2
    assert set(df["experiment"]) == {"yolov8n_baseline", "fasterrcnn_r50fpn"}
    # Ordenado por mAP50 descendente.
    assert df.iloc[0]["experiment"] == "fasterrcnn_r50fpn"


def test_load_metrics_summaries_ignora_carpetas_sin_el_archivo(tmp_path):
    (tmp_path / "eda").mkdir()
    (tmp_path / "eda" / "split_summary.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    carpeta = tmp_path / "yolov8n_baseline"
    carpeta.mkdir()
    write_metrics_summary(carpeta / "metrics_summary.csv", _metricas("yolov8n_baseline"))

    df = load_metrics_summaries(tmp_path)
    assert len(df) == 1


def test_load_metrics_summaries_sin_resultados_devuelve_df_vacio_con_columnas(tmp_path):
    df = load_metrics_summaries(tmp_path)
    assert df.empty
    assert list(df.columns) == METRICS_SUMMARY_COLUMNS
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_summary.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.reporting.summary'`.

- [ ] **Step 3: Implementar `src/reporting/summary.py`**

```python
"""Esquema común de métricas, para comparar modelos de librerías distintas.

El `results.csv` de Ultralytics trae columnas propias de esa librería, así que
no sirve para comparar contra torchvision. `metrics_summary.csv` es el contrato:
una fila por experimento, siempre las mismas columnas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRICS_SUMMARY_COLUMNS = [
    "experiment",
    "family",
    "model",
    "params_M",
    "epochs",
    "imgsz",
    "batch",
    "train_time_min",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "f1",
    "mAP50_smoke",
    "mAP50_fire",
    "mAP50_95_smoke",
    "mAP50_95_fire",
    "fps",
    "device",
    "split",
]

METRICS_SUMMARY_FILENAME = "metrics_summary.csv"


def write_metrics_summary(path, metrics: dict) -> pd.DataFrame:
    """Escribe la fila de resumen, validando el esquema antes de tocar el disco."""
    recibidas = set(metrics)
    esperadas = set(METRICS_SUMMARY_COLUMNS)

    faltantes = sorted(esperadas - recibidas)
    sobrantes = sorted(recibidas - esperadas)

    if faltantes or sobrantes:
        detalle = []
        if faltantes:
            detalle.append(f"faltan: {', '.join(faltantes)}")
        if sobrantes:
            detalle.append(f"sobran: {', '.join(sobrantes)}")
        raise ValueError(f"Esquema inválido para metrics_summary.csv ({'; '.join(detalle)})")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([metrics])[METRICS_SUMMARY_COLUMNS]
    df.to_csv(path, index=False)
    return df


def load_metrics_summaries(results_root) -> pd.DataFrame:
    """Junta los resúmenes de todos los experimentos, ordenados por mAP50."""
    rutas = sorted(Path(results_root).glob(f"*/{METRICS_SUMMARY_FILENAME}"))

    if not rutas:
        return pd.DataFrame(columns=METRICS_SUMMARY_COLUMNS)

    df = pd.concat([pd.read_csv(ruta) for ruta in rutas], ignore_index=True)
    return df.sort_values("mAP50", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_summary.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/reporting/summary.py tests/test_summary.py
git commit -m "feat: esquema comun de metrics_summary.csv"
```

---

### Task 8: Figuras

**Files:**
- Create: `src/reporting/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `CLASS_NAMES`, `LABEL_ORDER`; `compute_curves` (para la forma de los datos).
- Produces:
  - `plot_results_csv(results_csv, out_png) -> Path`
  - `plot_confusion_matrix(matrix, class_names: list[str], out_png, normalize: bool = False) -> Path`
  - `plot_pr_curve(curves: dict, out_png, class_names: dict[int, str] = CLASS_NAMES) -> Path`
  - `plot_metric_vs_confidence(curves: dict, metric: str, out_png, class_names=CLASS_NAMES) -> Path` con `metric` en `{"precision", "recall", "f1"}`
  - `plot_model_comparison(df, out_dir) -> list[Path]` (tres figuras: `map_por_modelo.png`, `map_por_clase.png`, `precision_vs_velocidad.png`)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_plots.py`:

```python
"""Tests de generación de figuras. Verifican que los PNG se crean y no están vacíos."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.plots import (
    mean_average_precision,
    normalize_by_column,
    plot_confusion_matrix,
    plot_metric_vs_confidence,
    plot_model_comparison,
    plot_pr_curve,
    plot_results_csv,
)


def test_normalize_by_column_normaliza_columnas_y_no_filas():
    # La matriz es asimétrica a propósito: con una simétrica, normalizar por
    # filas daría el mismo resultado y el test no detectaría el intercambio.
    normalizada = normalize_by_column(np.array([[2, 0], [2, 4]]))

    assert np.allclose(normalizada, [[0.5, 0.0], [0.5, 1.0]])
    assert np.allclose(normalizada.sum(axis=0), 1.0)


def test_normalize_by_column_deja_en_cero_la_columna_vacia():
    # La columna de fondo suele sumar cero; dividir daría NaN y un RuntimeWarning.
    normalizada = normalize_by_column(np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0]]))

    assert np.allclose(normalizada[:, 2], 0.0)
    assert not np.isnan(normalizada).any()


def test_mean_average_precision_ignora_las_clases_sin_ground_truth():
    pr_per_class = {1: {"ap": 0.8}, 2: {"ap": float("nan")}}

    assert mean_average_precision(pr_per_class) == pytest.approx(0.8)


def test_mean_average_precision_sin_clases_medibles_es_nan():
    assert np.isnan(mean_average_precision({1: {"ap": float("nan")}}))


def _curvas():
    confianza = np.linspace(0.0, 1.0, 101)
    return {
        "confidence": confianza,
        "precision": np.clip(confianza, 0, 1),
        "recall": np.clip(1 - confianza, 0, 1),
        "f1": np.full(101, 0.5),
        "best_confidence": 0.5,
        "best_f1": 0.5,
        "best_precision": 0.5,
        "best_recall": 0.5,
        "pr_per_class": {
            1: {
                "recall": np.linspace(0, 1, 50),
                "precision": np.linspace(1, 0.5, 50),
                "ap": 0.75,
            },
            2: {
                "recall": np.linspace(0, 1, 50),
                "precision": np.linspace(1, 0.4, 50),
                "ap": 0.70,
            },
        },
    }


def _resumen():
    filas = []
    for nombre, familia, map50, map95, fps in [
        ("yolov8n_baseline", "YOLOv8", 0.75, 0.43, 90.0),
        ("fasterrcnn_r50fpn", "Faster R-CNN", 0.80, 0.47, 12.0),
        ("rtdetr_l", "RT-DETR", 0.82, 0.50, 35.0),
    ]:
        filas.append(
            {
                "experiment": nombre, "family": familia, "model": nombre,
                "params_M": 10.0, "epochs": 12, "imgsz": 640, "batch": 4,
                "train_time_min": 100.0, "mAP50": map50, "mAP50_95": map95,
                "precision": 0.8, "recall": 0.7, "f1": 0.75,
                "mAP50_smoke": map50 - 0.02, "mAP50_fire": map50 + 0.02,
                "mAP50_95_smoke": map95 - 0.02, "mAP50_95_fire": map95 + 0.02,
                "fps": fps, "device": "cpu", "split": "val",
            }
        )
    return pd.DataFrame(filas)


def _no_esta_vacio(ruta):
    assert ruta.exists(), f"No se creó {ruta}"
    assert ruta.stat().st_size > 1000, f"{ruta} parece vacío"


def test_plot_results_csv(tmp_path):
    csv = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train/loss_total": [2.5, 1.8, 1.4],
            "train/loss_classifier": [1.0, 0.7, 0.5],
            "train/loss_box_reg": [0.8, 0.6, 0.5],
            "train/loss_objectness": [0.4, 0.3, 0.2],
            "train/loss_rpn_box_reg": [0.3, 0.2, 0.2],
            "metrics/mAP50": [0.3, 0.5, 0.6],
            "metrics/mAP50-95": [0.1, 0.2, 0.3],
            "metrics/precision": [0.4, 0.6, 0.7],
            "metrics/recall": [0.3, 0.5, 0.6],
            "lr": [0.005, 0.004, 0.003],
        }
    ).to_csv(csv, index=False)

    _no_esta_vacio(plot_results_csv(csv, tmp_path / "results.png"))


def test_plot_confusion_matrix_cruda_y_normalizada(tmp_path):
    matriz = np.array([[50, 5, 10], [4, 60, 8], [12, 9, 0]])

    _no_esta_vacio(
        plot_confusion_matrix(matriz, ["smoke", "fire", "background"], tmp_path / "cm.png")
    )
    _no_esta_vacio(
        plot_confusion_matrix(
            matriz, ["smoke", "fire", "background"], tmp_path / "cmn.png", normalize=True
        )
    )


def test_plot_confusion_matrix_normalizada_tolera_columna_en_cero(tmp_path):
    # La columna de fondo suele sumar cero: no debe producir una división por cero.
    matriz = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0]])
    _no_esta_vacio(
        plot_confusion_matrix(
            matriz, ["smoke", "fire", "background"], tmp_path / "cmn0.png", normalize=True
        )
    )


def test_plot_pr_curve(tmp_path):
    _no_esta_vacio(plot_pr_curve(_curvas(), tmp_path / "PR_curve.png"))


@pytest.mark.parametrize("metrica", ["precision", "recall", "f1"])
def test_plot_metric_vs_confidence(tmp_path, metrica):
    _no_esta_vacio(
        plot_metric_vs_confidence(_curvas(), metrica, tmp_path / f"{metrica}.png")
    )


def test_plot_metric_vs_confidence_rechaza_metrica_desconocida(tmp_path):
    with pytest.raises(ValueError, match="inventada"):
        plot_metric_vs_confidence(_curvas(), "inventada", tmp_path / "x.png")


def test_plot_model_comparison_genera_las_tres_figuras(tmp_path):
    rutas = plot_model_comparison(_resumen(), tmp_path)

    assert len(rutas) == 3
    nombres = {ruta.name for ruta in rutas}
    assert nombres == {
        "map_por_modelo.png",
        "map_por_clase.png",
        "precision_vs_velocidad.png",
    }
    for ruta in rutas:
        _no_esta_vacio(ruta)
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_plots.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.reporting.plots'`.

- [ ] **Step 3: Implementar `src/reporting/plots.py`**

```python
"""Figuras de los experimentos, con los mismos nombres de archivo que Ultralytics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend sin ventana: los notebooks corren en Colab y los tests sin display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.yolo_dataset import CLASS_NAMES  # noqa: E402

METRIC_LABELS = {
    "precision": "Precisión",
    "recall": "Recall",
    "f1": "F1",
}


def _guardar(fig, out_png) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def normalize_by_column(matrix) -> np.ndarray:
    """Normaliza cada columna de la matriz de confusión a suma 1.

    Es una función aparte y no código embebido en el gráfico porque es la única
    parte con aritmética: así se puede testear directamente. Una columna que
    suma cero —el caso normal de la columna de fondo— queda en cero, no en NaN.
    """
    matriz = np.asarray(matrix, dtype=float)
    sumas = matriz.sum(axis=0, keepdims=True)
    return np.divide(matriz, sumas, out=np.zeros_like(matriz), where=sumas > 0)


def mean_average_precision(pr_per_class: dict) -> float:
    """Promedia los AP por clase ignorando las clases sin ningún objeto real.

    `compute_map` devuelve NaN para esas clases; incluirlas en el promedio lo
    volvería NaN entero y el número desaparecería del gráfico sin explicación.
    """
    aps = [
        datos["ap"] for datos in pr_per_class.values() if not np.isnan(datos["ap"])
    ]
    return float(np.mean(aps)) if aps else float("nan")


def plot_results_csv(results_csv, out_png) -> Path:
    """Panel con las pérdidas de entrenamiento y las métricas de validación."""
    df = pd.read_csv(results_csv)
    columnas = [columna for columna in df.columns if columna != "epoch"]

    filas = 2
    columnas_grafico = int(np.ceil(len(columnas) / filas))
    fig, axes = plt.subplots(
        filas, columnas_grafico, figsize=(3.2 * columnas_grafico, 6), squeeze=False
    )

    for posicion, columna in enumerate(columnas):
        ax = axes[posicion // columnas_grafico][posicion % columnas_grafico]
        ax.plot(df["epoch"], df[columna], marker="o", markersize=3, linewidth=1.5)
        ax.set_title(columna, fontsize=9)
        ax.set_xlabel("época", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for posicion in range(len(columnas), filas * columnas_grafico):
        axes[posicion // columnas_grafico][posicion % columnas_grafico].axis("off")

    fig.suptitle("Entrenamiento y validación por época", fontsize=11)
    return _guardar(fig, out_png)


def plot_confusion_matrix(matrix, class_names, out_png, normalize: bool = False) -> Path:
    """Matriz `[predicho, real]`; normalizada, cada columna suma 1."""
    matriz = normalize_by_column(matrix) if normalize else np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    imagen = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.set_title("Matriz de confusión" + (" normalizada" if normalize else ""))

    umbral = matriz.max() / 2 if matriz.max() > 0 else 0.5
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            valor = matriz[fila, columna]
            ax.text(
                columna, fila,
                f"{valor:.2f}" if normalize else f"{int(valor)}",
                ha="center", va="center", fontsize=9,
                color="white" if valor > umbral else "black",
            )

    return _guardar(fig, out_png)


def plot_pr_curve(curves: dict, out_png, class_names: dict = CLASS_NAMES) -> Path:
    """Curva precisión-recall por clase, con el AP en la leyenda."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, datos in curves["pr_per_class"].items():
        nombre = class_names.get(label, str(label))
        ax.plot(
            datos["recall"],
            datos["precision"],
            linewidth=1.8,
            label=f"{nombre} (AP={datos['ap']:.3f})",
        )

    promedio = mean_average_precision(curves["pr_per_class"])
    if not np.isnan(promedio):
        ax.plot([], [], " ", label=f"mAP@0.5 = {promedio:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Curva precisión-recall (IoU 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    return _guardar(fig, out_png)


def plot_metric_vs_confidence(
    curves: dict, metric: str, out_png, class_names: dict = CLASS_NAMES
) -> Path:
    """Precisión, recall o F1 en función del umbral de confianza."""
    if metric not in METRIC_LABELS:
        raise ValueError(
            f"Métrica desconocida: {metric!r}. Opciones: {sorted(METRIC_LABELS)}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curves["confidence"], curves[metric], linewidth=2, color="tab:blue")

    if metric == "f1":
        ax.axvline(
            curves["best_confidence"], linestyle="--", color="tab:red", linewidth=1.2,
            label=f"mejor F1 = {curves['best_f1']:.3f} @ conf {curves['best_confidence']:.2f}",
        )
        ax.legend(loc="lower center", fontsize=9)

    ax.set_xlabel("Confianza")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{METRIC_LABELS[metric]} en función de la confianza")
    ax.grid(alpha=0.3)

    return _guardar(fig, out_png)


def plot_model_comparison(df: pd.DataFrame, out_dir) -> list[Path]:
    """Tres figuras comparativas a partir de los metrics_summary.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []

    etiquetas = df["experiment"].tolist()
    posiciones = np.arange(len(df))
    ancho = 0.38

    # 1) mAP50 y mAP50-95 por modelo
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(posiciones - ancho / 2, df["mAP50"], ancho, label="mAP@0.5")
    ax.bar(posiciones + ancho / 2, df["mAP50_95"], ancho, label="mAP@0.5:0.95")
    for posicion, (a, b) in enumerate(zip(df["mAP50"], df["mAP50_95"])):
        ax.text(posicion - ancho / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(posicion + ancho / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(posiciones, etiquetas, rotation=15, ha="right")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.05)
    ax.set_title("Desempeño por modelo (split de validación)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    rutas.append(_guardar(fig, out_dir / "map_por_modelo.png"))

    # 2) mAP50 por clase
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(posiciones - ancho / 2, df["mAP50_smoke"], ancho, label="smoke")
    ax.bar(posiciones + ancho / 2, df["mAP50_fire"], ancho, label="fire")
    ax.set_xticks(posiciones, etiquetas, rotation=15, ha="right")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("mAP@0.5 por clase")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    rutas.append(_guardar(fig, out_dir / "map_por_clase.png"))

    # 3) precisión contra velocidad
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["fps"], df["mAP50_95"], s=140, c=posiciones, cmap="viridis", zorder=3)
    for _, fila in df.iterrows():
        ax.annotate(
            fila["experiment"],
            (fila["fps"], fila["mAP50_95"]),
            textcoords="offset points", xytext=(8, 6), fontsize=9,
        )
    ax.set_xlabel("FPS en inferencia")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Compromiso entre precisión y velocidad")
    ax.grid(alpha=0.3)
    rutas.append(_guardar(fig, out_dir / "precision_vs_velocidad.png"))

    return rutas
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_plots.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/reporting/plots.py tests/test_plots.py
git commit -m "feat: figuras de resultados y de comparacion entre modelos"
```

---

### Task 9: Orquestador de reporte y configuraciones de experimento

**Files:**
- Create: `src/reporting/experiment_report.py`
- Create: `configs/experiments/fasterrcnn_r50fpn.yaml`
- Create: `configs/experiments/rtdetr_l.yaml`
- Test: `tests/test_experiment_report.py`

**Interfaces:**
- Consumes: `collect_predictions`, `compute_map`, `compute_curves`, `measure_inference_fps`; `confusion_matrix`; `count_parameters`; todo `plots.py`; `write_metrics_summary`; `LABEL_ORDER`, `CLASS_NAMES`.
- Produces:
  - `write_history_csv(history: list[dict], out_csv) -> Path`
  - `generate_experiment_report(model, val_loader, val_dataset, config: dict, history: list[dict], out_dir, device, train_time_min: float, device_name: str) -> dict` — devuelve el dict de métricas que se escribió en `metrics_summary.csv` y deja en `out_dir` los 10 archivos del spec.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_experiment_report.py`:

```python
"""Test de integración del reporte completo, en CPU y con un modelo sin entrenar."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.yolo_dataset import YoloDetectionDataset, collate_fn
from src.modeling.detectors import build_fasterrcnn
from src.reporting.experiment_report import (
    build_metrics_row,
    generate_experiment_report,
    write_history_csv,
)
from src.reporting.summary import METRICS_SUMMARY_COLUMNS

CONFIG = {
    "experiment": {
        "name": "fasterrcnn_test",
        "family": "Faster R-CNN",
        "model": "fasterrcnn_mobilenet_v3_large_fpn",
    },
    "training": {"epochs": 2, "imgsz": 64, "batch": 2},
}

HISTORIAL = [
    {
        "epoch": 1, "train/loss_total": 2.5, "train/loss_classifier": 1.0,
        "train/loss_box_reg": 0.8, "train/loss_objectness": 0.4,
        "train/loss_rpn_box_reg": 0.3, "metrics/mAP50": 0.10,
        "metrics/mAP50-95": 0.04, "metrics/precision": 0.2,
        "metrics/recall": 0.15, "lr": 0.005,
    },
    {
        "epoch": 2, "train/loss_total": 1.9, "train/loss_classifier": 0.7,
        "train/loss_box_reg": 0.6, "train/loss_objectness": 0.3,
        "train/loss_rpn_box_reg": 0.3, "metrics/mAP50": 0.18,
        "metrics/mAP50-95": 0.07, "metrics/precision": 0.3,
        "metrics/recall": 0.22, "lr": 0.003,
    },
]


def test_write_history_csv(tmp_path):
    ruta = write_history_csv(HISTORIAL, tmp_path / "results.csv")
    df = pd.read_csv(ruta)

    assert len(df) == 2
    assert df.columns[0] == "epoch"
    assert "metrics/mAP50" in df.columns


def test_build_metrics_row_asigna_cada_clase_a_su_columna():
    # Los cuatro valores por clase son distintos a propósito: si smoke y fire se
    # intercambiaran, el informe saldría plausible pero atribuiría el desempeño
    # de una clase a la otra, y con el modelo sin entrenar del test de
    # integración los dos números son casi iguales y no se notaría.
    fila = build_metrics_row(
        config=CONFIG,
        map_metrics={
            "map50": 0.70,
            "map50_95": 0.40,
            "map50_per_class": {1: 0.11, 2: 0.22},
            "map50_95_per_class": {1: 0.33, 2: 0.44},
        },
        curves={"best_precision": 0.60, "best_recall": 0.50, "best_f1": 0.55},
        params_M=3.2,
        fps=12.5,
        train_time_min=1.5,
        device_name="cpu",
    )

    assert fila["mAP50_smoke"] == 0.11
    assert fila["mAP50_fire"] == 0.22
    assert fila["mAP50_95_smoke"] == 0.33
    assert fila["mAP50_95_fire"] == 0.44
    assert set(fila) == set(METRICS_SUMMARY_COLUMNS)


def test_genera_todos_los_artefactos_del_spec(synthetic_dataset, tmp_path):
    dataset = YoloDetectionDataset(synthetic_dataset / "val")
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    modelo = build_fasterrcnn(
        num_classes=3, backbone="mobilenet_v3_large_fpn",
        min_size=64, max_size=128, pretrained=False,
    )

    metricas = generate_experiment_report(
        model=modelo,
        val_loader=loader,
        val_dataset=dataset,
        config=CONFIG,
        history=HISTORIAL,
        out_dir=tmp_path,
        device=torch.device("cpu"),
        train_time_min=1.5,
        device_name="cpu",
    )

    esperados = [
        "results.csv", "results.png",
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
        "experiment_config_used.yaml", "metrics_summary.csv",
    ]
    for nombre in esperados:
        ruta = tmp_path / nombre
        assert ruta.exists(), f"Falta {nombre}"
        assert ruta.stat().st_size > 0

    assert set(metricas) == set(METRICS_SUMMARY_COLUMNS)
    assert metricas["experiment"] == "fasterrcnn_test"
    assert metricas["family"] == "Faster R-CNN"
    assert metricas["split"] == "val"
    assert metricas["train_time_min"] == 1.5
    assert metricas["params_M"] > 0

    df = pd.read_csv(tmp_path / "metrics_summary.csv")
    assert list(df.columns) == METRICS_SUMMARY_COLUMNS
    assert len(df) == 1
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_experiment_report.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.reporting.experiment_report'`.

- [ ] **Step 3: Implementar `src/reporting/experiment_report.py`**

```python
"""De un modelo entrenado a una carpeta de resultados completa.

Concentra acá todo el reporte para que los notebooks queden finos y para que
Faster R-CNN produzca exactamente los mismos archivos que genera Ultralytics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.yolo_dataset import CLASS_NAMES, LABEL_ORDER
from src.engine.matching import confusion_matrix
from src.engine.metrics import (
    collect_predictions,
    compute_curves,
    compute_map,
    measure_inference_fps,
)
from src.modeling.detectors import count_parameters
from src.reporting.plots import (
    plot_confusion_matrix,
    plot_metric_vs_confidence,
    plot_pr_curve,
    plot_results_csv,
)
from src.reporting.summary import write_metrics_summary

CONFUSION_CONF_THRESHOLD = 0.25
CONFUSION_IOU_THRESHOLD = 0.45
CURVES_IOU_THRESHOLD = 0.5

# Las columnas del CSV llevan el nombre de la clase, así que la etiqueta se
# resuelve por nombre y no por posición: desacoplar esto de LABEL_ORDER evita
# que un reordenamiento futuro intercambie las métricas de smoke y fire.
LABEL_BY_NAME = {nombre: label for label, nombre in CLASS_NAMES.items()}


def build_metrics_row(
    config: dict,
    map_metrics: dict,
    curves: dict,
    params_M: float,
    fps: float,
    train_time_min: float,
    device_name: str,
) -> dict:
    """Arma la fila única de `metrics_summary.csv`.

    Está separada del reporte para poder testearla: un intercambio entre las
    columnas de smoke y fire produciría un informe plausible pero equivocado, y
    el test de integración usa un modelo sin entrenar cuyas métricas por clase
    son casi idénticas, así que no lo detectaría.
    """
    experiment = config["experiment"]
    training = config["training"]

    label_smoke = LABEL_BY_NAME["smoke"]
    label_fire = LABEL_BY_NAME["fire"]

    return {
        "experiment": experiment["name"],
        "family": experiment["family"],
        "model": experiment["model"],
        "params_M": round(params_M, 2),
        "epochs": training["epochs"],
        "imgsz": training["imgsz"],
        "batch": training["batch"],
        "train_time_min": round(float(train_time_min), 2),
        "mAP50": round(map_metrics["map50"], 4),
        "mAP50_95": round(map_metrics["map50_95"], 4),
        "precision": round(curves["best_precision"], 4),
        "recall": round(curves["best_recall"], 4),
        "f1": round(curves["best_f1"], 4),
        "mAP50_smoke": round(map_metrics["map50_per_class"][label_smoke], 4),
        "mAP50_fire": round(map_metrics["map50_per_class"][label_fire], 4),
        "mAP50_95_smoke": round(map_metrics["map50_95_per_class"][label_smoke], 4),
        "mAP50_95_fire": round(map_metrics["map50_95_per_class"][label_fire], 4),
        "fps": round(fps, 2),
        "device": device_name,
        "split": "val",
    }


def write_history_csv(history: list[dict], out_csv) -> Path:
    """Historial por época, con el mismo espíritu que el results.csv de Ultralytics."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_csv, index=False)
    return out_csv


def generate_experiment_report(
    model,
    val_loader,
    val_dataset,
    config: dict,
    history: list[dict],
    out_dir,
    device,
    train_time_min: float,
    device_name: str,
) -> dict:
    """Evalúa sobre validación y escribe los diez artefactos del experimento."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Recolectando predicciones sobre validación...")
    predictions, targets = collect_predictions(model, val_loader, device)

    print("Calculando mAP...")
    map_metrics = compute_map(predictions, targets, LABEL_ORDER)

    print("Calculando curvas de confianza...")
    curves = compute_curves(
        predictions, targets, LABEL_ORDER, iou_threshold=CURVES_IOU_THRESHOLD
    )

    print("Midiendo velocidad de inferencia...")
    fps = measure_inference_fps(model, val_dataset, device)

    matriz = confusion_matrix(
        predictions,
        targets,
        LABEL_ORDER,
        conf_threshold=CONFUSION_CONF_THRESHOLD,
        iou_threshold=CONFUSION_IOU_THRESHOLD,
    )
    nombres_con_fondo = [CLASS_NAMES[label] for label in LABEL_ORDER] + ["background"]

    write_history_csv(history, out_dir / "results.csv")
    plot_results_csv(out_dir / "results.csv", out_dir / "results.png")
    plot_confusion_matrix(matriz, nombres_con_fondo, out_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        matriz, nombres_con_fondo, out_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_pr_curve(curves, out_dir / "PR_curve.png")
    plot_metric_vs_confidence(curves, "f1", out_dir / "F1_curve.png")
    plot_metric_vs_confidence(curves, "precision", out_dir / "P_curve.png")
    plot_metric_vs_confidence(curves, "recall", out_dir / "R_curve.png")

    with open(out_dir / "experiment_config_used.yaml", "w", encoding="utf-8") as archivo:
        yaml.safe_dump(config, archivo, sort_keys=False, allow_unicode=True)

    metrics = build_metrics_row(
        config=config,
        map_metrics=map_metrics,
        curves=curves,
        params_M=count_parameters(model) / 1e6,
        fps=fps,
        train_time_min=train_time_min,
        device_name=device_name,
    )

    write_metrics_summary(out_dir / "metrics_summary.csv", metrics)
    print(f"Reporte completo en: {out_dir}")
    return metrics
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/test_experiment_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Crear `configs/experiments/fasterrcnn_r50fpn.yaml`**

```yaml
experiment:
  name: fasterrcnn_r50fpn
  family: Faster R-CNN
  model: fasterrcnn_resnet50_fpn_v2
  description: Detector de dos etapas con ResNet50-FPN v2 preentrenado en COCO.

training:
  epochs: 12
  imgsz: 640
  max_size: 1024
  batch: 4
  workers: 2
  optimizer: sgd
  lr0: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  lr_scheduler: cosine
  backbone: resnet50_fpn_v2
  trainable_backbone_layers: 3
  amp: true
  seed: 42

output:
  project: /content/drive/MyDrive/VCII_DFire/runs
  save_weights: true
```

- [ ] **Step 6: Crear `configs/experiments/rtdetr_l.yaml`**

```yaml
experiment:
  name: rtdetr_l
  family: RT-DETR
  model: rtdetr-l.pt
  description: Detector transformer en tiempo real, variante large.

training:
  epochs: 20
  imgsz: 640
  batch: 8
  patience: 10
  optimizer: auto
  lr0: 0.0001
  seed: 42

output:
  project: /content/drive/MyDrive/VCII_DFire/runs
  save_weights: true
```

- [ ] **Step 7: Correr toda la batería de tests**

Run: `.venv/bin/python -m pytest -v`
Expected: 66 passed (2 + 10 + 7 + 11 + 10 + 8 + 7 + 9 + 2).

- [ ] **Step 8: Commit**

```bash
git add src/reporting/experiment_report.py tests/test_experiment_report.py configs/experiments
git commit -m "feat: orquestador de reporte y configs de Faster R-CNN y RT-DETR"
```

---

### Task 10: Notebook 03 — entrenamiento de Faster R-CNN

**Files:**
- Create: `notebooks/03_entrenamiento_FasterRCNN.ipynb`

**Interfaces:**
- Consumes: todo `src/`, `configs/experiments/fasterrcnn_r50fpn.yaml`.
- Produces: `reports/results/fasterrcnn_r50fpn/` con los diez artefactos.

El notebook replica el esqueleto del `02_entrenamiento_YOLO.ipynb` (setup, GPU, Drive, clonado, kagglehub, config) y cambia solo la parte de entrenamiento.

- [ ] **Step 1: Escribir el script generador del notebook**

Crear `/private/tmp/claude-501/.../scratchpad/build_nb03.py` (fuera del repo) con el helper reusable:

```python
"""Genera notebooks/03_entrenamiento_FasterRCNN.ipynb."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
CELDAS = []  # se completa en el Step 2


def md(texto):
    CELDAS.append(nbf.v4.new_markdown_cell(texto))


def code(texto):
    CELDAS.append(nbf.v4.new_code_cell(texto))
```

- [ ] **Step 2: Completar el script con las celdas del notebook**

Agregar al script, después de las funciones `md` y `code`, las llamadas que definen las celdas:

```python
md("""# 03 - Entrenamiento de Faster R-CNN para detección de humo y fuego

Detector de dos etapas (`fasterrcnn_resnet50_fpn_v2` de torchvision) preentrenado
en COCO, con el cabezal ajustado a las clases `smoke` y `fire`.

El código reusable vive en `src/`; este notebook solo arma la configuración,
corre el bucle de épocas y delega el reporte.

## Salidas esperadas

En `reports/results/fasterrcnn_r50fpn/`: `results.csv`, `results.png`,
`confusion_matrix.png`, `confusion_matrix_normalized.png`, `PR_curve.png`,
`F1_curve.png`, `P_curve.png`, `R_curve.png`, `experiment_config_used.yaml`
y `metrics_summary.csv`.""")

code('''# ============================================================
# Setup general del entorno
# ============================================================

from pathlib import Path
import os
import sys
import random
import shutil
import time
import yaml

SEED = 42
random.seed(SEED)

IN_COLAB = "google.colab" in sys.modules

print("Ejecutando en Google Colab:", IN_COLAB)
print("Directorio actual:", Path.cwd())''')

code('''# ============================================================
# Instalación de dependencias
# ============================================================

REPO_URL = "https://github.com/Gabriela-Sol/VpC2---Deteccion-de-humo-y-fuego"
REPO_NAME = "VpC2---Deteccion-de-humo-y-fuego"

if IN_COLAB:
    !pip install -q -r https://raw.githubusercontent.com/Gabriela-Sol/VpC2---Deteccion-de-humo-y-fuego/main/requirements.txt

print("Dependencias instaladas.")''')

code('''# ============================================================
# Verificación de GPU
# ============================================================

import torch

print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
    print("GPU:", DEVICE_NAME)
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "cpu"
    print("No se detectó GPU. Faster R-CNN en CPU es inviable para 12 épocas.")

print("Device:", DEVICE)''')

code('''# ============================================================
# Montar Google Drive
# ============================================================

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/VCII_DFire")
DRIVE_RUNS_DIR = DRIVE_PROJECT_DIR / "runs"
DRIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

print("Carpeta principal en Drive:", DRIVE_PROJECT_DIR)
print("Carpeta de corridas:", DRIVE_RUNS_DIR)''')

code('''# ============================================================
# Clonado o actualización del repositorio
# ============================================================

PROJECT_DIR = Path("/content") / REPO_NAME

if IN_COLAB:
    if PROJECT_DIR.exists():
        print("El repositorio ya existe. Actualizando...")
        %cd {PROJECT_DIR}
        !git pull
    else:
        print("Clonando repositorio...")
        %cd /content
        !git clone {REPO_URL}.git
        %cd {PROJECT_DIR}
else:
    PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

# Necesario para que `import src...` funcione.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

print("PROJECT_DIR:", PROJECT_DIR)
print("Contenido del proyecto:", os.listdir(PROJECT_DIR))''')

code('''# ============================================================
# Carga de configuración del experimento
# ============================================================

EXPERIMENT_CONFIG_PATH = PROJECT_DIR / "configs" / "experiments" / "fasterrcnn_r50fpn.yaml"

if not EXPERIMENT_CONFIG_PATH.exists():
    raise FileNotFoundError(f"No se encontró la configuración: {EXPERIMENT_CONFIG_PATH}")

with open(EXPERIMENT_CONFIG_PATH, "r", encoding="utf-8") as file:
    experiment_config = yaml.safe_load(file)

experiment_name = experiment_config["experiment"]["name"]
training_cfg = experiment_config["training"]

print("Experimento:", experiment_name)
print("Familia:", experiment_config["experiment"]["family"])
print("Modelo:", experiment_config["experiment"]["model"])
print("Épocas:", training_cfg["epochs"], "| batch:", training_cfg["batch"])''')

code('''# ============================================================
# Descarga o localización del dataset
# ============================================================

import kagglehub

DATASET_ID = "sayedgamal99/smoke-fire-detection-yolo"
dataset_root = Path(kagglehub.dataset_download(DATASET_ID))


def find_yolo_dataset_dir(root: Path) -> Path:
    """Busca la carpeta con train/images, train/labels, val/images y val/labels."""
    for candidate in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if all(
            (candidate / split / kind).exists()
            for split in ["train", "val"]
            for kind in ["images", "labels"]
        ):
            return candidate
    raise FileNotFoundError("No se encontró una estructura YOLO válida.")


DATA_DIR = find_yolo_dataset_dir(dataset_root)
print("Carpeta de datos:", DATA_DIR)''')

code('''# ============================================================
# Datasets y dataloaders
# ============================================================

from torch.utils.data import DataLoader

from src.data.yolo_dataset import YoloDetectionDataset, collate_fn

train_dataset = YoloDetectionDataset(DATA_DIR / "train", train=True, hflip_prob=0.5, seed=SEED)
val_dataset = YoloDetectionDataset(DATA_DIR / "val", train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=training_cfg["batch"],
    shuffle=True,
    num_workers=training_cfg["workers"],
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=training_cfg["batch"],
    shuffle=False,
    num_workers=training_cfg["workers"],
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
)

print("Imágenes de train:", len(train_dataset))
print("Imágenes de val:  ", len(val_dataset))

# Casi la mitad de las imágenes de train son negativos deliberados.
sin_cajas = sum(1 for i in range(200) if len(train_dataset[i][1]["boxes"]) == 0)
print(f"Negativos en las primeras 200 imágenes: {sin_cajas}")''')

code('''# ============================================================
# Modelo, optimizador y scheduler
# ============================================================

from src.engine.trainer import build_optimizer, build_scheduler
from src.modeling.detectors import build_fasterrcnn, count_parameters

torch.manual_seed(SEED)

model = build_fasterrcnn(
    num_classes=3,  # fondo + smoke + fire
    backbone=training_cfg["backbone"],
    trainable_backbone_layers=training_cfg["trainable_backbone_layers"],
    min_size=training_cfg["imgsz"],
    max_size=training_cfg["max_size"],
    pretrained=True,
)
model.to(DEVICE)

optimizer = build_optimizer(model, training_cfg)
scheduler = build_scheduler(optimizer, training_cfg, training_cfg["epochs"])
scaler = torch.amp.GradScaler("cuda") if (training_cfg["amp"] and DEVICE.type == "cuda") else None

print(f"Parámetros entrenables: {count_parameters(model) / 1e6:.2f} M")
print("Optimizador:", type(optimizer).__name__, "| scheduler:", type(scheduler).__name__)
print("AMP:", scaler is not None)''')

code('''# ============================================================
# Reanudar desde el último checkpoint si existe
# ============================================================

from src.engine.trainer import load_checkpoint

CHECKPOINT_PATH = DRIVE_RUNS_DIR / experiment_name / "last_checkpoint.pth"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

start_epoch = 0
history = []

if CHECKPOINT_PATH.exists():
    start_epoch, history = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, DEVICE)
    print(f"Checkpoint encontrado. Reanudando desde la época {start_epoch + 1}.")
else:
    print("No hay checkpoint previo. Entrenamiento desde cero.")''')

code('''# ============================================================
# Bucle de entrenamiento
# ============================================================

from src.engine.metrics import collect_predictions, compute_curves, compute_map
from src.engine.trainer import save_checkpoint, train_one_epoch

EPOCHS = training_cfg["epochs"]
elapsed_before = sum(row.get("epoch_time_min", 0.0) for row in history)
start_time = time.time()

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    print(f"\\n{'=' * 70}\\nÉpoca {epoch + 1}/{EPOCHS}\\n{'=' * 70}")

    losses = train_one_epoch(model, optimizer, train_loader, DEVICE, scaler=scaler)

    if scheduler is not None:
        scheduler.step()

    predictions, targets = collect_predictions(model, val_loader, DEVICE)
    map_metrics = compute_map(predictions, targets)
    curves = compute_curves(predictions, targets)

    history.append(
        {
            "epoch": epoch + 1,
            "train/loss_total": losses["loss_total"],
            "train/loss_classifier": losses["loss_classifier"],
            "train/loss_box_reg": losses["loss_box_reg"],
            "train/loss_objectness": losses["loss_objectness"],
            "train/loss_rpn_box_reg": losses["loss_rpn_box_reg"],
            "metrics/mAP50": map_metrics["map50"],
            "metrics/mAP50-95": map_metrics["map50_95"],
            "metrics/precision": curves["best_precision"],
            "metrics/recall": curves["best_recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_min": (time.time() - epoch_start) / 60,
        }
    )

    print(
        f"loss={losses['loss_total']:.4f} | "
        f"mAP50={map_metrics['map50']:.4f} | mAP50-95={map_metrics['map50_95']:.4f}"
    )

    # Se guarda al final de cada época: la sesión de Colab puede cortarse.
    save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, epoch + 1, history)
    print("Checkpoint guardado en:", CHECKPOINT_PATH)

train_time_min = elapsed_before + (time.time() - start_time) / 60
print(f"\\nEntrenamiento finalizado. Tiempo total acumulado: {train_time_min:.1f} min")''')

code('''# ============================================================
# Guardar los pesos finales en Drive
# ============================================================

if experiment_config["output"]["save_weights"]:
    WEIGHTS_PATH = DRIVE_RUNS_DIR / experiment_name / "best.pth"
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("Pesos guardados en:", WEIGHTS_PATH)''')

code('''# ============================================================
# Reporte completo del experimento
# ============================================================

from src.reporting.experiment_report import generate_experiment_report

REPORTS_RESULTS_DIR = PROJECT_DIR / "reports" / "results" / experiment_name

metrics = generate_experiment_report(
    model=model,
    val_loader=val_loader,
    val_dataset=val_dataset,
    config=experiment_config,
    history=history,
    out_dir=REPORTS_RESULTS_DIR,
    device=DEVICE,
    train_time_min=train_time_min,
    device_name=DEVICE_NAME,
)

import pandas as pd
display(pd.DataFrame([metrics]).T.rename(columns={0: "valor"}))''')

code('''# ============================================================
# Visualización de las figuras generadas
# ============================================================

from IPython.display import Image, display

for nombre in [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
]:
    ruta = REPORTS_RESULTS_DIR / nombre
    if ruta.exists():
        print(nombre)
        display(Image(filename=str(ruta)))
    else:
        print("No encontrado:", ruta)''')

code('''# ============================================================
# Commit y push de resultados al repositorio
# ============================================================

import subprocess

%cd {PROJECT_DIR}

!git config user.name "Gabriela-Sol"
!git config user.email "solgab.salazar@gmail.com"

pull_result = subprocess.run(
    ["git", "pull", "--rebase", "origin", "main"], text=True, capture_output=True
)
print(pull_result.stdout, pull_result.stderr)

if pull_result.returncode != 0:
    raise RuntimeError("No se pudo completar git pull --rebase. Revisar conflictos.")

for path in [f"reports/results/{experiment_name}/", "configs/experiments/fasterrcnn_r50fpn.yaml"]:
    if Path(path).exists():
        subprocess.run(["git", "add", path], check=True)
        print("Agregado:", path)

status = subprocess.run(["git", "status", "--short"], text=True, capture_output=True)
print(status.stdout)

if not status.stdout.strip():
    print("No hay cambios nuevos para commitear.")
else:
    subprocess.run(
        ["git", "commit", "-m", f"results: update {experiment_name} outputs"], check=True
    )
    print("Commit creado. Para publicarlo: !git push origin main")''')

nb["cells"] = CELDAS
nbf.write(nb, "notebooks/03_entrenamiento_FasterRCNN.ipynb")
print("Notebook escrito con", len(CELDAS), "celdas")
```

- [ ] **Step 3: Generar el notebook**

Run desde la raíz del repo: `.venv/bin/python <ruta-del-script>/build_nb03.py`
Expected: `Notebook escrito con 16 celdas`

- [ ] **Step 4: Validar que el notebook es JSON válido y que las celdas de `src` compilan**

```bash
.venv/bin/python - <<'PY'
import json, ast
nb = json.load(open("notebooks/03_entrenamiento_FasterRCNN.ipynb"))
print("celdas:", len(nb["cells"]))

saltadas = 0
for i, celda in enumerate(nb["cells"]):
    if celda["cell_type"] != "code":
        continue
    fuente = "".join(celda["source"])
    # Las celdas con magias de IPython (%cd, !pip) no son Python válido.
    if any(linea.lstrip().startswith(("!", "%")) for linea in fuente.splitlines()):
        saltadas += 1
        continue
    ast.parse(fuente)  # falla ruidosamente si hay un error de sintaxis
print("celdas de código verificadas; con magias, salteadas:", saltadas)
PY
```
Expected: `celdas: 16` y ningún `SyntaxError`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_entrenamiento_FasterRCNN.ipynb
git commit -m "feat: notebook de entrenamiento de Faster R-CNN"
```

---

### Task 11: Notebook 04 — entrenamiento de RT-DETR

**Files:**
- Create: `notebooks/04_entrenamiento_RTDETR.ipynb`

**Interfaces:**
- Consumes: `configs/experiments/rtdetr_l.yaml`, `src/reporting/summary.py`.
- Produces: `reports/results/rtdetr_l/` con los artefactos de Ultralytics más `metrics_summary.csv`.

RT-DETR reusa el camino de Ultralytics, así que este notebook es el 02 con `RTDETR` en lugar de `YOLO`, más la celda que exporta el resumen común.

- [ ] **Step 1: Escribir el script generador**

Mismo helper que en la Task 10 (`md`, `code`, `nbf`), guardado como `build_nb04.py` en el scratchpad. El notebook tiene 13 celdas en este orden: la celda markdown de título, las cinco de infraestructura copiadas **verbatim** de `build_nb03.py` (setup, dependencias, GPU, Drive, clonado — incluida la línea `sys.path.insert`, que este notebook necesita para importar `src.reporting.summary`) y luego las siete específicas que se detallan abajo:

```python
md("""# 04 - Entrenamiento de RT-DETR para detección de humo y fuego

RT-DETR es un detector basado en transformers con NMS-free y velocidad de
tiempo real. Se entrena con la misma API de Ultralytics que el baseline YOLOv8n,
así que este notebook reusa el pipeline del notebook 02.""")

code('''# ============================================================
# Carga de configuración del experimento
# ============================================================

EXPERIMENT_CONFIG_PATH = PROJECT_DIR / "configs" / "experiments" / "rtdetr_l.yaml"

with open(EXPERIMENT_CONFIG_PATH, "r", encoding="utf-8") as file:
    experiment_config = yaml.safe_load(file)

experiment_name = experiment_config["experiment"]["name"]
model_name = experiment_config["experiment"]["model"]
training_cfg = experiment_config["training"]

print("Experimento:", experiment_name)
print("Modelo:", model_name, "| épocas:", training_cfg["epochs"])''')

code('''# ============================================================
# Dataset y YAML para Ultralytics
# ============================================================

import kagglehub

DATASET_ID = "sayedgamal99/smoke-fire-detection-yolo"
dataset_root = Path(kagglehub.dataset_download(DATASET_ID))


def find_yolo_dataset_dir(root: Path) -> Path:
    for candidate in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if all(
            (candidate / split / kind).exists()
            for split in ["train", "val"]
            for kind in ["images", "labels"]
        ):
            return candidate
    raise FileNotFoundError("No se encontró una estructura YOLO válida.")


DATA_DIR = find_yolo_dataset_dir(dataset_root)
DFIRE_YAML = Path("/content/dfire_colab.yaml")

with open(DFIRE_YAML, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        {
            "path": str(DATA_DIR),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 2,
            "names": {0: "smoke", 1: "fire"},
        },
        file,
        sort_keys=False,
        allow_unicode=True,
    )

print("Dataset:", DATA_DIR)
print(DFIRE_YAML.read_text())''')

code('''# ============================================================
# Entrenamiento con RT-DETR
# ============================================================

from ultralytics import RTDETR

project_dir = Path(experiment_config["output"]["project"])
project_dir.mkdir(parents=True, exist_ok=True)
experiment_dir = project_dir / experiment_name

RESUME = (experiment_dir / "weights" / "last.pt").exists()
start_time = time.time()

if RESUME:
    print("Checkpoint encontrado, reanudando entrenamiento.")
    model = RTDETR(str(experiment_dir / "weights" / "last.pt"))
    results = model.train(resume=True)
else:
    print("Entrenamiento desde los pesos preentrenados.")
    model = RTDETR(model_name)
    results = model.train(
        data=str(DFIRE_YAML),
        epochs=training_cfg["epochs"],
        imgsz=training_cfg["imgsz"],
        batch=training_cfg["batch"],
        patience=training_cfg["patience"],
        optimizer=training_cfg["optimizer"],
        lr0=training_cfg["lr0"],
        seed=training_cfg["seed"],
        project=str(project_dir),
        name=experiment_name,
        exist_ok=True,
        plots=True,
    )

train_time_min = (time.time() - start_time) / 60
print(f"Entrenamiento finalizado en {train_time_min:.1f} min")
print("Resultados en:", experiment_dir)''')

code('''# ============================================================
# Validación y métricas finales
# ============================================================

best_weights = experiment_dir / "weights" / "best.pt"
model = RTDETR(str(best_weights))

metrics = model.val(data=str(DFIRE_YAML), split="val", plots=True)

print("mAP50   :", metrics.box.map50)
print("mAP50-95:", metrics.box.map)''')

code('''# ============================================================
# Copiar los artefactos de Ultralytics al repositorio
# ============================================================

REPORTS_RESULTS_DIR = PROJECT_DIR / "reports" / "results" / experiment_name
REPORTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for filename in [
    "results.csv", "results.png",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
]:
    src_file = experiment_dir / filename
    if src_file.exists():
        shutil.copy(src_file, REPORTS_RESULTS_DIR / filename)
        print("Copiado:", filename)
    else:
        print("No encontrado:", src_file)

with open(REPORTS_RESULTS_DIR / "experiment_config_used.yaml", "w", encoding="utf-8") as file:
    yaml.safe_dump(experiment_config, file, sort_keys=False, allow_unicode=True)

print("Config usada guardada.")''')

code('''# ============================================================
# Exportar metrics_summary.csv con el esquema común
# ============================================================

import numpy as np

from src.reporting.summary import write_metrics_summary

# Ultralytics ordena las clases como en el YAML: 0 = smoke, 1 = fire.
map50_por_clase = metrics.box.ap50
map95_por_clase = metrics.box.ap

# La velocidad viene en milisegundos por imagen: inferencia más postproceso.
ms_por_imagen = metrics.speed["inference"] + metrics.speed["postprocess"]

precision = float(np.mean(metrics.box.p))
recall = float(np.mean(metrics.box.r))

resumen = {
    "experiment": experiment_name,
    "family": experiment_config["experiment"]["family"],
    "model": model_name,
    "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
    "epochs": training_cfg["epochs"],
    "imgsz": training_cfg["imgsz"],
    "batch": training_cfg["batch"],
    "train_time_min": round(train_time_min, 2),
    "mAP50": round(float(metrics.box.map50), 4),
    "mAP50_95": round(float(metrics.box.map), 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
    "mAP50_smoke": round(float(map50_por_clase[0]), 4),
    "mAP50_fire": round(float(map50_por_clase[1]), 4),
    "mAP50_95_smoke": round(float(map95_por_clase[0]), 4),
    "mAP50_95_fire": round(float(map95_por_clase[1]), 4),
    "fps": round(1000.0 / ms_por_imagen, 2),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "split": "val",
}

df_resumen = write_metrics_summary(REPORTS_RESULTS_DIR / "metrics_summary.csv", resumen)
display(df_resumen.T.rename(columns={0: "valor"}))''')

code('''# ============================================================
# Commit y push de resultados al repositorio
# ============================================================

import subprocess

%cd {PROJECT_DIR}

!git config user.name "Gabriela-Sol"
!git config user.email "solgab.salazar@gmail.com"

pull_result = subprocess.run(
    ["git", "pull", "--rebase", "origin", "main"], text=True, capture_output=True
)
print(pull_result.stdout, pull_result.stderr)

if pull_result.returncode != 0:
    raise RuntimeError("No se pudo completar git pull --rebase. Revisar conflictos.")

for path in [f"reports/results/{experiment_name}/", "configs/experiments/rtdetr_l.yaml"]:
    if Path(path).exists():
        subprocess.run(["git", "add", path], check=True)
        print("Agregado:", path)

status = subprocess.run(["git", "status", "--short"], text=True, capture_output=True)
print(status.stdout)

if not status.stdout.strip():
    print("No hay cambios nuevos para commitear.")
else:
    subprocess.run(
        ["git", "commit", "-m", f"results: update {experiment_name} outputs"], check=True
    )
    print("Commit creado. Para publicarlo: !git push origin main")''')
```

- [ ] **Step 2: Generar y validar el notebook**

Run: `.venv/bin/python <ruta-del-script>/build_nb04.py`
Luego el mismo chequeo de sintaxis de la Task 10 Step 4, cambiando el nombre del archivo a `notebooks/04_entrenamiento_RTDETR.ipynb`.
Expected: JSON válido, sin `SyntaxError`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_entrenamiento_RTDETR.ipynb
git commit -m "feat: notebook de entrenamiento de RT-DETR"
```

---

### Task 12: Notebook 05 de comparación y exportación del baseline YOLO

**Files:**
- Create: `notebooks/05_comparacion_modelos.ipynb`
- Modify: `notebooks/02_entrenamiento_YOLO.ipynb` (agregar una celda al final)
- Modify: `reports/README.md`

**Interfaces:**
- Consumes: `load_metrics_summaries`, `write_metrics_summary`, `plot_model_comparison`.
- Produces: `reports/figures/comparacion/` con las tres figuras, y `reports/results/comparacion_modelos.csv`.

- [ ] **Step 1: Agregar al notebook 02 la celda que exporta el resumen del baseline**

Sin este paso el baseline YOLOv8n no aparece en la comparación. La celda se agrega al final de `notebooks/02_entrenamiento_YOLO.ipynb` y se ejecuta en Colab con acceso a Drive.

```python
.venv/bin/python - <<'PY'
import nbformat as nbf

CELDA = '''# ============================================================
# Exportar metrics_summary.csv del baseline con el esquema común
# ============================================================
# Esta celda permite que YOLOv8n participe de la comparación del notebook 05.

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROJECT_DIR = Path("/content/VpC2---Deteccion-de-humo-y-fuego")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.reporting.summary import write_metrics_summary

experiment_name = "yolov8n_baseline"
experiment_dir = Path("/content/drive/MyDrive/VCII_DFire/runs") / experiment_name

with open(PROJECT_DIR / "configs" / "experiments" / f"{experiment_name}.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = YOLO(str(experiment_dir / "weights" / "best.pt"))
metrics = model.val(data="/content/dfire_colab.yaml", split="val")

# El tiempo de entrenamiento sale de la última fila de results.csv, en segundos.
import pandas as pd
train_time_min = round(pd.read_csv(experiment_dir / "results.csv")["time"].iloc[-1] / 60, 2)

ms_por_imagen = metrics.speed["inference"] + metrics.speed["postprocess"]
precision = float(np.mean(metrics.box.p))
recall = float(np.mean(metrics.box.r))

resumen = {
    "experiment": experiment_name,
    "family": cfg["experiment"]["family"],
    "model": cfg["experiment"]["model"],
    "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
    "epochs": cfg["training"]["epochs"],
    "imgsz": cfg["training"]["imgsz"],
    "batch": cfg["training"]["batch"],
    "train_time_min": train_time_min,
    "mAP50": round(float(metrics.box.map50), 4),
    "mAP50_95": round(float(metrics.box.map), 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
    "mAP50_smoke": round(float(metrics.box.ap50[0]), 4),
    "mAP50_fire": round(float(metrics.box.ap50[1]), 4),
    "mAP50_95_smoke": round(float(metrics.box.ap[0]), 4),
    "mAP50_95_fire": round(float(metrics.box.ap[1]), 4),
    "fps": round(1000.0 / ms_por_imagen, 2),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "split": "val",
}

destino = PROJECT_DIR / "reports" / "results" / experiment_name / "metrics_summary.csv"
display(write_metrics_summary(destino, resumen).T.rename(columns={0: "valor"}))
print("Guardado en:", destino)
'''

ruta = "notebooks/02_entrenamiento_YOLO.ipynb"
nb = nbf.read(ruta, as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(CELDA))
nbf.write(nb, ruta)
print("Celda agregada. Total de celdas:", len(nb["cells"]))
PY
```

Expected: `Celda agregada. Total de celdas: 27`

- [ ] **Step 2: Escribir el script generador del notebook 05**

Guardar como `build_nb05.py` en el scratchpad, con el mismo helper `md`/`code` de la Task 10:

```python
md("""# 05 - Comparación de modelos de detección de humo y fuego

Consolida los `metrics_summary.csv` de todos los experimentos y genera la tabla
y las figuras comparativas.

Este notebook **no necesita GPU, ni Drive, ni el dataset**: trabaja solo con los
CSV versionados en el repositorio, así que la comparación es reproducible por
cualquiera que clone el proyecto.""")

code('''# ============================================================
# Setup
# ============================================================

from pathlib import Path
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !pip install -q -r https://raw.githubusercontent.com/Gabriela-Sol/VpC2---Deteccion-de-humo-y-fuego/main/requirements.txt
    !git clone -q https://github.com/Gabriela-Sol/VpC2---Deteccion-de-humo-y-fuego.git /content/VpC2---Deteccion-de-humo-y-fuego
    PROJECT_DIR = Path("/content/VpC2---Deteccion-de-humo-y-fuego")
else:
    PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

print("PROJECT_DIR:", PROJECT_DIR)''')

code('''# ============================================================
# Carga de los resúmenes de cada experimento
# ============================================================

import pandas as pd

from src.reporting.summary import load_metrics_summaries

RESULTS_DIR = PROJECT_DIR / "reports" / "results"
df = load_metrics_summaries(RESULTS_DIR)

if df.empty:
    raise RuntimeError(
        f"No se encontró ningún metrics_summary.csv en {RESULTS_DIR}. "
        "Correr antes los notebooks 02, 03 y 04."
    )

print(f"Experimentos encontrados: {len(df)}")
display(df)''')

code('''# ============================================================
# Tabla comparativa principal
# ============================================================

COLUMNAS = [
    "experiment", "family", "params_M", "epochs", "train_time_min",
    "mAP50", "mAP50_95", "precision", "recall", "f1", "fps",
]

tabla = df[COLUMNAS].copy()
tabla.columns = [
    "Experimento", "Familia", "Params (M)", "Épocas", "Entrenamiento (min)",
    "mAP@0.5", "mAP@0.5:0.95", "Precisión", "Recall", "F1", "FPS",
]

display(tabla.style.background_gradient(subset=["mAP@0.5", "mAP@0.5:0.95"], cmap="Greens"))''')

code('''# ============================================================
# Desempeño por clase
# ============================================================

por_clase = df[
    ["experiment", "mAP50_smoke", "mAP50_fire", "mAP50_95_smoke", "mAP50_95_fire"]
].copy()
por_clase.columns = [
    "Experimento", "mAP@0.5 smoke", "mAP@0.5 fire",
    "mAP@0.5:0.95 smoke", "mAP@0.5:0.95 fire",
]

display(por_clase)

# El humo suele ser más difícil que el fuego: bordes difusos y sin forma definida.
diferencia = (df["mAP50_fire"] - df["mAP50_smoke"]).mean()
print(f"\\nVentaja media de fire sobre smoke en mAP@0.5: {diferencia:+.4f}")''')

code('''# ============================================================
# Figuras comparativas
# ============================================================

from IPython.display import Image, display

from src.reporting.plots import plot_model_comparison

FIGURES_DIR = PROJECT_DIR / "reports" / "figures" / "comparacion"
rutas = plot_model_comparison(df, FIGURES_DIR)

for ruta in rutas:
    print(ruta.name)
    display(Image(filename=str(ruta)))''')

code('''# ============================================================
# Guardar la tabla consolidada
# ============================================================

COMPARISON_CSV = PROJECT_DIR / "reports" / "results" / "comparacion_modelos.csv"
df.to_csv(COMPARISON_CSV, index=False)

print("Tabla comparativa guardada en:", COMPARISON_CSV)

mejor = df.iloc[0]
print(
    f"\\nMejor modelo por mAP@0.5: {mejor['experiment']} "
    f"({mejor['mAP50']:.4f}), a {mejor['fps']:.1f} FPS."
)''')

code('''# ============================================================
# Commit de la comparación
# ============================================================

import subprocess

%cd {PROJECT_DIR}

!git config user.name "Gabriela-Sol"
!git config user.email "solgab.salazar@gmail.com"

for path in ["reports/figures/comparacion/", "reports/results/comparacion_modelos.csv"]:
    if Path(path).exists():
        subprocess.run(["git", "add", path], check=True)
        print("Agregado:", path)

status = subprocess.run(["git", "status", "--short"], text=True, capture_output=True)
print(status.stdout)

if status.stdout.strip():
    subprocess.run(["git", "commit", "-m", "results: comparacion entre modelos"], check=True)
    print("Commit creado. Para publicarlo: !git push origin main")
else:
    print("No hay cambios nuevos para commitear.")''')
```

- [ ] **Step 3: Generar el notebook y verificar la comparación con datos de prueba**

```bash
.venv/bin/python <ruta-del-script>/build_nb05.py
```

Después, verificar que la lógica del notebook funciona sobre un directorio simulado:

```bash
.venv/bin/python - <<'PY'
import tempfile
from pathlib import Path

import pandas as pd

from src.reporting.plots import plot_model_comparison
from src.reporting.summary import load_metrics_summaries, write_metrics_summary

BASE = {
    "family": "X", "model": "m", "params_M": 3.2, "epochs": 12, "imgsz": 640,
    "batch": 4, "train_time_min": 100.0, "precision": 0.8, "recall": 0.7,
    "f1": 0.75, "mAP50_95_smoke": 0.4, "mAP50_95_fire": 0.45,
    "device": "cpu", "split": "val",
}

with tempfile.TemporaryDirectory() as tmp:
    raiz = Path(tmp)
    (raiz / "eda").mkdir()  # debe ser ignorada por el glob
    for nombre, map50, map95, fps in [
        ("yolov8n_baseline", 0.75, 0.43, 90.0),
        ("fasterrcnn_r50fpn", 0.80, 0.47, 12.0),
        ("rtdetr_l", 0.82, 0.50, 35.0),
    ]:
        carpeta = raiz / nombre
        carpeta.mkdir()
        write_metrics_summary(
            carpeta / "metrics_summary.csv",
            {**BASE, "experiment": nombre, "mAP50": map50, "mAP50_95": map95,
             "fps": fps, "mAP50_smoke": map50 - 0.02, "mAP50_fire": map50 + 0.02},
        )

    df = load_metrics_summaries(raiz)
    assert len(df) == 3, f"esperaba 3 experimentos, hubo {len(df)}"
    assert df.iloc[0]["experiment"] == "rtdetr_l", "debe ordenar por mAP50 desc"

    rutas = plot_model_comparison(df, raiz / "figuras")
    assert len(rutas) == 3 and all(r.stat().st_size > 1000 for r in rutas)
    print("Comparación verificada:", [r.name for r in rutas])
PY
```

Expected: `Comparación verificada: ['map_por_modelo.png', 'map_por_clase.png', 'precision_vs_velocidad.png']`

- [ ] **Step 4: Documentar la estructura en `reports/README.md`**

El archivo está vacío. Reemplazar su contenido con:

```markdown
# Resultados

## Estructura

- `results/eda/` — tablas del análisis exploratorio (notebook 01).
- `results/<experimento>/` — una carpeta por experimento entrenado.
- `results/comparacion_modelos.csv` — tabla consolidada de todos los experimentos.
- `figures/eda/` — figuras del análisis exploratorio.
- `figures/comparacion/` — figuras comparativas entre modelos (notebook 05).

## Contenido de cada carpeta de experimento

| Archivo | Descripción |
|---|---|
| `results.csv` | métricas por época |
| `results.png` | curvas de pérdida y métricas |
| `confusion_matrix.png` | matriz de confusión (IoU 0.45, confianza 0.25) |
| `confusion_matrix_normalized.png` | la misma, normalizada por columna |
| `PR_curve.png` | curva precisión-recall por clase (IoU 0.5) |
| `F1_curve.png`, `P_curve.png`, `R_curve.png` | métricas en función de la confianza |
| `experiment_config_used.yaml` | configuración exacta con la que se corrió |
| `metrics_summary.csv` | resumen de una fila, con esquema común a todos los modelos |

## `metrics_summary.csv`

Es el archivo que hace posible comparar modelos entrenados con librerías
distintas: el `results.csv` de Ultralytics tiene columnas propias que no aplican
a torchvision. Todas las métricas se calculan sobre el split de **validación**
con protocolo COCO. Las columnas `precision`, `recall` y `f1` se reportan en el
umbral de confianza que maximiza F1.

## Experimentos

| Experimento | Familia | Notebook |
|---|---|---|
| `yolov8n_baseline` | Una etapa, CNN | `02_entrenamiento_YOLO.ipynb` |
| `fasterrcnn_r50fpn` | Dos etapas, CNN | `03_entrenamiento_FasterRCNN.ipynb` |
| `rtdetr_l` | Transformer | `04_entrenamiento_RTDETR.ipynb` |
```

- [ ] **Step 5: Correr toda la batería de tests una última vez**

Run: `.venv/bin/python -m pytest -v`
Expected: 66 passed.

- [ ] **Step 6: Commit**

```bash
git add notebooks/05_comparacion_modelos.ipynb notebooks/02_entrenamiento_YOLO.ipynb reports/README.md
git commit -m "feat: notebook de comparacion entre modelos y export del baseline"
```

---

## Orden de ejecución en Colab

Los tests de este plan corren en CPU y verifican el código, no el entrenamiento real. Una vez implementado todo, el orden en Colab es:

1. `03_entrenamiento_FasterRCNN.ipynb` — unas 5 a 6 horas con GPU T4. Si la sesión se corta, volver a ejecutar el notebook completo: reanuda desde el último checkpoint en Drive.
2. `04_entrenamiento_RTDETR.ipynb` — RT-DETR-l a 20 épocas. También reanudable.
3. La celda nueva del final de `02_entrenamiento_YOLO.ipynb`, para exportar el resumen del baseline.
4. `05_comparacion_modelos.ipynb` — sin GPU, en cualquier lado.

## Notas de riesgo

- **Memoria de GPU:** si `fasterrcnn_resnet50_fpn_v2` con batch 4 a 640 px agota la memoria de la T4, bajar `batch` a 2 en el YAML antes de tocar cualquier otra cosa. Si aun así no entra, cambiar `backbone` a `mobilenet_v3_large_fpn`, que no requiere cambios de código.
- **Tiempo:** si 12 épocas resultan inviables, reducir a 8 en el YAML. El reporte y la comparación funcionan igual; solo hay que dejar constancia del número real de épocas, que el propio `metrics_summary.csv` registra.
- **`num_workers`:** en Colab, valores por encima de 2 suelen producir cuelgues del DataLoader. El YAML fija 2 a propósito.
