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
