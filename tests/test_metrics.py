"""Tests de métricas: mAP, curvas de confianza y FPS."""

import numpy as np
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
