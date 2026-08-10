"""Test de integración del reporte completo, en CPU y con un modelo sin entrenar."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.yolo_dataset import YoloDetectionDataset, collate_fn
from src.modeling.detectors import build_fasterrcnn
from src.reporting.experiment_report import (
    build_metrics_row,
    build_ultralytics_metrics_row,
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


class _BoxMetricsFalsas:
    def __init__(self, ap50, ap, p, r, ap_class_index):
        self.map50, self.map = 0.70, 0.40
        self.ap50, self.ap = np.array(ap50), np.array(ap)
        self.p, self.r = np.array(p), np.array(r)
        self.ap_class_index = np.array(ap_class_index)


def test_build_ultralytics_metrics_row_asigna_cada_clase_a_su_columna():
    # Cuatro valores distintos por la misma razón que en build_metrics_row: un
    # intercambio entre smoke y fire daría un informe plausible y equivocado.
    fila = build_ultralytics_metrics_row(
        config=CONFIG,
        box_metrics=_BoxMetricsFalsas(
            ap50=[0.11, 0.22], ap=[0.33, 0.44],
            p=[0.5, 0.7], r=[0.4, 0.6], ap_class_index=[0, 1],
        ),
        speed={"preprocess": 1.0, "inference": 8.0, "postprocess": 2.0},
        params_M=3.2,
        train_time_min=1.5,
        device_name="Tesla T4",
    )

    assert fila["mAP50_smoke"] == 0.11
    assert fila["mAP50_fire"] == 0.22
    assert fila["mAP50_95_smoke"] == 0.33
    assert fila["mAP50_95_fire"] == 0.44
    assert fila["precision"] == 0.6
    assert fila["recall"] == 0.5
    # Media armónica de 0.6 y 0.5.
    assert fila["f1"] == round(2 * 0.6 * 0.5 / 1.1, 4)
    # El preproceso no cuenta: 1000 / (8 + 2).
    assert fila["fps"] == 100.0
    assert fila["split"] == "val"
    assert set(fila) == set(METRICS_SUMMARY_COLUMNS)


def test_build_ultralytics_metrics_row_con_una_clase_ausente():
    # Solo fire tuvo etiquetas en validación: ap_class_index = [1]. Indexar por
    # posición le daría a smoke los números de fire.
    fila = build_ultralytics_metrics_row(
        config=CONFIG,
        box_metrics=_BoxMetricsFalsas(
            ap50=[0.22], ap=[0.44], p=[0.7], r=[0.6], ap_class_index=[1],
        ),
        speed={"inference": 8.0, "postprocess": 2.0},
        params_M=3.2,
        train_time_min=1.5,
        device_name="Tesla T4",
    )

    assert np.isnan(fila["mAP50_smoke"])
    assert np.isnan(fila["mAP50_95_smoke"])
    assert fila["mAP50_fire"] == 0.22
    assert fila["mAP50_95_fire"] == 0.44


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
