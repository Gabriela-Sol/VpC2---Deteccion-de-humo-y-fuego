"""Test de integración del reporte completo, en CPU y con un modelo sin entrenar."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.yolo_dataset import YoloDetectionDataset, collate_fn
from src.modeling.detectors import build_fasterrcnn
from src.reporting.experiment_report import generate_experiment_report, write_history_csv
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
