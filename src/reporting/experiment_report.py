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

    experiment = config["experiment"]
    training = config["training"]

    # Los nombres de clase en el CSV son fijos, así que se resuelven por etiqueta.
    label_smoke, label_fire = LABEL_ORDER

    metrics = {
        "experiment": experiment["name"],
        "family": experiment["family"],
        "model": experiment["model"],
        "params_M": round(count_parameters(model) / 1e6, 2),
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

    write_metrics_summary(out_dir / "metrics_summary.csv", metrics)
    print(f"Reporte completo en: {out_dir}")
    return metrics
