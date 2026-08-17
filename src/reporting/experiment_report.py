"""De un modelo entrenado a una carpeta de resultados completa.

Concentra acá todo el reporte para que los notebooks queden finos y para que
Faster R-CNN produzca exactamente los mismos archivos que genera Ultralytics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def write_history_csv(history: list[dict], out_csv) -> Path:
    """Historial por época, con el mismo espíritu que el results.csv de Ultralytics."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_csv, index=False)
    return out_csv


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


def build_ultralytics_metrics_row(
    config: dict,
    box_metrics,
    speed: dict,
    params_M: float,
    train_time_min: float,
    device_name: str,
) -> dict:
    """Arma la fila de `metrics_summary.csv` a partir de un resultado de Ultralytics.

    `box_metrics` es el `metrics.box` que devuelve `model.val()`. Sus vectores
    `ap50` y `ap` traen una fila por clase QUE TUVO ETIQUETAS, indexada por
    `ap_class_index` y no por id de clase: si una clase faltara en validación,
    indexar por posición le atribuiría sus métricas a la otra.

    Está acá y no duplicada en los notebooks 02 y 04 porque un intercambio
    entre smoke y fire produce un informe plausible y equivocado, y porque la
    versión duplicada ya costó una corrección doble.
    """
    experiment = config["experiment"]
    training = config["training"]

    # Ids del YAML del dataset (D-Fire), no las etiquetas de torchvision.
    smoke_id, fire_id = 0, 1
    fila_de_clase = {
        int(clase): posicion
        for posicion, clase in enumerate(box_metrics.ap_class_index)
    }

    def ap_de(vector, clase: int) -> float:
        indice = fila_de_clase.get(clase)
        return float(vector[indice]) if indice is not None else float("nan")

    precision = float(np.mean(box_metrics.p))
    recall = float(np.mean(box_metrics.r))
    ms_por_imagen = speed["inference"] + speed["postprocess"]

    return {
        "experiment": experiment["name"],
        "family": experiment["family"],
        "model": experiment["model"],
        "params_M": round(params_M, 2),
        "epochs": training["epochs"],
        "imgsz": training["imgsz"],
        "batch": training["batch"],
        "train_time_min": round(float(train_time_min), 2),
        "mAP50": round(float(box_metrics.map50), 4),
        "mAP50_95": round(float(box_metrics.map), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall)
        else 0.0,
        "mAP50_smoke": round(ap_de(box_metrics.ap50, smoke_id), 4),
        "mAP50_fire": round(ap_de(box_metrics.ap50, fire_id), 4),
        "mAP50_95_smoke": round(ap_de(box_metrics.ap, smoke_id), 4),
        "mAP50_95_fire": round(ap_de(box_metrics.ap, fire_id), 4),
        "fps": round(1000.0 / ms_por_imagen, 2),
        "device": device_name,
        "split": "val",
    }


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
    """Evalúa el modelo sobre el loader recibido y escribe los diez artefactos.

    Los parámetros se llaman `val_loader`/`val_dataset` por historia: los
    notebooks hoy pasan el split de test para las métricas finales.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Recolectando predicciones sobre el set de evaluación...")
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
        # La columna params_M significa parámetros TOTALES, que es lo que suman
        # los notebooks de Ultralytics. `count_parameters` cuenta solo los
        # entrenables y con trainable_backbone_layers=3 parte del ResNet50 está
        # congelada, así que usarla acá haría incomparable la columna.
        params_M=sum(p.numel() for p in model.parameters()) / 1e6,
        fps=fps,
        train_time_min=train_time_min,
        device_name=device_name,
    )

    write_metrics_summary(out_dir / "metrics_summary.csv", metrics)
    print(f"Reporte completo en: {out_dir}")
    return metrics
