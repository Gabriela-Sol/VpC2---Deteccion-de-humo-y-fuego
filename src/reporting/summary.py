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
