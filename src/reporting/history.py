"""Historial por época de cada experimento, en un formato común.

`metrics_summary.csv` compara los valores finales; esto compara la evolución.
La fuente es el `results.csv` de cada experimento, que existe en los tres pero
con nombres de columna distintos: Ultralytics le pone el sufijo `(B)` a las
métricas de bounding box (`metrics/mAP50(B)`) y el bucle de Faster R-CNN, que
escribe el suyo con `write_history_csv`, no. Sin traducir esos nombres a uno
solo, las curvas de los tres modelos no pueden ir al mismo eje.

Las pérdidas quedan deliberadamente afuera. Son `box/cls/dfl` en YOLOv8n,
`giou/cls/l1` en RT-DETR y `loss_total` en Faster R-CNN: formulaciones distintas
en escalas distintas, así que superponerlas insinuaría una comparación que no
significa nada.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORY_COLUMNS = ["experiment", "epoch", "mAP50", "mAP50_95", "precision", "recall"]

# Nombre común -> los alias con los que puede venir en un results.csv.
COLUMNAS_POR_METRICA = {
    "mAP50": ("metrics/mAP50(B)", "metrics/mAP50"),
    "mAP50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    "precision": ("metrics/precision(B)", "metrics/precision"),
    "recall": ("metrics/recall(B)", "metrics/recall"),
}

RESULTS_FILENAME = "results.csv"


def _normalizar(df: pd.DataFrame, experimento: str) -> pd.DataFrame | None:
    """Traduce un results.csv al esquema común, o `None` si le falta algo."""
    normalizado = {"experiment": experimento, "epoch": df["epoch"]}

    for metrica, alias in COLUMNAS_POR_METRICA.items():
        presentes = [nombre for nombre in alias if nombre in df.columns]
        if not presentes:
            return None
        normalizado[metrica] = df[presentes[0]]

    return pd.DataFrame(normalizado)[HISTORY_COLUMNS]


def load_training_histories(results_root) -> pd.DataFrame:
    """Junta los `results.csv` de todos los experimentos, una fila por época.

    Un experimento al que le falte alguna de las cuatro métricas se saltea en
    vez de cortar: un `results.csv` truncado (una corrida que murió antes de
    cerrar la primera época) no tiene por qué llevarse puesta la figura de los
    demás.
    """
    filas = []

    for ruta in sorted(Path(results_root).glob(f"*/{RESULTS_FILENAME}")):
        experimento = ruta.parent.name
        normalizado = _normalizar(pd.read_csv(ruta), experimento)

        if normalizado is None:
            print(f"Sin métricas por época, se saltea: {experimento}")
            continue

        filas.append(normalizado)

    if not filas:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    return pd.concat(filas, ignore_index=True)
