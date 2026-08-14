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


def test_el_nan_sobrevive_la_ida_y_vuelta_por_csv(tmp_path):
    # compute_map devuelve NaN para las clases sin ningún objeto real, así que
    # mAP50_fire puede ser NaN legítimamente, escribirse al CSV y volver a
    # leerse en el notebook 05. Un cambio de na_rep o de dtype rompería esto.
    metricas = _metricas()
    metricas["mAP50_fire"] = float("nan")
    metricas["mAP50_95_fire"] = float("nan")

    carpeta = tmp_path / "fasterrcnn_r50fpn"
    carpeta.mkdir()
    write_metrics_summary(carpeta / "metrics_summary.csv", metricas)

    df = load_metrics_summaries(tmp_path)

    assert pd.isna(df.loc[0, "mAP50_fire"])
    assert pd.isna(df.loc[0, "mAP50_95_fire"])
    # Las demás columnas no se contaminan.
    assert df.loc[0, "mAP50_smoke"] == 0.77
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
