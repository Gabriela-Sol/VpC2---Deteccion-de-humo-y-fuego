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
                "ap": float("nan"),
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
