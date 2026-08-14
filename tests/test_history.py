"""Tests del historial por época, común a Ultralytics y a torchvision."""

import pandas as pd

from src.reporting.history import HISTORY_COLUMNS, load_training_histories


def _results_ultralytics(epocas=3):
    """El results.csv de Ultralytics: métricas con sufijo `(B)` y `time` acumulado."""
    return pd.DataFrame(
        {
            "epoch": range(1, epocas + 1),
            "time": [100.0 * i for i in range(1, epocas + 1)],
            "train/box_loss": [1.5] * epocas,
            "metrics/precision(B)": [0.10 * i for i in range(1, epocas + 1)],
            "metrics/recall(B)": [0.11 * i for i in range(1, epocas + 1)],
            "metrics/mAP50(B)": [0.12 * i for i in range(1, epocas + 1)],
            "metrics/mAP50-95(B)": [0.13 * i for i in range(1, epocas + 1)],
        }
    )


def _results_torchvision(epocas=3):
    """El results.csv que escribe write_history_csv: los mismos datos, sin `(B)`."""
    return pd.DataFrame(
        {
            "epoch": range(1, epocas + 1),
            "train/loss_total": [0.2] * epocas,
            "metrics/mAP50": [0.12 * i for i in range(1, epocas + 1)],
            "metrics/mAP50-95": [0.13 * i for i in range(1, epocas + 1)],
            "metrics/precision": [0.10 * i for i in range(1, epocas + 1)],
            "metrics/recall": [0.11 * i for i in range(1, epocas + 1)],
            "epoch_time_min": [5.0] * epocas,
        }
    )


def _escribir(root, nombre, df):
    destino = root / nombre
    destino.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino / "results.csv", index=False)


def test_las_dos_convenciones_de_nombres_dan_el_mismo_resultado(tmp_path):
    """`metrics/mAP50(B)` y `metrics/mAP50` son la misma métrica.

    Es el punto de todo el módulo: Ultralytics le pone el sufijo `(B)` (de "box")
    y el bucle de Faster R-CNN no, así que sin normalizar las curvas de los tres
    modelos no se pueden poner en el mismo eje.
    """
    _escribir(tmp_path, "rtdetr_l", _results_ultralytics())
    _escribir(tmp_path, "fasterrcnn_r50fpn", _results_torchvision())

    df = load_training_histories(tmp_path)

    ultralytics = df[df["experiment"] == "rtdetr_l"].reset_index(drop=True)
    torchvision = df[df["experiment"] == "fasterrcnn_r50fpn"].reset_index(drop=True)

    metricas = ["mAP50", "mAP50_95", "precision", "recall"]
    pd.testing.assert_frame_equal(ultralytics[metricas], torchvision[metricas])


def test_devuelve_las_columnas_del_spec_y_una_fila_por_epoca(tmp_path):
    _escribir(tmp_path, "yolov8n_baseline", _results_ultralytics(epocas=4))

    df = load_training_histories(tmp_path)

    assert list(df.columns) == HISTORY_COLUMNS
    assert len(df) == 4
    assert df["experiment"].unique().tolist() == ["yolov8n_baseline"]
    assert df["epoch"].tolist() == [1, 2, 3, 4]


def test_junta_los_experimentos_ordenados_por_nombre(tmp_path):
    _escribir(tmp_path, "rtdetr_l", _results_ultralytics())
    _escribir(tmp_path, "fasterrcnn_r50fpn", _results_torchvision())
    _escribir(tmp_path, "yolov8n_baseline", _results_ultralytics())

    df = load_training_histories(tmp_path)

    assert df["experiment"].unique().tolist() == [
        "fasterrcnn_r50fpn",
        "rtdetr_l",
        "yolov8n_baseline",
    ]
    assert len(df) == 9


def test_saltea_el_experimento_al_que_le_falta_una_metrica(tmp_path):
    """Un results.csv incompleto no puede llevarse puesta la figura entera.

    Pasa de verdad: una corrida que murió antes de cerrar la primera época deja
    el archivo con las columnas de pérdida y sin las de métricas.
    """
    _escribir(tmp_path, "completo", _results_ultralytics())

    incompleto = _results_ultralytics().drop(columns=["metrics/mAP50-95(B)"])
    _escribir(tmp_path, "incompleto", incompleto)

    df = load_training_histories(tmp_path)

    assert df["experiment"].unique().tolist() == ["completo"]


def test_sin_experimentos_devuelve_df_vacio_con_columnas(tmp_path):
    """Vacío pero tipado: el notebook chequea `df.empty`, no un TypeError."""
    df = load_training_histories(tmp_path)

    assert df.empty
    assert list(df.columns) == HISTORY_COLUMNS


def test_ignora_las_carpetas_sin_results_csv(tmp_path):
    _escribir(tmp_path, "con_historial", _results_ultralytics())
    (tmp_path / "eda").mkdir()
    (tmp_path / "eda" / "split_summary.csv").write_text("split,imagenes\ntrain,10\n")

    df = load_training_histories(tmp_path)

    assert df["experiment"].unique().tolist() == ["con_historial"]
