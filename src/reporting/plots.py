"""Figuras de los experimentos, con los mismos nombres de archivo que Ultralytics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend sin ventana: los notebooks corren en Colab y los tests sin display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.yolo_dataset import CLASS_NAMES  # noqa: E402

METRIC_LABELS = {
    "precision": "Precisión",
    "recall": "Recall",
    "f1": "F1",
}


def _guardar(fig, out_png) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def normalize_by_column(matrix) -> np.ndarray:
    """Normaliza cada columna de la matriz de confusión a suma 1.

    Es una función aparte y no código embebido en el gráfico porque es la única
    parte con aritmética: así se puede testear directamente. Una columna que
    suma cero —el caso normal de la columna de fondo— queda en cero, no en NaN.
    """
    matriz = np.asarray(matrix, dtype=float)
    sumas = matriz.sum(axis=0, keepdims=True)
    return np.divide(matriz, sumas, out=np.zeros_like(matriz), where=sumas > 0)


def mean_average_precision(pr_per_class: dict) -> float:
    """Promedia los AP por clase ignorando las clases sin ningún objeto real.

    `compute_map` devuelve NaN para esas clases; incluirlas en el promedio lo
    volvería NaN entero y el número desaparecería del gráfico sin explicación.
    """
    aps = [
        datos["ap"] for datos in pr_per_class.values() if not np.isnan(datos["ap"])
    ]
    return float(np.mean(aps)) if aps else float("nan")


def plot_results_csv(results_csv, out_png) -> Path:
    """Panel con las pérdidas de entrenamiento y las métricas de validación."""
    df = pd.read_csv(results_csv)
    columnas = [columna for columna in df.columns if columna != "epoch"]

    filas = 2
    columnas_grafico = int(np.ceil(len(columnas) / filas))
    fig, axes = plt.subplots(
        filas, columnas_grafico, figsize=(3.2 * columnas_grafico, 6), squeeze=False
    )

    for posicion, columna in enumerate(columnas):
        ax = axes[posicion // columnas_grafico][posicion % columnas_grafico]
        ax.plot(df["epoch"], df[columna], marker="o", markersize=3, linewidth=1.5)
        ax.set_title(columna, fontsize=9)
        ax.set_xlabel("época", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for posicion in range(len(columnas), filas * columnas_grafico):
        axes[posicion // columnas_grafico][posicion % columnas_grafico].axis("off")

    fig.suptitle("Entrenamiento y validación por época", fontsize=11)
    return _guardar(fig, out_png)


def plot_confusion_matrix(matrix, class_names, out_png, normalize: bool = False) -> Path:
    """Matriz `[predicho, real]`; normalizada, cada columna suma 1."""
    matriz = normalize_by_column(matrix) if normalize else np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    imagen = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.set_title("Matriz de confusión" + (" normalizada" if normalize else ""))

    umbral = matriz.max() / 2 if matriz.max() > 0 else 0.5
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            valor = matriz[fila, columna]
            ax.text(
                columna, fila,
                f"{valor:.2f}" if normalize else f"{int(valor)}",
                ha="center", va="center", fontsize=9,
                color="white" if valor > umbral else "black",
            )

    return _guardar(fig, out_png)


def plot_pr_curve(curves: dict, out_png, class_names: dict = CLASS_NAMES) -> Path:
    """Curva precisión-recall por clase, con el AP en la leyenda."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, datos in curves["pr_per_class"].items():
        nombre = class_names.get(label, str(label))
        ax.plot(
            datos["recall"],
            datos["precision"],
            linewidth=1.8,
            label=f"{nombre} (AP={datos['ap']:.3f})",
        )

    promedio = mean_average_precision(curves["pr_per_class"])
    if not np.isnan(promedio):
        ax.plot([], [], " ", label=f"mAP@0.5 = {promedio:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Curva precisión-recall (IoU 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    return _guardar(fig, out_png)


def plot_metric_vs_confidence(
    curves: dict, metric: str, out_png, class_names: dict = CLASS_NAMES
) -> Path:
    """Precisión, recall o F1 en función del umbral de confianza."""
    if metric not in METRIC_LABELS:
        raise ValueError(
            f"Métrica desconocida: {metric!r}. Opciones: {sorted(METRIC_LABELS)}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curves["confidence"], curves[metric], linewidth=2, color="tab:blue")

    if metric == "f1":
        ax.axvline(
            curves["best_confidence"], linestyle="--", color="tab:red", linewidth=1.2,
            label=f"mejor F1 = {curves['best_f1']:.3f} @ conf {curves['best_confidence']:.2f}",
        )
        ax.legend(loc="lower center", fontsize=9)

    ax.set_xlabel("Confianza")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{METRIC_LABELS[metric]} en función de la confianza")
    ax.grid(alpha=0.3)

    return _guardar(fig, out_png)


def plot_model_comparison(df: pd.DataFrame, out_dir) -> list[Path]:
    """Tres figuras comparativas a partir de los metrics_summary.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []

    etiquetas = df["experiment"].tolist()
    posiciones = np.arange(len(df))
    ancho = 0.38

    # 1) mAP50 y mAP50-95 por modelo
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(posiciones - ancho / 2, df["mAP50"], ancho, label="mAP@0.5")
    ax.bar(posiciones + ancho / 2, df["mAP50_95"], ancho, label="mAP@0.5:0.95")
    for posicion, (a, b) in enumerate(zip(df["mAP50"], df["mAP50_95"])):
        ax.text(posicion - ancho / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(posicion + ancho / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(posiciones, etiquetas, rotation=15, ha="right")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.05)
    ax.set_title("Desempeño por modelo (split de validación)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    rutas.append(_guardar(fig, out_dir / "map_por_modelo.png"))

    # 2) mAP50 por clase
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(posiciones - ancho / 2, df["mAP50_smoke"], ancho, label="smoke")
    ax.bar(posiciones + ancho / 2, df["mAP50_fire"], ancho, label="fire")
    ax.set_xticks(posiciones, etiquetas, rotation=15, ha="right")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("mAP@0.5 por clase")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    rutas.append(_guardar(fig, out_dir / "map_por_clase.png"))

    # 3) precisión contra velocidad
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["fps"], df["mAP50_95"], s=140, c=posiciones, cmap="viridis", zorder=3)
    for _, fila in df.iterrows():
        ax.annotate(
            fila["experiment"],
            (fila["fps"], fila["mAP50_95"]),
            textcoords="offset points", xytext=(8, 6), fontsize=9,
        )
    ax.set_xlabel("FPS en inferencia")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Compromiso entre precisión y velocidad")
    ax.grid(alpha=0.3)
    rutas.append(_guardar(fig, out_dir / "precision_vs_velocidad.png"))

    return rutas
