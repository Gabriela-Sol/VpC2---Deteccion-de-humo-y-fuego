"""Métricas de detección: mAP con protocolo COCO, curvas y velocidad.

El mAP sale de torchmetrics, que usa pycocotools por debajo, el mismo criterio
que aplica Ultralytics. Las curvas de precisión, recall y F1 en función de la
confianza se calculan acá a partir del emparejamiento de `matching.py`, porque
torchmetrics no las expone.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from src.data.yolo_dataset import LABEL_ORDER
from src.engine.matching import match_dataset

# Detecciones por debajo de este score no aportan a ninguna métrica y sí
# encarecen el cálculo. Coincide a propósito con el `box_score_thresh` con el
# que `build_fasterrcnn` arma el detector y con el `conf=0.001` que usa
# Ultralytics al validar: es la misma cota para los tres modelos.
MIN_SCORE = 1e-3


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Corre el modelo sobre todo el loader y devuelve predicciones y targets en CPU."""
    model.eval()
    model.to(device)

    predictions: list[dict] = []
    targets: list[dict] = []

    for images, batch_targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for output, target in zip(outputs, batch_targets):
            conservar = output["scores"].detach().cpu() >= MIN_SCORE
            predictions.append(
                {
                    "boxes": output["boxes"].detach().cpu()[conservar],
                    "scores": output["scores"].detach().cpu()[conservar],
                    "labels": output["labels"].detach().cpu()[conservar],
                }
            )
            targets.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

    return predictions, targets


def _per_class_dict(resultado, label_order: list[int]) -> dict[int, float]:
    """Convierte `map_per_class` de torchmetrics en un dict label -> valor.

    Las clases sin ningún objeto real quedan como NaN en lugar del -1 que
    devuelve pycocotools, para que promedios y gráficos las ignoren solos.
    """
    clases = np.atleast_1d(resultado["classes"].cpu().numpy())
    valores = np.atleast_1d(resultado["map_per_class"].cpu().numpy())
    presentes = {int(c): float(v) for c, v in zip(clases, valores)}

    salida: dict[int, float] = {}
    for label in label_order:
        valor = presentes.get(label, -1.0)
        salida[label] = float("nan") if valor < 0 else valor
    return salida


def compute_map(
    predictions: list[dict],
    targets: list[dict],
    label_order: list[int] = LABEL_ORDER,
) -> dict:
    """mAP50 y mAP50-95, globales y por clase.

    Se instancian dos métricas porque torchmetrics expone `map_per_class` solo
    para el promedio sobre los umbrales configurados: para obtener el mAP50 por
    clase hay que restringir `iou_thresholds` a [0.5].
    """
    metrica_completa = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True
    )
    metrica_completa.update(predictions, targets)
    resultado_completo = metrica_completa.compute()

    metrica_50 = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=True
    )
    metrica_50.update(predictions, targets)
    resultado_50 = metrica_50.compute()

    def _escalar(valor) -> float:
        numero = float(valor)
        return float("nan") if numero < 0 else numero

    return {
        "map50": _escalar(resultado_completo["map_50"]),
        "map50_95": _escalar(resultado_completo["map"]),
        "map50_per_class": _per_class_dict(resultado_50, label_order),
        "map50_95_per_class": _per_class_dict(resultado_completo, label_order),
    }


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP en 101 puntos con el criterio de COCO.

    COCO no interpola linealmente entre puntos de recall: para cada umbral toma
    la precisión envolvente del primer recall que lo alcanza, o sea una función
    escalonada. Interpolar linealmente suaviza las caídas de precisión y
    sobreestima el AP, y además daría un número que no coincide con el
    `map50_per_class` que devuelve torchmetrics para la misma clase.
    """
    if recall.size == 0:
        return 0.0

    # La precisión se vuelve monótona decreciente de derecha a izquierda.
    precision_envolvente = np.maximum.accumulate(precision[::-1])[::-1]
    puntos = np.linspace(0.0, 1.0, 101)

    indices = np.searchsorted(recall, puntos, side="left")
    valores = np.zeros(puntos.size)
    alcanzables = indices < precision_envolvente.size
    valores[alcanzables] = precision_envolvente[indices[alcanzables]]

    return float(valores.mean())


def compute_curves(
    predictions: list[dict],
    targets: list[dict],
    label_order: list[int] = LABEL_ORDER,
    iou_threshold: float = 0.5,
    n_thresholds: int = 101,
) -> dict:
    """Curvas P, R y F1 contra confianza, más la curva PR por clase.

    La precisión y el recall se promedian **macro**: se calculan por clase y
    después se promedian entre las clases que tienen algún objeto real. Es lo
    que reporta Ultralytics (promedio de sus vectores por clase), y es la
    librería que produjo el baseline ya commiteado. Agrupar todas las
    detecciones (micro) daría números distintos cuando las clases tienen
    cantidades de cajas muy diferentes, como pasa entre smoke y fire en D-Fire.
    """
    match = match_dataset(predictions, targets, iou_threshold=iou_threshold)

    confidence = np.linspace(0.0, 1.0, n_thresholds)
    precision = np.zeros(n_thresholds)
    recall = np.zeros(n_thresholds)
    f1 = np.zeros(n_thresholds)

    clases_medibles = [
        label for label in label_order if match.n_ground_truth.get(label, 0) > 0
    ]

    for posicion, umbral in enumerate(confidence):
        seleccion = match.scores >= umbral

        precisiones, recalls = [], []
        for label in clases_medibles:
            de_la_clase = seleccion & (match.labels == label)
            verdaderos = int(match.true_positive[de_la_clase].sum())
            detectadas = int(de_la_clase.sum())

            precisiones.append(verdaderos / detectadas if detectadas else 0.0)
            recalls.append(verdaderos / match.n_ground_truth[label])

        p = float(np.mean(precisiones)) if precisiones else 0.0
        r = float(np.mean(recalls)) if recalls else 0.0

        precision[posicion] = p
        recall[posicion] = r
        f1[posicion] = 2 * p * r / (p + r) if (p + r) else 0.0

    mejor = int(np.argmax(f1))

    pr_per_class: dict[int, dict] = {}
    for label in label_order:
        de_la_clase = match.labels == label
        n_gt = match.n_ground_truth.get(label, 0)

        tp_acumulado = np.cumsum(match.true_positive[de_la_clase])
        cantidad = np.arange(1, tp_acumulado.size + 1)

        recall_clase = tp_acumulado / n_gt if n_gt else np.zeros_like(tp_acumulado, dtype=float)
        precision_clase = tp_acumulado / cantidad if cantidad.size else np.zeros(0)

        pr_per_class[label] = {
            "recall": np.asarray(recall_clase, dtype=float),
            "precision": np.asarray(precision_clase, dtype=float),
            "ap": _average_precision(
                np.asarray(recall_clase, dtype=float),
                np.asarray(precision_clase, dtype=float),
            )
            if n_gt
            else float("nan"),
        }

    return {
        "confidence": confidence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_confidence": float(confidence[mejor]),
        "best_f1": float(f1[mejor]),
        "best_precision": float(precision[mejor]),
        "best_recall": float(recall[mejor]),
        "pr_per_class": pr_per_class,
    }


@torch.no_grad()
def measure_inference_fps(model, dataset, device, num_images: int = 50, warmup: int = 5) -> float:
    """Imágenes por segundo en inferencia, de a una imagen por vez.

    Las primeras `warmup` pasadas se descartan: la primera inferencia paga la
    inicialización de kernels y no representa el régimen estacionario.

    La decodificación de los JPEG, la lectura de disco y la transferencia al
    device quedan deliberadamente FUERA de la medición: los tensores se
    materializan antes de arrancar el cronómetro. Ultralytics reporta
    `speed["inference"] + speed["postprocess"]`, que tampoco incluye el
    preproceso, así que medir el disco acá haría incomparables las dos
    columnas de FPS.
    """
    model.eval()
    model.to(device)

    total = min(num_images, len(dataset))
    if total == 0:
        return 0.0

    imagenes = [dataset[indice][0].to(device) for indice in range(total)]

    for imagen in imagenes[: min(warmup, total)]:
        model([imagen])

    if device.type == "cuda":
        torch.cuda.synchronize()

    inicio = time.perf_counter()
    for imagen in imagenes:
        model([imagen])
    if device.type == "cuda":
        torch.cuda.synchronize()
    transcurrido = time.perf_counter() - inicio

    return total / transcurrido if transcurrido > 0 else 0.0
