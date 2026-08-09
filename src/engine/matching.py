"""Emparejamiento de detecciones con ground truth por IoU.

Son funciones puras sobre listas de predicciones y targets. De acá salen tanto
las curvas de precisión y recall como la matriz de confusión, así que el
criterio de emparejamiento queda definido en un solo lugar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision.ops import box_iou


@dataclass
class MatchResult:
    """Detecciones de todo el dataset ordenadas por score descendente."""

    scores: np.ndarray
    true_positive: np.ndarray
    labels: np.ndarray
    n_ground_truth: dict[int, int] = field(default_factory=dict)


def _match_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Empareja de forma golosa, por score descendente, exigiendo misma clase.

    Devuelve un tensor booleano alineado con las detecciones ya ordenadas.
    Cada objeto real puede consumir una sola detección: las siguientes que lo
    solapen cuentan como falsos positivos, que es lo que penaliza los duplicados.
    """
    true_positive = torch.zeros(len(pred_scores), dtype=torch.bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return true_positive

    ious = box_iou(pred_boxes, gt_boxes)
    gt_taken = torch.zeros(len(gt_boxes), dtype=torch.bool)

    for index in range(len(pred_boxes)):
        elegibles = (gt_labels == pred_labels[index]) & (~gt_taken)
        if not bool(elegibles.any()):
            continue

        candidatos = ious[index].clone()
        candidatos[~elegibles] = -1.0
        mejor = int(torch.argmax(candidatos))

        if float(candidatos[mejor]) >= iou_threshold:
            true_positive[index] = True
            gt_taken[mejor] = True

    return true_positive


def match_dataset(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Empareja todas las imágenes y concatena el resultado."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions y targets deben estar alineados: "
            f"{len(predictions)} != {len(targets)}"
        )

    scores_por_imagen: list[torch.Tensor] = []
    tp_por_imagen: list[torch.Tensor] = []
    labels_por_imagen: list[torch.Tensor] = []
    n_ground_truth: dict[int, int] = {}

    for prediction, target in zip(predictions, targets):
        gt_labels = target["labels"].detach().cpu()
        for label in gt_labels.tolist():
            n_ground_truth[label] = n_ground_truth.get(label, 0) + 1

        pred_scores = prediction["scores"].detach().cpu()
        orden = torch.argsort(pred_scores, descending=True)

        pred_boxes = prediction["boxes"].detach().cpu()[orden]
        pred_scores = pred_scores[orden]
        pred_labels = prediction["labels"].detach().cpu()[orden]

        tp_por_imagen.append(
            _match_image(
                pred_boxes,
                pred_scores,
                pred_labels,
                target["boxes"].detach().cpu(),
                gt_labels,
                iou_threshold,
            )
        )
        scores_por_imagen.append(pred_scores)
        labels_por_imagen.append(pred_labels)

    scores = torch.cat(scores_por_imagen) if scores_por_imagen else torch.zeros(0)
    true_positive = torch.cat(tp_por_imagen) if tp_por_imagen else torch.zeros(0, dtype=torch.bool)
    labels = torch.cat(labels_por_imagen) if labels_por_imagen else torch.zeros(0, dtype=torch.int64)

    orden_global = torch.argsort(scores, descending=True)

    return MatchResult(
        scores=scores[orden_global].numpy(),
        true_positive=true_positive[orden_global].numpy(),
        labels=labels[orden_global].numpy(),
        n_ground_truth=n_ground_truth,
    )


def confusion_matrix(
    predictions: list[dict],
    targets: list[dict],
    label_order: list[int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """Matriz `[predicho, real]` con una fila y columna extra para el fondo.

    El emparejamiento es agnóstico de clase, igual que en Ultralytics: así una
    detección con la caja correcta pero la clase equivocada cae fuera de la
    diagonal en vez de contarse como un falso positivo más un falso negativo.
    """
    n_clases = len(label_order)
    fondo = n_clases
    indice_de_label = {label: posicion for posicion, label in enumerate(label_order)}
    matriz = np.zeros((n_clases + 1, n_clases + 1), dtype=np.int64)

    for prediction, target in zip(predictions, targets):
        scores = prediction["scores"].detach().cpu()
        conservar = scores >= conf_threshold

        pred_boxes = prediction["boxes"].detach().cpu()[conservar]
        pred_labels = prediction["labels"].detach().cpu()[conservar]
        scores = scores[conservar]

        orden = torch.argsort(scores, descending=True)
        pred_boxes, pred_labels = pred_boxes[orden], pred_labels[orden]

        gt_boxes = target["boxes"].detach().cpu()
        gt_labels = target["labels"].detach().cpu()

        gt_taken = torch.zeros(len(gt_boxes), dtype=torch.bool)
        pred_taken = torch.zeros(len(pred_boxes), dtype=torch.bool)

        if len(pred_boxes) and len(gt_boxes):
            ious = box_iou(pred_boxes, gt_boxes)
            for index in range(len(pred_boxes)):
                candidatos = ious[index].clone()
                candidatos[gt_taken] = -1.0
                mejor = int(torch.argmax(candidatos))
                if float(candidatos[mejor]) >= iou_threshold:
                    matriz[
                        indice_de_label[int(pred_labels[index])],
                        indice_de_label[int(gt_labels[mejor])],
                    ] += 1
                    gt_taken[mejor] = True
                    pred_taken[index] = True

        for index in range(len(pred_boxes)):
            if not bool(pred_taken[index]):
                matriz[indice_de_label[int(pred_labels[index])], fondo] += 1

        for index in range(len(gt_boxes)):
            if not bool(gt_taken[index]):
                matriz[fondo, indice_de_label[int(gt_labels[index])]] += 1

    return matriz
