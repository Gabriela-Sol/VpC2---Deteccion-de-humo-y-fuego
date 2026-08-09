"""Tests de las primitivas de emparejamiento por IoU.

Las cajas están elegidas para que los IoU sean fáciles de verificar a mano.
"""

import numpy as np
import torch

from src.engine.matching import confusion_matrix, match_dataset


def _pred(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _gt(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_deteccion_perfecta_es_verdadero_positivo():
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
    )
    assert resultado.true_positive.tolist() == [True]
    assert resultado.n_ground_truth == {1: 1}


def test_iou_por_debajo_del_umbral_es_falso_positivo():
    # Solapamiento de 2x10 sobre una unión de 18x10 -> IoU = 0.111
    resultado = match_dataset(
        [_pred([[8, 0, 18, 10]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        iou_threshold=0.5,
    )
    assert resultado.true_positive.tolist() == [False]


def test_clase_distinta_no_empareja():
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10]], [0.9], [2])],
        [_gt([[0, 0, 10, 10]], [1])],
    )
    assert resultado.true_positive.tolist() == [False]
    assert resultado.n_ground_truth == {1: 1}


def test_solo_la_deteccion_de_mayor_score_se_queda_con_el_gt():
    # Dos detecciones sobre un único objeto: la segunda es un duplicado.
    # Los scores son 0.5 y 0.75 porque ambos son exactos en binario; con 0.6 y
    # 0.9 la comparación por igualdad fallaría contra el redondeo de float32.
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10], [0, 0, 10, 10]], [0.5, 0.75], [1, 1])],
        [_gt([[0, 0, 10, 10]], [1])],
    )
    # El resultado viene ordenado por score descendente.
    assert resultado.scores.tolist() == [0.75, 0.5]
    assert resultado.true_positive.tolist() == [True, False]


def test_imagen_negativa_sin_detecciones_no_aporta_nada():
    resultado = match_dataset(
        [_pred(np.zeros((0, 4)), [], [])],
        [_gt(np.zeros((0, 4)), [])],
    )
    assert resultado.scores.size == 0
    assert resultado.n_ground_truth == {}


def test_deteccion_sobre_imagen_negativa_es_falso_positivo():
    resultado = match_dataset(
        [_pred([[0, 0, 10, 10]], [0.8], [1])],
        [_gt(np.zeros((0, 4)), [])],
    )
    assert resultado.true_positive.tolist() == [False]
    assert resultado.n_ground_truth == {}


def test_cuenta_ground_truth_por_clase_en_varias_imagenes():
    resultado = match_dataset(
        [
            _pred(np.zeros((0, 4)), [], []),
            _pred(np.zeros((0, 4)), [], []),
        ],
        [
            _gt([[0, 0, 10, 10], [20, 20, 30, 30]], [1, 2]),
            _gt([[0, 0, 10, 10]], [1]),
        ],
    )
    assert resultado.n_ground_truth == {1: 2, 2: 1}


def test_matriz_de_confusion_acierto_va_a_la_diagonal():
    matriz = confusion_matrix(
        [_pred([[0, 0, 10, 10]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
    )
    assert matriz.shape == (3, 3)
    assert matriz[0, 0] == 1
    assert matriz.sum() == 1


def test_matriz_de_confusion_registra_confusion_entre_clases():
    # Emparejamiento agnóstico de clase: se detecta fire donde había smoke.
    matriz = confusion_matrix(
        [_pred([[0, 0, 10, 10]], [0.9], [2])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
    )
    # fila = predicho (fire -> 1), columna = real (smoke -> 0)
    assert matriz[1, 0] == 1
    assert matriz.sum() == 1


def test_matriz_de_confusion_falso_positivo_va_a_la_columna_de_fondo():
    matriz = confusion_matrix(
        [_pred([[100, 100, 110, 110]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
    )
    assert matriz[0, 2] == 1  # predicho smoke, no había nada
    assert matriz[2, 0] == 1  # había smoke, no se detectó
    assert matriz.sum() == 2


def test_matriz_de_confusion_ignora_detecciones_bajo_el_umbral_de_confianza():
    matriz = confusion_matrix(
        [_pred([[0, 0, 10, 10]], [0.1], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
        label_order=[1, 2],
        conf_threshold=0.25,
    )
    # La detección se descarta, así que el objeto queda sin detectar.
    assert matriz[2, 0] == 1
    assert matriz.sum() == 1
