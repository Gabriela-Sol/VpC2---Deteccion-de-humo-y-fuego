"""Tests del bucle de entrenamiento y de la persistencia de checkpoints."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.yolo_dataset import YoloDetectionDataset, collate_fn
from src.engine.trainer import (
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from src.modeling.detectors import build_fasterrcnn

CONFIG_BASE = {
    "optimizer": "sgd",
    "lr0": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler": "cosine",
}


def _modelo():
    return build_fasterrcnn(
        num_classes=3, backbone="mobilenet_v3_large_fpn",
        min_size=64, max_size=128, pretrained=False,
    )


def test_build_optimizer_sgd_usa_los_hiperparametros():
    optimizer = build_optimizer(_modelo(), CONFIG_BASE)

    assert isinstance(optimizer, torch.optim.SGD)
    grupo = optimizer.param_groups[0]
    assert grupo["lr"] == 0.005
    assert grupo["momentum"] == 0.9
    assert grupo["weight_decay"] == 0.0005


def test_build_optimizer_adamw():
    optimizer = build_optimizer(_modelo(), {**CONFIG_BASE, "optimizer": "adamw"})
    assert isinstance(optimizer, torch.optim.AdamW)


def test_build_scheduler_cosine_baja_el_lr():
    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    scheduler = build_scheduler(optimizer, CONFIG_BASE, epochs=10)

    lr_inicial = optimizer.param_groups[0]["lr"]
    for _ in range(5):
        # optimizer.step() antes de scheduler.step(), el mismo orden que usa
        # train_one_epoch. Al revés, PyTorch avisa que se saltea el primer valor
        # del schedule. Sin gradientes el paso del optimizador no mueve nada.
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] < lr_inicial


def test_build_scheduler_none_devuelve_none():
    optimizer = build_optimizer(_modelo(), CONFIG_BASE)
    assert build_scheduler(optimizer, {**CONFIG_BASE, "lr_scheduler": "none"}, 10) is None


def test_build_optimizer_rechaza_un_nombre_desconocido():
    with pytest.raises(ValueError, match="sgd"):
        build_optimizer(_modelo(), {**CONFIG_BASE, "optimizer": "inventado"})


def test_build_scheduler_rechaza_un_nombre_desconocido():
    optimizer = build_optimizer(_modelo(), CONFIG_BASE)
    with pytest.raises(ValueError, match="cosine"):
        build_scheduler(optimizer, {**CONFIG_BASE, "lr_scheduler": "inventado"}, 10)


def test_train_one_epoch_devuelve_las_perdidas_promedio(synthetic_dataset):
    dataset = YoloDetectionDataset(synthetic_dataset / "train", train=True, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)

    perdidas = train_one_epoch(
        modelo, optimizer, loader, torch.device("cpu"), max_batches=2
    )

    assert set(perdidas) == {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_total",
    }
    for valor in perdidas.values():
        assert isinstance(valor, float)
        assert valor == valor  # descarta NaN


def test_train_one_epoch_modifica_los_pesos(synthetic_dataset):
    dataset = YoloDetectionDataset(synthetic_dataset / "train", train=True, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    modelo = _modelo()
    antes = modelo.roi_heads.box_predictor.cls_score.weight.detach().clone()

    optimizer = build_optimizer(modelo, CONFIG_BASE)
    train_one_epoch(modelo, optimizer, loader, torch.device("cpu"), max_batches=2)

    despues = modelo.roi_heads.box_predictor.cls_score.weight.detach()
    assert not torch.allclose(antes, despues)


def test_checkpoint_ida_y_vuelta(tmp_path):
    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    scheduler = build_scheduler(optimizer, CONFIG_BASE, epochs=10)
    historial = [{"epoch": 1, "loss_total": 2.5}]

    ruta = tmp_path / "checkpoint.pth"
    save_checkpoint(ruta, modelo, optimizer, scheduler, epoch=1, history=historial)
    assert ruta.exists()

    modelo_nuevo = _modelo()
    optimizer_nuevo = build_optimizer(modelo_nuevo, CONFIG_BASE)
    scheduler_nuevo = build_scheduler(optimizer_nuevo, CONFIG_BASE, epochs=10)

    epoca, historial_recuperado = load_checkpoint(
        ruta, modelo_nuevo, optimizer_nuevo, scheduler_nuevo, torch.device("cpu")
    )

    assert epoca == 1
    assert historial_recuperado == historial
    assert torch.allclose(
        modelo.roi_heads.box_predictor.cls_score.weight,
        modelo_nuevo.roi_heads.box_predictor.cls_score.weight,
    )


def test_checkpoint_restaura_estado_ya_acumulado(synthetic_dataset, tmp_path):
    # El test anterior guarda un optimizador recién construido, cuyo estado está
    # vacío: pasaría igual si load_checkpoint no restaurara nada. Este entrena
    # primero para que haya momentum y el scheduler haya avanzado, que es lo que
    # de verdad tiene que sobrevivir cuando una corrida de 6 horas se reanuda.
    dataset = YoloDetectionDataset(synthetic_dataset / "train", train=True, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    scheduler = build_scheduler(optimizer, CONFIG_BASE, epochs=10)

    train_one_epoch(modelo, optimizer, loader, torch.device("cpu"), max_batches=2)
    scheduler.step()
    scheduler.step()

    estado_previo = optimizer.state_dict()["state"]
    assert estado_previo, "el optimizador debería haber acumulado momentum"
    lr_previo = optimizer.param_groups[0]["lr"]
    last_epoch_previo = scheduler.last_epoch

    ruta = tmp_path / "checkpoint.pth"
    save_checkpoint(ruta, modelo, optimizer, scheduler, epoch=1, history=[])

    modelo_nuevo = _modelo()
    optimizer_nuevo = build_optimizer(modelo_nuevo, CONFIG_BASE)
    scheduler_nuevo = build_scheduler(optimizer_nuevo, CONFIG_BASE, epochs=10)
    load_checkpoint(
        ruta, modelo_nuevo, optimizer_nuevo, scheduler_nuevo, torch.device("cpu")
    )

    assert scheduler_nuevo.last_epoch == last_epoch_previo
    assert optimizer_nuevo.param_groups[0]["lr"] == pytest.approx(lr_previo)

    estado_nuevo = optimizer_nuevo.state_dict()["state"]
    assert set(estado_nuevo) == set(estado_previo)
    for clave, valores in estado_previo.items():
        assert torch.allclose(
            valores["momentum_buffer"], estado_nuevo[clave]["momentum_buffer"]
        )


def test_load_checkpoint_sin_optimizador_tambien_funciona(tmp_path):
    modelo = _modelo()
    optimizer = build_optimizer(modelo, CONFIG_BASE)
    ruta = tmp_path / "checkpoint.pth"
    save_checkpoint(ruta, modelo, optimizer, None, epoch=3, history=[])

    epoca, historial = load_checkpoint(ruta, _modelo())
    assert epoca == 3
    assert historial == []
