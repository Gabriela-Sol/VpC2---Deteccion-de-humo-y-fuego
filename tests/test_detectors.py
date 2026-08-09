"""Tests del constructor de Faster R-CNN. Corren sin red: pretrained=False."""

import pytest
import torch

from src.modeling.detectors import (
    AVAILABLE_BACKBONES,
    build_fasterrcnn,
    count_parameters,
)


def test_backbones_disponibles():
    assert set(AVAILABLE_BACKBONES) == {"resnet50_fpn_v2", "mobilenet_v3_large_fpn"}


def test_cabezal_tiene_la_cantidad_de_clases_pedida():
    model = build_fasterrcnn(
        num_classes=3, backbone="mobilenet_v3_large_fpn", pretrained=False
    )
    assert model.roi_heads.box_predictor.cls_score.out_features == 3
    # 4 coordenadas por clase.
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 12


def test_min_size_y_max_size_llegan_al_transform():
    model = build_fasterrcnn(
        backbone="mobilenet_v3_large_fpn", min_size=640, max_size=1024, pretrained=False
    )
    assert model.transform.min_size == (640,)
    assert model.transform.max_size == 1024


def test_backbone_desconocido_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="resnet50_fpn_v2"):
        build_fasterrcnn(backbone="inexistente", pretrained=False)


def test_count_parameters_cuenta_solo_entrenables():
    model = build_fasterrcnn(backbone="mobilenet_v3_large_fpn", pretrained=False)
    total = count_parameters(model)
    assert total > 1_000_000

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert count_parameters(model) == 0


def test_forward_de_entrenamiento_acepta_un_target_vacio():
    model = build_fasterrcnn(
        num_classes=3, backbone="mobilenet_v3_large_fpn", min_size=64, max_size=128,
        pretrained=False,
    )
    model.train()

    images = [torch.rand(3, 80, 100), torch.rand(3, 80, 100)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 70.0]]),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        },
    ]

    losses = model(images, targets)
    assert set(losses) == {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total.detach())


def test_forward_de_inferencia_devuelve_boxes_scores_labels():
    model = build_fasterrcnn(
        backbone="mobilenet_v3_large_fpn", min_size=64, max_size=128, pretrained=False
    )
    model.eval()

    with torch.no_grad():
        salidas = model([torch.rand(3, 80, 100)])

    assert sorted(salidas[0]) == ["boxes", "labels", "scores"]
    assert salidas[0]["boxes"].shape[1] == 4
