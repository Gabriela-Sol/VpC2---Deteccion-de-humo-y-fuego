"""Construcción de detectores Faster R-CNN sobre torchvision."""

from __future__ import annotations

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FastRCNNPredictor,
)

# nombre -> (constructor, enum de pesos COCO)
AVAILABLE_BACKBONES = {
    "resnet50_fpn_v2": (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    ),
    "mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    ),
}


def build_fasterrcnn(
    num_classes: int = 3,
    backbone: str = "resnet50_fpn_v2",
    trainable_backbone_layers: int = 3,
    min_size: int = 640,
    max_size: int = 1024,
    pretrained: bool = True,
) -> FasterRCNN:
    """Arma un Faster R-CNN con el cabezal ajustado a `num_classes`.

    `num_classes` incluye el fondo: para smoke y fire el valor es 3.
    Con `pretrained=False` no se descarga ningún peso, que es lo que necesitan
    los tests para correr sin red.
    """
    if backbone not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Backbone desconocido: {backbone!r}. "
            f"Opciones válidas: {sorted(AVAILABLE_BACKBONES)}"
        )

    builder, weights_enum = AVAILABLE_BACKBONES[backbone]

    model = builder(
        weights=weights_enum.DEFAULT if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_parameters(model) -> int:
    """Cantidad de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
