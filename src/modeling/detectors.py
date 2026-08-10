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
    box_score_thresh: float = 0.001,
    box_detections_per_img: int = 300,
) -> FasterRCNN:
    """Arma un Faster R-CNN con el cabezal ajustado a `num_classes`.

    `num_classes` incluye el fondo: para smoke y fire el valor es 3.
    Con `pretrained=False` no se descarga ningún peso, que es lo que necesitan
    los tests para correr sin red.

    `box_score_thresh` y `box_detections_per_img` NO usan los valores por
    defecto de torchvision (0.05 y 100) sino los que Ultralytics aplica al
    validar (`conf=0.001`, `max_det=300`). El motivo es la comparabilidad: el
    filtrado ocurre dentro del modelo, antes de que `collect_predictions` vea
    nada, así que con los defaults de torchvision la cola de baja confianza de
    Faster R-CNN se descartaría mientras que la de YOLOv8 y RT-DETR sobrevive.
    Como el mAP integra toda la curva precisión-recall, eso sería una
    penalización sistemática sobre uno solo de los tres modelos, por una razón
    que no tiene que ver con el modelo.
    """
    if backbone not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Backbone desconocido: {backbone!r}. "
            f"Opciones válidas: {sorted(AVAILABLE_BACKBONES)}"
        )

    builder, weights_enum = AVAILABLE_BACKBONES[backbone]

    # torchvision avisa (UserWarning) que `trainable_backbone_layers` no tiene
    # efecto cuando no hay pesos preentrenados: sin pesos congelables la opción
    # es ruido. Se pasa solo cuando corresponde para no emitir ese warning en
    # los tests, que construyen todo con pretrained=False.
    extra = (
        {"trainable_backbone_layers": trainable_backbone_layers} if pretrained else {}
    )

    model = builder(
        weights=weights_enum.DEFAULT if pretrained else None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
        box_detections_per_img=box_detections_per_img,
        **extra,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_parameters(model) -> int:
    """Cantidad de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
