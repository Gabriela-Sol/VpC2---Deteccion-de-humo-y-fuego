"""Bucle de entrenamiento para detectores de torchvision.

Los checkpoints guardan modelo, optimizador, scheduler, época e historial. Un
entrenamiento de Faster R-CNN sobre D-Fire supera la duración de una sesión de
Colab, así que reanudar tiene que ser el caso normal, no la excepción.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


def build_optimizer(model, training_config: dict) -> torch.optim.Optimizer:
    nombre = str(training_config.get("optimizer", "sgd")).lower()
    parametros = [p for p in model.parameters() if p.requires_grad]

    if nombre == "sgd":
        return torch.optim.SGD(
            parametros,
            lr=training_config["lr0"],
            momentum=training_config.get("momentum", 0.9),
            weight_decay=training_config.get("weight_decay", 0.0005),
        )
    if nombre == "adamw":
        return torch.optim.AdamW(
            parametros,
            lr=training_config["lr0"],
            weight_decay=training_config.get("weight_decay", 0.0005),
        )
    raise ValueError(f"Optimizador desconocido: {nombre!r}. Usar 'sgd' o 'adamw'.")


def build_scheduler(optimizer, training_config: dict, epochs: int):
    nombre = str(training_config.get("lr_scheduler", "cosine")).lower()

    if nombre == "none":
        return None
    if nombre == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if nombre == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.1
        )
    raise ValueError(
        f"Scheduler desconocido: {nombre!r}. Usar 'cosine', 'step' o 'none'."
    )


def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    scaler=None,
    max_batches: int | None = None,
    log_every: int = 50,
) -> dict[str, float]:
    """Entrena una época y devuelve el promedio de cada componente de la pérdida."""
    model.train()
    model.to(device)

    acumulado = {clave: 0.0 for clave in LOSS_KEYS}
    acumulado["loss_total"] = 0.0
    batches = 0

    for indice, (images, targets) in enumerate(loader):
        if max_batches is not None and indice >= max_batches:
            break

        images = [image.to(device) for image in images]
        targets = [
            {clave: valor.to(device) for clave, valor in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=scaler is not None and device.type == "cuda"
        ):
            perdidas = model(images, targets)
            total = sum(perdidas.values())

        if not math.isfinite(float(total.detach())):
            # Un batch con pérdida infinita corrompe los pesos; se saltea.
            print(f"[aviso] batch {indice} con pérdida no finita, se omite")
            continue

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        for clave in LOSS_KEYS:
            acumulado[clave] += float(perdidas[clave].detach())
        acumulado["loss_total"] += float(total.detach())
        batches += 1

        if log_every and indice % log_every == 0:
            print(f"  batch {indice}: loss={float(total.detach()):.4f}")

    if batches == 0:
        return {clave: float("nan") for clave in acumulado}
    return {clave: valor / batches for clave, valor in acumulado.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, history: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    """Restaura el estado y devuelve `(ultima_epoca_completada, historial)`."""
    checkpoint = torch.load(
        Path(path), map_location=device or torch.device("cpu"), weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint["epoch"]), list(checkpoint.get("history", []))
