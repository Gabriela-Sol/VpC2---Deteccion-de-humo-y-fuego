"""Tests de la recuperación de checkpoints desde /kaggle/input."""

import os
import stat
from pathlib import Path

from src.data.kaggle_inputs import restore_checkpoint_from_inputs


def _montaje_readonly(root: Path, slug: str, relativo: str, contenido: bytes) -> Path:
    """Simula un input de Kaggle: archivo dentro de un slug, en modo lectura."""
    destino = root / slug / relativo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(contenido)
    destino.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return destino


def test_devuelve_none_cuando_no_hay_inputs(tmp_path: Path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    assert restore_checkpoint_from_inputs(inputs, destino) is None
    assert not destino.exists()


def test_devuelve_none_cuando_el_directorio_no_existe(tmp_path: Path):
    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    assert restore_checkpoint_from_inputs(tmp_path / "no-existe", destino) is None


def test_copia_el_checkpoint_y_lo_deja_escribible(tmp_path: Path):
    inputs = tmp_path / "input"
    _montaje_readonly(inputs, "checkpoint-fasterrcnn", "last_checkpoint.pth", b"pesos")
    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    resultado = restore_checkpoint_from_inputs(inputs, destino)

    assert resultado == destino
    assert destino.read_bytes() == b"pesos"
    # El bucle sobreescribe el checkpoint en cada época: sin permiso de
    # escritura la corrida muere con PermissionError al cerrar la primera.
    assert os.access(destino, os.W_OK)


def test_encuentra_el_checkpoint_anidado(tmp_path: Path):
    inputs = tmp_path / "input"
    _montaje_readonly(
        inputs, "mi-version-5", "runs/fasterrcnn_r50fpn/last_checkpoint.pth", b"pesos"
    )
    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    assert restore_checkpoint_from_inputs(inputs, destino) == destino
    assert destino.read_bytes() == b"pesos"


def test_no_pisa_un_checkpoint_ya_presente(tmp_path: Path):
    inputs = tmp_path / "input"
    _montaje_readonly(inputs, "checkpoint-viejo", "last_checkpoint.pth", b"viejo")
    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"
    destino.parent.mkdir(parents=True)
    destino.write_bytes(b"nuevo")

    assert restore_checkpoint_from_inputs(inputs, destino) == destino
    # Lo de /kaggle/working es siempre más reciente que lo montado.
    assert destino.read_bytes() == b"nuevo"


def test_elige_el_ultimo_slug_cuando_hay_varios(tmp_path: Path):
    inputs = tmp_path / "input"
    _montaje_readonly(inputs, "mi-version-1", "last_checkpoint.pth", b"vieja")
    _montaje_readonly(inputs, "mi-version-2", "last_checkpoint.pth", b"nueva")
    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    restore_checkpoint_from_inputs(inputs, destino)

    assert destino.read_bytes() == b"nueva"


def test_elige_por_mtime_no_alfabetico_con_digitos_diferentes(tmp_path: Path):
    """Verifica que se elige por fecha, no por orden alfabético de slugs.

    Crítico: sorted() sobre strings pone 'mi-version-10' antes que 'mi-version-9',
    así que un algoritmo alfabético devolvería el checkpoint viejo (v9) cuando el
    más reciente es v10. Este test fija explícitamente los mtime para asegurar que
    la función elige correctamente por fecha de modificación.
    """
    inputs = tmp_path / "input"

    # Crear v9 con contenido viejo y v10 con contenido nuevo
    checkpoint_v9 = _montaje_readonly(inputs, "mi-version-9", "last_checkpoint.pth", b"viejo")
    checkpoint_v10 = _montaje_readonly(inputs, "mi-version-10", "last_checkpoint.pth", b"nuevo")

    # Fijar mtimes explícitamente: v9 en un tiempo anterior, v10 en uno posterior
    mtime_v9 = 1000000.0
    mtime_v10 = 1000001.0
    os.utime(checkpoint_v9, (mtime_v9, mtime_v9))
    os.utime(checkpoint_v10, (mtime_v10, mtime_v10))

    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    resultado = restore_checkpoint_from_inputs(inputs, destino)

    # Sin la corrección (sorted alfabético), esto fallaría porque elegiría v9 (b"viejo")
    assert resultado == destino
    assert destino.read_bytes() == b"nuevo"


def test_prefiere_el_output_de_notebook_sobre_un_dataset(tmp_path: Path):
    """El output de notebook gana aunque el Dataset tenga un mtime más nuevo.

    Los outputs de Add Input > Your Work se montan bajo notebooks/<usuario>/<slug>/
    y son la corrida más avanzada. Un Dataset subido a mano puede quedar con un
    mtime posterior (Kaggle estampa la fecha de procesamiento de la subida), así
    que elegir por mtime a secas devuelve el checkpoint viejo del Dataset.
    """
    inputs = tmp_path / "input"
    de_dataset = _montaje_readonly(
        inputs, "checkpoint-migrado", "last_checkpoint.pth", b"viejo-del-dataset"
    )
    de_notebook = _montaje_readonly(
        inputs,
        "notebooks",
        "marcoslund/notebook-fasterrcnn/runs/fasterrcnn_r50fpn/last_checkpoint.pth",
        b"nuevo-del-notebook",
    )
    os.utime(de_notebook, (1000000.0, 1000000.0))
    os.utime(de_dataset, (1000001.0, 1000001.0))

    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    assert restore_checkpoint_from_inputs(inputs, destino) == destino
    assert destino.read_bytes() == b"nuevo-del-notebook"


def test_desempata_por_mtime_entre_outputs_de_notebook(tmp_path: Path):
    inputs = tmp_path / "input"
    viejo = _montaje_readonly(
        inputs, "notebooks", "usuario/version-9/last_checkpoint.pth", b"viejo"
    )
    nuevo = _montaje_readonly(
        inputs, "notebooks", "usuario/version-10/last_checkpoint.pth", b"nuevo"
    )
    os.utime(viejo, (1000000.0, 1000000.0))
    os.utime(nuevo, (1000001.0, 1000001.0))

    destino = tmp_path / "runs" / "fasterrcnn_r50fpn" / "last_checkpoint.pth"

    assert restore_checkpoint_from_inputs(inputs, destino) == destino
    assert destino.read_bytes() == b"nuevo"
