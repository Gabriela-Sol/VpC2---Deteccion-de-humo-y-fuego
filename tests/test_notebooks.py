"""Invariantes estructurales de los notebooks.

Se chequean acá y no a mano porque los dos modos de falla que cubren son caros:
sin `kernelspec`, papermill aborta la ejecución en Kaggle antes de la primera
celda; con una celda rota, el error aparece a mitad de una corrida de horas.
"""

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def _codigo_python(notebook: dict) -> str:
    """Concatena las celdas de código como un módulo compilable.

    Los magics de IPython (`%cd`) y los comandos de shell (`!pip`) no son Python
    válido, así que se reemplazan por `pass` conservando la indentación: varios
    de ellos viven dentro de un `if`, y borrar la línea dejaría el bloque vacío.
    """
    lineas: list[str] = []

    for celda in notebook["cells"]:
        if celda["cell_type"] != "code":
            continue

        for linea in celda["source"]:
            despojada = linea.lstrip()
            if despojada.startswith(("!", "%")):
                linea = " " * (len(linea) - len(despojada)) + "pass\n"
            lineas.append(linea)

        if lineas and not lineas[-1].endswith("\n"):
            lineas.append("\n")
        lineas.append("\n")

    return "".join(lineas)


def test_hay_notebooks_que_verificar():
    assert NOTEBOOKS, f"No se encontraron notebooks en {NOTEBOOKS_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_declara_kernelspec(path: Path):
    metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
    assert metadata.get("kernelspec", {}).get("name") == "python3"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_el_codigo_compila(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    compile(_codigo_python(notebook), str(path), "exec")
