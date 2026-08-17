"""Invariantes estructurales de los notebooks.

Se chequean acá y no a mano porque los dos modos de falla que cubren son caros:
sin `kernelspec`, papermill aborta la ejecución en Kaggle antes de la primera
celda; con una celda rota, el error aparece a mitad de una corrida de horas.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def _codigo_python(notebook: dict) -> str:
    """Concatena las celdas de código como un módulo compilable.

    Los magics de IPython (`%cd`) y los comandos de shell (`!pip`) no son Python
    válido, así que se reemplazan por `pass` conservando la indentación: varios
    de ellos viven dentro de un `if`, y borrar la línea dejaría el bloque vacío.

    También maneja asignaciones con magics (ej. `x = !ls`), común en IPython para
    capturar salida de comandos: se reemplazan por `x = None` para preservar la
    variable sin sintaxis inválida.
    """
    lineas: list[str] = []

    for celda in notebook["cells"]:
        if celda["cell_type"] != "code":
            continue

        # Normalizar source: puede ser una lista de líneas o un string único.
        # nbformat acepta ambas formas, pero algunas herramientas producen strings.
        source = celda["source"]
        if isinstance(source, str):
            source = source.splitlines(keepends=True)

        continuacion_shell = False

        for linea in source:
            despojada = linea.lstrip()
            indentacion = " " * (len(linea) - len(despojada))

            # Un `!comando \` sigue en las líneas siguientes mientras cada una
            # termine en `\`: todas pertenecen al shell y se descartan, porque
            # la primera ya quedó reemplazada por `pass`.
            if continuacion_shell:
                continuacion_shell = despojada.rstrip().endswith("\\")
                continue

            # Magic al inicio de línea: reemplazar por pass
            if despojada.startswith(("!", "%")):
                continuacion_shell = despojada.rstrip().endswith("\\")
                linea = indentacion + "pass\n"
            # Asignación con magic: nombre = !comando o nombre = %magic
            # Se reemplaza por nombre = None para mantener compilabilidad
            elif re.match(r"^(\w+)\s*=\s*[!%]", despojada):
                continuacion_shell = despojada.rstrip().endswith("\\")
                match = re.match(r"^(\w+)\s*=\s*", despojada)
                nombre = match.group(1)
                linea = indentacion + f"{nombre} = None\n"

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


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_source_de_celdas_es_lista(path: Path):
    """El `source` de cada celda tiene que ser una lista de líneas, no un string.

    `_codigo_python` tolera ambas formas (ver más abajo), pero esa tolerancia
    está para no romper si algo se cuela, no para dar por buena la forma no
    canónica: un `source` como string único es la huella de una herramienta
    que reescribió la celda entera (pasó durante este mismo trabajo), y diluye
    los diffs de git porque una edición de una línea se ve como un cambio del
    párrafo completo.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for indice, celda in enumerate(notebook["cells"]):
        assert isinstance(celda["source"], list), (
            f"{path.name}: la celda {indice} tiene `source` como string, "
            "no como lista de líneas."
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_termina_en_salto_de_linea(path: Path):
    """El archivo tiene que terminar en `\\n`.

    Sin newline final, `git diff` marca "\\ No newline at end of file" en
    cualquier edición futura de la última línea, lo que ensucia el diff de
    algo que no cambió. Pasó durante este mismo trabajo con un notebook
    reescrito a mano.
    """
    contenido = path.read_bytes()
    assert contenido.endswith(b"\n"), f"{path.name} no termina en salto de línea."


def test_maneja_asignacion_con_magic():
    """Verifica que se detecte y reemplace correctamente `nombre = !comando`.

    Patrón común en IPython para capturar salida de comandos shell.
    Sin este manejo, el código generado sería sintácticamente inválido.
    """
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["gpu_info = !nvidia-smi -L\n", "print(gpu_info)\n"],
            }
        ]
    }
    codigo = _codigo_python(nb)
    # Debe compilar sin errores
    compile(codigo, "test", "exec")
    # Y contener la asignación reemplazada
    assert "gpu_info = None" in codigo
    assert "!nvidia-smi" not in codigo


def test_maneja_shell_multilinea_con_continuacion():
    """`!comando \\` sigue en las líneas siguientes: son shell, no Python.

    Sin este manejo, la primera línea se reemplaza por `pass` pero las
    continuaciones quedan como texto suelto que no compila (pasó con el
    `!git clone \\` multilínea del notebook 07).
    """
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "!git clone \\\n",
                    "    --branch main \\\n",
                    "    https://ejemplo/repo.git\n",
                    "print('ok')\n",
                ],
            }
        ]
    }
    codigo = _codigo_python(nb)
    compile(codigo, "test", "exec")
    assert "--branch" not in codigo
    assert "print('ok')" in codigo


def test_source_como_string_equivale_a_source_como_lista():
    """nbformat admite `source` como string o como lista, y `_codigo_python`
    tiene que dar lo mismo con las dos formas.

    Sin normalización, iterar un string carácter por carácter produce salida
    diferente: `"%cd\nprint('ok')\n"` carácter por carácter dispara reemplazo
    de `%` → `pass` en el primer carácter pero el resto se corrompe de forma
    que igual compila por accidente (`cd / tmp`). La función tiene que
    producir exactamente la misma salida con ambos formatos.
    """
    como_string = {
        "cell_type": "code",
        "source": "%cd /tmp\nprint('ok')\n",
    }
    como_lista = {
        "cell_type": "code",
        "source": ["%cd /tmp\n", "print('ok')\n"],
    }

    # Ambas formas deben producir exactamente la misma salida
    assert _codigo_python({"cells": [como_string]}) == _codigo_python(
        {"cells": [como_lista]}
    )
