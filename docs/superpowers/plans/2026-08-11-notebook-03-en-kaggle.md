# Notebook 03 en Kaggle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adaptar el notebook `03_entrenamiento_FasterRCNN.ipynb` para que complete en Kaggle Notebooks las 11 épocas que le faltan, reanudando desde el checkpoint de 19 épocas que quedó en Google Drive.

**Architecture:** El notebook 04 (RT-DETR) ya resolvió esta adaptación y completó sus 30 épocas en Kaggle. El 03 calca ese patrón: detección de entorno (`colab`/`kaggle`/`local`), separación `WORK_DIR`/`RUNS_DIR`, dataset desde `/kaggle/input`, y empaquetado en zip porque Kaggle no tiene credenciales de git. La única lógica nueva —recuperar el checkpoint desde los inputs montados en modo solo lectura— va a `src/` con tests, en vez de quedar inline como en el 04, porque tiene casos borde (idempotencia, permisos) que en el notebook solo se descubren quemando una sesión de GPU.

**Tech Stack:** Python 3.12, PyTorch + torchvision, torchmetrics, pytest, Jupyter (nbformat 4.5).

## Global Constraints

- Diseño de referencia: `docs/superpowers/specs/2026-08-11-notebook-03-en-kaggle-design.md`.
- El notebook 03 es el entregable del TP: tiene que seguir corriendo de punta a punta en Colab y en local, sin regresiones.
- No se tocan hiperparámetros, el bucle de entrenamiento, el código existente de `src/`, ni la lógica de re-anclado del scheduler.
- No se toca el notebook 04: ya está adaptado.
- No se modifica el esquema de `metrics_summary.csv`: el notebook 05 lo lee para los tres modelos.
- En Kaggle **no** se instala `requirements.txt`: reemplazaría el torch preinstalado, que viene compilado contra el CUDA de la imagen.
- Rama de trabajo: `feat/modelos-adicionales-deteccion`. El valor de `REPO_BRANCH` dentro de los notebooks no cambia.
- Todos los comentarios y mensajes al usuario, en español, siguiendo el estilo del repositorio: explican el *por qué*, no el *qué*.
- Comandos desde la raíz del repositorio. Los tests se corren con `pytest` (config en `pytest.ini`, `filterwarnings = error`).

---

## File Structure

| Archivo | Responsabilidad |
|---|---|
| `tests/test_notebooks.py` | **Nuevo.** Invariantes estructurales de los 5 notebooks: declaran kernelspec y su código compila. Guarda de regresión que corre en segundos, contra errores que en Kaggle cuestan una sesión entera. |
| `src/data/kaggle_inputs.py` | **Nuevo.** Recuperación de un checkpoint desde `/kaggle/input`, con permisos e idempotencia. Único módulo nuevo. |
| `tests/test_kaggle_inputs.py` | **Nuevo.** Tests del módulo anterior. |
| `notebooks/03_entrenamiento_FasterRCNN.ipynb` | **Modificado.** Celdas 0-5, 7, 10, 12, 15 y una celda nueva. |
| `notebooks/05_comparacion_modelos.ipynb` | **Modificado.** Una celda markdown con la constancia de la corrida partida. |

---

### Task 1: Guarda de regresión para los notebooks

Los notebooks ya fallaron dos veces por cosas que un test de segundos habría atrapado: metadata sin `kernelspec` (papermill aborta en Kaggle antes de la primera celda) y, potencialmente, una celda con sintaxis rota que recién se descubre a mitad de una corrida. Este test se escribe primero porque las tareas 3 a 8 lo usan como verificación.

**Files:**
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: nada.
- Produces: `pytest tests/test_notebooks.py`, la verificación que corren todas las tareas de notebook.

- [ ] **Step 1: Escribir el test**

Crear `tests/test_notebooks.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que pasa**

Run: `pytest tests/test_notebooks.py -v`
Expected: PASS, 11 tests (1 + 5 + 5).

- [ ] **Step 3: Verificar que el test detecta el problema que dice detectar**

Un test que nunca vio rojo no sirve. Romper a propósito y comprobar:

```bash
python3 -c "
import json, pathlib
p = pathlib.Path('notebooks/05_comparacion_modelos.ipynb')
nb = json.loads(p.read_text())
nb['metadata'] = {}
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
"
pytest tests/test_notebooks.py -v
```

Expected: FAIL en `test_declara_kernelspec[05_comparacion_modelos.ipynb]`.

- [ ] **Step 4: Restaurar el notebook y confirmar verde**

```bash
git checkout notebooks/05_comparacion_modelos.ipynb
pytest tests/test_notebooks.py -v
```

Expected: PASS, 11 tests. `git status --short` no debe listar `notebooks/05_comparacion_modelos.ipynb`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebooks.py
git commit -m "test: invariantes estructurales de los notebooks"
```

---

### Task 2: Recuperación del checkpoint desde los inputs de Kaggle

**Files:**
- Create: `src/data/kaggle_inputs.py`
- Test: `tests/test_kaggle_inputs.py`

**Interfaces:**
- Consumes: nada.
- Produces: `restore_checkpoint_from_inputs(inputs_root: Path, dest_path: Path) -> Path | None`. Devuelve la ruta del checkpoint listo para usar en `dest_path`, o `None` si no había nada que recuperar. Lo consume la Task 6.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_kaggle_inputs.py`:

```python
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
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `pytest tests/test_kaggle_inputs.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.kaggle_inputs'`.

- [ ] **Step 3: Escribir la implementación mínima**

Crear `src/data/kaggle_inputs.py`:

```python
"""Recuperación de una corrida previa desde los inputs de Kaggle.

Kaggle borra `/kaggle/working` al cerrar la sesión. Lo que sobrevive vuelve
montado en `/kaggle/input`, en modo solo lectura, y llega por dos caminos que
este módulo trata igual: un Dataset subido a mano (por ejemplo el checkpoint que
venía de Colab) o el output de una versión guardada con Quick Save y remontado
con Add Input > Your Work.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path


def restore_checkpoint_from_inputs(inputs_root, dest_path) -> Path | None:
    """Deja en `dest_path` el checkpoint montado bajo `inputs_root`.

    Devuelve la ruta del checkpoint listo para usar, o `None` si no se encontró
    ninguno. Es idempotente: si `dest_path` ya existe no lo pisa, porque lo que
    está en el directorio de trabajo es siempre más reciente que lo montado.
    """
    inputs_root = Path(inputs_root)
    dest_path = Path(dest_path)

    if dest_path.exists():
        return dest_path

    if not inputs_root.is_dir():
        return None

    # El patrón arranca con `*/` para saltear el nivel de slug y `**` matchea
    # cero o más directorios, así que cubre tanto el archivo suelto en la raíz
    # del Dataset como el anidado en runs/<experimento>/.
    candidatos = sorted(inputs_root.glob(f"*/**/{dest_path.name}"))
    if not candidatos:
        return None

    # El último alfabéticamente es el de mayor versión del notebook, porque
    # Kaggle numera los slugs de output de forma creciente.
    origen = candidatos[-1]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, dest_path)

    # copy2 preserva los permisos del origen, que viene de /kaggle/input y es
    # de solo lectura. El bucle de entrenamiento sobreescribe el checkpoint al
    # cerrar cada época, así que sin esto muere con PermissionError.
    dest_path.chmod(dest_path.stat().st_mode | stat.S_IWUSR)

    return dest_path
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `pytest tests/test_kaggle_inputs.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Correr la suite completa**

Run: `pytest`
Expected: PASS, sin regresiones.

- [ ] **Step 6: Commit**

```bash
git add src/data/kaggle_inputs.py tests/test_kaggle_inputs.py
git commit -m "feat: recuperar el checkpoint desde los inputs de Kaggle"
```

---

### Task 3: Detección de entorno, dependencias y verificación de GPU

**Files:**
- Modify: `notebooks/03_entrenamiento_FasterRCNN.ipynb`, celdas de índice 1 (`c8b4ce34`), 2 (`3b38fe6b`) y 3 (`666423f9`)
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: nada.
- Produces: las variables `IN_COLAB`, `IN_KAGGLE`, `ENV`, `DEVICE`, `DEVICE_NAME` y el import de `shutil`, que usan las tareas 4 a 7.

Usar la herramienta NotebookEdit (`edit_mode: replace`) con el `cell_id` indicado, o un script de Python sobre el JSON. No reordenar celdas.

- [ ] **Step 1: Reemplazar la celda 1 (`c8b4ce34`), setup general**

```python
# ============================================================
# Setup general del entorno
# ============================================================

from pathlib import Path
import os
import sys
import random
import shutil
import time
import yaml

SEED = 42
random.seed(SEED)

IN_COLAB = "google.colab" in sys.modules
# KAGGLE_KERNEL_RUN_TYPE lo define el runtime de Kaggle y no existe si alguien
# instala el paquete `kaggle` en otra máquina, a diferencia de /kaggle o de la
# variable KAGGLE_URL_BASE que trae la librería.
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IN_COLAB:
    ENV = "colab"
elif IN_KAGGLE:
    ENV = "kaggle"
else:
    ENV = "local"

print("Entorno:", ENV)
print("Directorio actual:", Path.cwd())
```

- [ ] **Step 2: Reemplazar la celda 2 (`3b38fe6b`), dependencias**

```python
# ============================================================
# Instalación de dependencias
# ============================================================

REPO_URL = "https://github.com/Gabriela-Sol/VpC2---Deteccion-de-humo-y-fuego"
REPO_NAME = "VpC2---Deteccion-de-humo-y-fuego"
# Rama del repositorio desde la que se clona y a la que se commitean los
# resultados. Mientras el PR este abierto tiene que apuntar a la rama del PR;
# una vez mergeado, cambiar a "main".
REPO_BRANCH = "feat/modelos-adicionales-deteccion"

RAW_REQUIREMENTS = (
    "https://raw.githubusercontent.com/Gabriela-Sol/"
    f"{REPO_NAME}/{REPO_BRANCH}/requirements.txt"
)

if IN_COLAB:
    !pip install -q -r {RAW_REQUIREMENTS}
elif IN_KAGGLE:
    # Solo lo que falta en la imagen: torch, torchvision, PIL, matplotlib,
    # numpy, pandas y yaml ya vienen. requirements.txt lista `torch` sin pinear
    # la build de CUDA, y dejar que pip lo resuelva en Kaggle reemplazaría el
    # torch preinstalado (que sí viene compilado contra el CUDA de la imagen)
    # por uno cualquiera de PyPI.
    !pip install -q torchmetrics pycocotools

print("Dependencias instaladas.")
```

- [ ] **Step 3: Reemplazar la celda 3 (`666423f9`), verificación de GPU**

```python
# ============================================================
# Verificación de GPU
# ============================================================

import torch

print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
    print("GPU:", DEVICE_NAME)

    # La P100 de Kaggle es sm_60 y el PyTorch de la imagen ya no trae kernels
    # por debajo de sm_70. CUDA igual reporta la placa como disponible, así que
    # sin este corte el error aparece recién en el primer forward, después de
    # haber esperado todo el setup y la carga del dataset.
    def _sm_a_tupla(arch: str) -> tuple[int, int]:
        numero = arch.removeprefix("sm_")
        return int(numero[:-1]), int(numero[-1])

    soportadas = [
        _sm_a_tupla(arch)
        for arch in torch.cuda.get_arch_list()
        if arch.startswith("sm_")
    ]
    capacidad = torch.cuda.get_device_capability(0)

    # Se compara contra el mínimo y no con `in soportadas` porque el binario de
    # una sm es compatible hacia arriba dentro de la misma major (sm_86 corre en
    # sm_89): la lista no enumera todas las placas que efectivamente funcionan.
    if soportadas and capacidad < min(soportadas):
        minima = "sm_{}{}".format(*min(soportadas))
        raise RuntimeError(
            f"{DEVICE_NAME} es sm_{capacidad[0]}{capacidad[1]} y este PyTorch "
            f"solo soporta desde {minima}. Elegir otra GPU: en Kaggle, "
            f"Session options > Accelerator > GPU T4 x2."
        )
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "cpu"
    if IN_COLAB:
        # Cortar acá y no avisar nomás: en CPU la corrida no termina nunca y el
        # usuario se enteraría recién dentro de varias horas.
        raise RuntimeError(
            "No se detectó GPU. Faster R-CNN en CPU es inviable para 30 épocas. "
            "Activar Entorno de ejecución > Cambiar tipo de entorno > GPU."
        )
    if IN_KAGGLE:
        raise RuntimeError(
            "No se detectó GPU. Faster R-CNN en CPU es inviable para 30 épocas. "
            "Activar Session options > Accelerator > GPU T4 x2."
        )
    print("Sin GPU y fuera de Colab: se sigue en CPU, solo sirve para pruebas cortas.")

print("Device:", DEVICE)
```

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_notebooks.py -v`
Expected: PASS, 11 tests. Si `test_el_codigo_compila[03_...]` falla, hay un error de sintaxis en lo pegado.

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_entrenamiento_FasterRCNN.ipynb
git commit -m "notebooks: el 03 detecta el entorno y verifica la GPU"
```

---

### Task 4: Rutas de trabajo y clonado del repositorio

**Files:**
- Modify: `notebooks/03_entrenamiento_FasterRCNN.ipynb`, celdas de índice 4 (`6adb7b36`) y 5 (`178f3a1d`)
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `IN_COLAB`, `IN_KAGGLE`, `ENV`, `REPO_URL`, `REPO_NAME`, `REPO_BRANCH` (Task 3).
- Produces: `WORK_DIR`, `RUNS_DIR` y `PROJECT_DIR`. `DRIVE_RUNS_DIR` deja de existir: las tareas 6 y 7 usan `RUNS_DIR`.

- [ ] **Step 1: Reemplazar la celda 4 (`6adb7b36`), almacenamiento de trabajo**

Hoy esta celda hace `mkdir` sobre `/content/drive` sin condicionar, así que fuera de Colab rompe.

```python
# ============================================================
# Almacenamiento de trabajo (Drive en Colab)
# ============================================================
# WORK_DIR es la raíz escribible del entorno: de ahí sale el clon del repo.
# RUNS_DIR es donde viven los checkpoints y los pesos, y tiene que persistir
# entre sesiones: en Colab eso es Drive, en Kaggle es /kaggle/working más un
# Quick Save.

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

    DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/VCII_DFire")
    WORK_DIR = Path("/content")
    RUNS_DIR = DRIVE_PROJECT_DIR / "runs"
    print("Carpeta principal en Drive:", DRIVE_PROJECT_DIR)
elif IN_KAGGLE:
    # Único directorio escribible que Kaggle preserva al hacer Quick Save, y por
    # lo tanto el único desde el que se puede reanudar en otra sesión.
    WORK_DIR = Path("/kaggle/working")
    RUNS_DIR = WORK_DIR / "runs"
else:
    WORK_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    RUNS_DIR = WORK_DIR / "runs"

RUNS_DIR.mkdir(parents=True, exist_ok=True)

print("WORK_DIR:", WORK_DIR)
print("Carpeta de corridas:", RUNS_DIR)
```

- [ ] **Step 2: Reemplazar la celda 5 (`178f3a1d`), clonado del repositorio**

```python
# ============================================================
# Clonado o actualización del repositorio
# ============================================================

if ENV == "local":
    # Ya estamos dentro del repo: no hay nada que clonar.
    PROJECT_DIR = WORK_DIR
else:
    PROJECT_DIR = WORK_DIR / REPO_NAME

    if PROJECT_DIR.exists():
        print("El repositorio ya existe. Actualizando...")
        %cd {PROJECT_DIR}
        !git checkout {REPO_BRANCH}
        !git pull origin {REPO_BRANCH}
    else:
        print("Clonando repositorio...")
        %cd {WORK_DIR}
        !git clone -b {REPO_BRANCH} {REPO_URL}.git
        %cd {PROJECT_DIR}

# Necesario para que `import src...` funcione.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

print("PROJECT_DIR:", PROJECT_DIR)
print("Contenido del proyecto:", os.listdir(PROJECT_DIR))
```

- [ ] **Step 3: Verificar que no queda ninguna referencia colgada a `DRIVE_RUNS_DIR`**

Run:
```bash
grep -c "DRIVE_RUNS_DIR" notebooks/03_entrenamiento_FasterRCNN.ipynb
```
Expected: `2` — las dos que quedan por migrar en las tareas 6 y 7 (celdas 10 y 12). Si da `0`, alguien ya las tocó fuera de plan; si da más, revisar.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_notebooks.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_entrenamiento_FasterRCNN.ipynb
git commit -m "notebooks: el 03 separa WORK_DIR de RUNS_DIR"
```

---

### Task 5: Dataset desde los inputs de Kaggle

**Files:**
- Modify: `notebooks/03_entrenamiento_FasterRCNN.ipynb`, celda de índice 7 (`b0fea70e`)
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `IN_KAGGLE` (Task 3).
- Produces: `DATA_DIR`, que consume la celda 8 (datasets y dataloaders), sin cambios.

- [ ] **Step 1: Reemplazar la celda 7 (`b0fea70e`)**

```python
# ============================================================
# Descarga o localización del dataset
# ============================================================

DATASET_ID = "sayedgamal99/smoke-fire-detection-yolo"


def find_yolo_dataset_dir(root: Path) -> Path:
    """Busca la carpeta con train/images, train/labels, val/images y val/labels."""
    for candidate in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if all(
            (candidate / split / kind).exists()
            for split in ["train", "val"]
            for kind in ["images", "labels"]
        ):
            return candidate
    raise FileNotFoundError("No se encontró una estructura YOLO válida.")


DATA_DIR = None

if IN_KAGGLE:
    # El dataset ya está publicado en Kaggle, así que si se agregó con
    # Add Input > Datasets se monta en /kaggle/input y no hay nada que bajar.
    for mount in sorted(Path("/kaggle/input").glob("*")):
        if not mount.is_dir():
            continue
        try:
            DATA_DIR = find_yolo_dataset_dir(mount)
        except FileNotFoundError:
            continue
        print("Dataset montado desde los inputs del notebook:", mount.name)
        break
    if DATA_DIR is None:
        print(
            f"No se encontró el dataset en /kaggle/input. Agregarlo con "
            f"Add Input > Datasets > {DATASET_ID} para evitar la descarga."
        )

if DATA_DIR is None:
    import kagglehub

    DATA_DIR = find_yolo_dataset_dir(Path(kagglehub.dataset_download(DATASET_ID)))

print("Carpeta de datos:", DATA_DIR)
```

- [ ] **Step 2: Correr los tests**

Run: `pytest tests/test_notebooks.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_entrenamiento_FasterRCNN.ipynb
git commit -m "notebooks: el 03 toma el dataset desde los inputs de Kaggle"
```

---

### Task 6: Recuperación del checkpoint y reanudación

**Files:**
- Modify: `notebooks/03_entrenamiento_FasterRCNN.ipynb` — insertar una celda nueva en el índice 10, y modificar las que quedan en los índices 11 (`662937bd`, era la 10) y 13 (`0dca7d81`, era la 12)
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `restore_checkpoint_from_inputs` (Task 2), `RUNS_DIR` (Task 4), `IN_KAGGLE` (Task 3), `experiment_name` (celda 6, sin cambios).
- Produces: `CHECKPOINT_PATH`, que la celda de reanudación y el bucle de entrenamiento ya usan.

**Cuidado con los índices:** insertar corre en uno todas las celdas siguientes. Hacer la inserción primero y después editar por `cell_id`, que no cambia.

- [ ] **Step 1: Insertar la celda nueva en el índice 10**

Con NotebookEdit (`edit_mode: insert`, `cell_type: code`) justo antes de la celda `662937bd`:

```python
# ============================================================
# Recuperar la corrida de una sesión anterior (Kaggle)
# ============================================================
# Kaggle borra /kaggle/working al cerrar la sesión. El checkpoint sobrevive de
# dos formas y las dos se montan igual en /kaggle/input: como Dataset privado
# (el que trae las 19 épocas entrenadas en Colab) o como output de una versión
# guardada con Quick Save y remontada con Add Input > Your Work.
#
# En Colab no hace falta: la corrida vive en Drive, que persiste entre sesiones.

from src.data.kaggle_inputs import restore_checkpoint_from_inputs

CHECKPOINT_PATH = RUNS_DIR / experiment_name / "last_checkpoint.pth"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

if IN_KAGGLE:
    recuperado = restore_checkpoint_from_inputs(Path("/kaggle/input"), CHECKPOINT_PATH)
    if recuperado is None:
        print("No hay checkpoint en /kaggle/input: el entrenamiento arranca de cero.")
    else:
        print("Checkpoint recuperado en:", recuperado)
```

- [ ] **Step 2: Reemplazar la celda `662937bd` (reanudación), que ahora está en el índice 11**

Solo cambian las tres primeras líneas ejecutables: `CHECKPOINT_PATH` ya viene definido por la celda nueva. El resto —incluido el re-anclado del scheduler— queda idéntico.

```python
# ============================================================
# Reanudar desde el último checkpoint si existe
# ============================================================

from src.engine.trainer import load_checkpoint

start_epoch = 0
history = []

if CHECKPOINT_PATH.exists():
    start_epoch, history = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, DEVICE)
    print(f"Checkpoint encontrado. Reanudando desde la época {start_epoch + 1}.")

    # El checkpoint también restaura el estado del scheduler, incluido el total
    # de épocas con el que se creó (T_max del coseno). Si después de esa corrida
    # se amplió `epochs` en el YAML, hay que reanclarlo al total nuevo: se
    # devuelve el LR a lr0 y se adelanta el scheduler `start_epoch` pasos. Así
    # las épocas que faltan siguen exactamente el coseno de una corrida de
    # `epochs` épocas y terminan en LR ~0. Sin esto, el coseno viejo ya llegó a
    # su mínimo y las épocas extra entrenarían con LR ~0 (no aprenden nada) o,
    # peor, con el LR subiendo de nuevo hacia lr0 sobre el final.
    if scheduler is not None and start_epoch < training_cfg["epochs"]:
        for grupo in optimizer.param_groups:
            grupo["lr"] = training_cfg["lr0"]
            grupo.pop("initial_lr", None)
        scheduler = build_scheduler(optimizer, training_cfg, training_cfg["epochs"])
        for _ in range(start_epoch):
            scheduler.step()
        print(
            f"Scheduler reanclado a {training_cfg['epochs']} épocas. "
            f"LR de la próxima época: {optimizer.param_groups[0]['lr']:.6f}"
        )
else:
    print("No hay checkpoint previo. Entrenamiento desde cero.")
```

- [ ] **Step 3: Reemplazar la celda `0dca7d81` (pesos finales), que ahora está en el índice 13**

```python
# ============================================================
# Guardar los pesos finales
# ============================================================

# Se guarda como last.pth y no best.pth: el bucle no hace seguimiento de la
# mejor época, así que estos son los pesos de la última, que son también los
# que evalúa el reporte de más abajo.
if experiment_config["output"]["save_weights"]:
    WEIGHTS_PATH = RUNS_DIR / experiment_name / "last.pth"
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("Pesos guardados en:", WEIGHTS_PATH)
```

- [ ] **Step 4: Verificar que no quedan referencias a `DRIVE_RUNS_DIR`**

Run:
```bash
grep -c "DRIVE_RUNS_DIR" notebooks/03_entrenamiento_FasterRCNN.ipynb || echo 0
```
Expected: `0`.

- [ ] **Step 5: Verificar el orden de las celdas**

Run:
```bash
python3 -c "
import json
nb = json.load(open('notebooks/03_entrenamiento_FasterRCNN.ipynb'))
for i, c in enumerate(nb['cells']):
    print(i, c.get('id'), ''.join(c['source']).split(chr(10))[1][:60])
"
```
Expected: 17 celdas; la nueva en el índice 10, `662937bd` en el 11, `4f9e8e77` (bucle de entrenamiento) en el 12, `0dca7d81` en el 13.

- [ ] **Step 6: Correr los tests**

Run: `pytest tests/test_notebooks.py -v && pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add notebooks/03_entrenamiento_FasterRCNN.ipynb
git commit -m "notebooks: el 03 reanuda desde el checkpoint montado en Kaggle"
```

---

### Task 7: Publicación de resultados desde Kaggle

**Files:**
- Modify: `notebooks/03_entrenamiento_FasterRCNN.ipynb`, celda `00cbb88d` (última, índice 16 después de la inserción)
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `IN_KAGGLE` (Task 3), `PROJECT_DIR` (Task 4), `RUNS_DIR` (Task 4), `REPORTS_RESULTS_DIR` y `experiment_name` (celdas 6 y 14, sin cambios), `shutil` (Task 3).
- Produces: nada que consuman otras tareas.

- [ ] **Step 1: Reemplazar la celda `00cbb88d`**

```python
# ============================================================
# Publicación de resultados
# ============================================================
# En Colab y local se commitea directo al repo. En Kaggle no hay credenciales de
# git, así que se empaqueta todo y el commit se hace después desde la máquina
# donde sí están configuradas.

import subprocess

if IN_KAGGLE:
    entregable = Path("/kaggle/working") / f"{experiment_name}_resultados"
    if entregable.exists():
        shutil.rmtree(entregable)
    shutil.copytree(REPORTS_RESULTS_DIR, entregable / "reports_results")

    # last.pth va aparte del reporte: lo necesitan el notebook 05 y la demo, y
    # pesa demasiado para commitearlo al repo. El last_checkpoint.pth queda
    # fuera del zip a propósito: son 500 MB que Quick Save ya preserva como
    # output de la versión, y solo sirven para reanudar dentro de Kaggle.
    pesos = RUNS_DIR / experiment_name / "last.pth"
    if pesos.exists():
        shutil.copy(pesos, entregable / "last.pth")

    zip_path = shutil.make_archive(str(entregable), "zip", root_dir=entregable)
    shutil.rmtree(entregable)

    print("Paquete listo:", zip_path)
    print(f"Tamaño: {Path(zip_path).stat().st_size / 1e6:.1f} MB")
    print()
    print("Pasos siguientes:")
    print("  1. Save Version > Quick Save, para conservar la corrida completa.")
    print("  2. Bajar el zip desde el panel Output.")
    print(f"  3. Descomprimir reports_results/ en "
          f"reports/results/{experiment_name}/ del repo y commitear.")
    print(f"  4. Subir last.pth a Drive en VCII_DFire/runs/{experiment_name}/.")
else:
    %cd {PROJECT_DIR}

    !git config user.name "Gabriela-Sol"
    !git config user.email "solgab.salazar@gmail.com"

    pull_result = subprocess.run(
        ["git", "pull", "--rebase", "origin", REPO_BRANCH], text=True, capture_output=True
    )
    print(pull_result.stdout, pull_result.stderr)

    if pull_result.returncode != 0:
        raise RuntimeError("No se pudo completar git pull --rebase. Revisar conflictos.")

    for path in [
        f"reports/results/{experiment_name}/",
        "configs/experiments/fasterrcnn_r50fpn.yaml",
    ]:
        if Path(path).exists():
            subprocess.run(["git", "add", path], check=True)
            print("Agregado:", path)

    status = subprocess.run(["git", "status", "--short"], text=True, capture_output=True)
    print(status.stdout)

    if not status.stdout.strip():
        print("No hay cambios nuevos para commitear.")
    else:
        subprocess.run(
            ["git", "commit", "-m", f"results: update {experiment_name} outputs"], check=True
        )
        print(f"Commit creado. Para publicarlo: !git push origin {REPO_BRANCH}")
```

- [ ] **Step 2: Correr los tests**

Run: `pytest tests/test_notebooks.py -v`
Expected: PASS, 11 tests. Este paso importa más que en las otras tareas: el `%cd` quedó dentro de un `else`, y si la indentación del magic saliera mal, `test_el_codigo_compila` lo caza.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_entrenamiento_FasterRCNN.ipynb
git commit -m "notebooks: el 03 empaqueta los resultados cuando corre en Kaggle"
```

---

### Task 8: Documentación de la corrida partida

**Files:**
- Modify: `notebooks/03_entrenamiento_FasterRCNN.ipynb`, celda markdown `6045ad03` (índice 0)
- Modify: `notebooks/05_comparacion_modelos.ipynb`, celda markdown nueva
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: nada.
- Produces: nada.

- [ ] **Step 1: Reemplazar la celda markdown `6045ad03` del notebook 03**

```markdown
# 03 - Entrenamiento de Faster R-CNN para detección de humo y fuego

Detector de dos etapas (`fasterrcnn_resnet50_fpn_v2` de torchvision) preentrenado
en COCO, con el cabezal ajustado a las clases `smoke` y `fire`.

El código reusable vive en `src/`; este notebook solo arma la configuración,
corre el bucle de épocas y delega el reporte.

## Dónde correrlo

El notebook detecta el entorno y se adapta solo: corre en **Colab**, en **Kaggle
Notebooks** o en una **GPU local**.

Esta corrida se partió entre dos plataformas: las épocas 1 a 19 se entrenaron en
Colab y las 20 a 30 en la T4 de Kaggle, porque se agotaron los créditos de GPU
de Colab. El `train_time_min` que reporta `metrics_summary.csv` es la suma de
ambos tramos, así que mide el costo real de entrenar el modelo pero no es
estrictamente comparable contra los otros dos, que corrieron enteros en una
sola máquina.

### Cómo seguir en Kaggle

1. Notebook nuevo, panel derecho **Session options > Accelerator > GPU T4 x2**
   e **Internet > On**. Ambas opciones aparecen recién cuando la cuenta está
   verificada por teléfono. No elegir la P100: es `sm_60` y el PyTorch que trae
   la imagen actual de Kaggle solo incluye kernels desde `sm_70`.
2. **Add Input > Datasets** y buscar `sayedgamal99/smoke-fire-detection-yolo`.
3. Subir `last_checkpoint.pth` como Dataset privado y agregarlo también con
   **Add Input**. La celda de recuperación lo copia a `/kaggle/working` y el
   entrenamiento sigue desde donde quedó.
4. Correr todo. Antes de las 12 h, **Save Version > Quick Save**: conserva
   `/kaggle/working` como output de la versión. **Save & Run All** no sirve acá,
   porque reejecuta el notebook desde cero en un contenedor limpio.
5. Al terminar, la última celda arma un `.zip` con los artefactos y `last.pth`
   para bajar, commitear en el repo y subir a Drive.

## Salidas esperadas

En `reports/results/fasterrcnn_r50fpn/`: `results.csv`, `results.png`,
`confusion_matrix.png`, `confusion_matrix_normalized.png`, `PR_curve.png`,
`F1_curve.png`, `P_curve.png`, `R_curve.png`, `experiment_config_used.yaml`
y `metrics_summary.csv`.
```

- [ ] **Step 2: Insertar la celda markdown en el notebook 05**

El notebook 05 tiene 8 celdas y la única markdown es la 0. La nota va como celda
markdown nueva en el **índice 3**, justo antes de la celda de código `44e8f87f`
("Tabla comparativa principal"), que es donde el lector ve los tiempos por
primera vez. Usar NotebookEdit con `edit_mode: insert` y `cell_type: markdown`.

```markdown
> **Sobre `train_time_min`.** Los tres modelos entrenaron 30 épocas, pero
> `fasterrcnn_r50fpn` se partió entre dos plataformas: 19 épocas en Colab y 11
> en la T4 de Kaggle, porque se agotaron los créditos de GPU. Su tiempo es la
> suma de ambos tramos y refleja el costo real de entrenarlo, pero para comparar
> velocidad entre arquitecturas conviene mirar `fps`, que se mide en una sola
> pasada de validación sobre la misma máquina que produjo el resto de la fila.
```

- [ ] **Step 3: Correr los tests**

Run: `pytest tests/test_notebooks.py -v && pytest`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add notebooks/03_entrenamiento_FasterRCNN.ipynb notebooks/05_comparacion_modelos.ipynb
git commit -m "docs: constancia de la corrida partida de Faster R-CNN"
```

---

## Verificación final

Después de la Task 8, antes de tocar Kaggle:

- [ ] `pytest` completo en verde.
- [ ] `git status --short` sin cambios sin commitear en `notebooks/` ni en `src/`.
- [ ] Simulacro local del camino `local` del notebook: abrirlo y correr las celdas 1 a 7. Sin GPU, la celda 3 imprime el aviso y sigue; la 4 crea `runs/` en la raíz del repo; la 5 no clona nada; la 7 baja el dataset con `kagglehub` o lo toma de la caché. Las celdas de entrenamiento en adelante **no** se corren en local.
- [ ] `git diff main --stat -- notebooks/04_entrenamiento_RTDETR.ipynb` vacío: el 04 no se tocó.

## Pasos manuales en Kaggle (fuera del código)

No son parte del plan de implementación, pero sin ellos el notebook no arranca:

1. Descargar `last_checkpoint.pth` de `MyDrive/VCII_DFire/runs/fasterrcnn_r50fpn/`.
2. Subirlo a Kaggle como Dataset privado (unos 500 MB).
3. Crear el notebook desde el repo, agregar los dos inputs y elegir GPU T4 x2 con Internet On.
