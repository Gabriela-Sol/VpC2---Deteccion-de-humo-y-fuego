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
