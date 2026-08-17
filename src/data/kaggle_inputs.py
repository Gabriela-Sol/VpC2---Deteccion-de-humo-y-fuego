"""Recuperación de una corrida previa desde los inputs de Kaggle.

Kaggle borra `/kaggle/working` al cerrar la sesión. Lo que sobrevive vuelve
montado en `/kaggle/input`, en modo solo lectura, y llega por dos caminos que
este módulo trata igual: un Dataset subido a mano, por ejemplo un checkpoint
traído de otra plataforma, o el output de una versión guardada con Quick Save y
remontado con Add Input > Your Work.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path


def restore_checkpoint_from_inputs(inputs_root: Path, dest_path: Path) -> Path | None:
    """Deja en `dest_path` el checkpoint montado bajo `inputs_root`.

    Devuelve la ruta del checkpoint listo para usar, o `None` si no se encontró
    ninguno. Es idempotente: si `dest_path` ya existe no lo pisa, porque lo que
    está en el directorio de trabajo es siempre más reciente que lo montado.
    """
    inputs_root = Path(inputs_root)
    dest_path = Path(dest_path)

    if dest_path.exists():
        # Sin este aviso, un checkpoint viejo copiado por una corrida anterior
        # de la celda en la misma sesión sobrevive en silencio aunque los
        # inputs montados hayan cambiado.
        print(f"Se conserva el checkpoint ya presente en {dest_path} (no se copió nada).")
        return dest_path

    if not inputs_root.is_dir():
        return None

    # El patrón arranca con `*/` para saltear el nivel de slug y `**` matchea
    # cero o más directorios, así que cubre tanto el archivo suelto en la raíz
    # del Dataset como el anidado en runs/<experimento>/.
    candidatos = list(inputs_root.glob(f"*/**/{dest_path.name}"))
    if not candidatos:
        return None

    # Los outputs de notebook (Add Input > Your Work) se montan bajo
    # notebooks/<usuario>/<slug>/ y traen la corrida más avanzada. Un Dataset
    # subido a mano puede tener un mtime posterior aunque su checkpoint sea más
    # viejo, porque Kaggle estampa la fecha de procesamiento de la subida; por
    # eso los outputs de notebook tienen prioridad y el mtime solo desempata
    # dentro del grupo que quede.
    de_notebooks = [
        c for c in candidatos if c.relative_to(inputs_root).parts[0] == "notebooks"
    ]
    if de_notebooks:
        candidatos = de_notebooks

    # Elegimos por fecha de modificación del archivo para asegurar la versión más
    # reciente. El orden alfabético es incorrecto para números con diferente cantidad
    # de dígitos (e.g. mi-version-9 vs mi-version-10). Si dos empatan en mtime,
    # sorted() proporciona un desempate determinista por ruta.
    origen = max(candidatos, key=lambda p: (p.stat().st_mtime, str(p)))

    print(f"Checkpoint elegido entre {len(candidatos)} candidato(s): {origen}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, dest_path)

    # copy2 preserva los permisos del origen, que viene de /kaggle/input y es
    # de solo lectura. El bucle de entrenamiento sobreescribe el checkpoint al
    # cerrar cada época, así que sin esto muere con PermissionError.
    dest_path.chmod(dest_path.stat().st_mode | stat.S_IWUSR)

    return dest_path
