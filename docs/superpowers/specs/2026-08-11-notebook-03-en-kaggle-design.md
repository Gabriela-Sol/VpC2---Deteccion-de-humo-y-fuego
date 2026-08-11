# Diseño: ejecutar el notebook 03 (Faster R-CNN) en Kaggle

Fecha: 2026-08-11
Estado: aprobado

## Objetivo

El entrenamiento de `fasterrcnn_r50fpn` quedó a mitad de camino: 19 de las 30 épocas
corrieron en Google Colab y los créditos de GPU se agotaron. Este diseño adapta el
notebook 03 para que las 11 épocas restantes se completen en Kaggle Notebooks,
reanudando desde el checkpoint que quedó en Drive.

El notebook 03 es parte del entregable del trabajo práctico, así que la solución tiene
que mantenerlo como un notebook ejecutable de punta a punta, no reemplazarlo por un
script externo.

## Contexto actual

El notebook 03 asume Colab en todo su recorrido: instala `requirements.txt` completo,
monta Google Drive sin condicionarlo, y guarda checkpoints y pesos en
`/content/drive/MyDrive/VCII_DFire/runs/`. Fuera de Colab, la celda 4 falla al intentar
`mkdir` sobre `/content/drive`.

El notebook 04 (RT-DETR) ya resolvió este mismo problema y completó sus 30 épocas en
Kaggle: 562.91 min en una Tesla T4, mAP50 = 0.7573. Su patrón de adaptación —detección
de entorno, `WORK_DIR`/`RUNS_DIR`, dataset desde `/kaggle/input`, recuperación de la
corrida previa, empaquetado en zip— está probado y es lo que el 03 hereda.

El checkpoint de las 19 épocas (`last_checkpoint.pth`, del orden de 500 MB) está en
Drive y es accesible. `src/engine/trainer.save_checkpoint` guarda modelo, optimizador,
scheduler, número de época e historial completo en un único archivo, así que la corrida
es reanudable en otra máquina sin pérdida de estado ni de métricas por época.

## Alternativas descartadas

**Modal.** GPU más rápida (L4/A10G), sin tope de sesión, y los USD 30 mensuales de
crédito gratuito alcanzarían para las 11 épocas. Se descarta porque no ejecuta
notebooks: requiere un `train.py` con `@app.function`, autenticación por CLI y subir el
dataset completo a un Volume, mientras que en Kaggle D-Fire ya está publicado. El
entregable dejaría de ser reproducible por un tercero.

**Cerrar en 19 épocas.** Cero trabajo, pero los YAML fijan 30 épocas para los tres
modelos precisamente para que la comparación no premie al que entrenó más tiempo. Con
19 contra 30 y 30, la tabla del notebook 05 pierde validez en la dimensión que ese
presupuesto controla.

## Cambios

### 1. Detección de entorno y dependencias (celdas 1-3)

`IN_KAGGLE` se resuelve por la presencia de `KAGGLE_KERNEL_RUN_TYPE`, que solo define el
runtime de Kaggle, y `ENV` toma `colab`, `kaggle` o `local`. Es el mismo criterio del
notebook 04.

La instalación se bifurca:

- Colab: `requirements.txt` completo, como hoy.
- Kaggle: solo `torchmetrics` y `pycocotools`. El resto de lo que importa `src/` —torch,
  torchvision, PIL, matplotlib, numpy, pandas, yaml— ya está en la imagen. Instalar
  `requirements.txt` entero reemplazaría el torch preinstalado, que sí viene compilado
  contra el CUDA de la imagen, por uno cualquiera de PyPI.

La verificación de GPU incorpora el chequeo de compute capability que ya tiene el 04:
compara la capacidad de la placa contra el mínimo de `torch.cuda.get_arch_list()` y corta
con un error explicativo si es menor. Sin eso, una P100 (`sm_60`) pasa la verificación y
muere recién en el primer forward.

### 2. Rutas de trabajo (celdas 4-5)

Se introducen `WORK_DIR` y `RUNS_DIR` con la misma semántica del notebook 04:

| Variable | Colab | Kaggle | Local |
|---|---|---|---|
| `WORK_DIR` | `/content` | `/kaggle/working` | raíz del repo |
| `RUNS_DIR` | `MyDrive/VCII_DFire/runs` | `/kaggle/working/runs` | `<repo>/runs` |

El montaje de Drive queda condicionado a Colab. Las celdas 10 y 12, que hoy referencian
`DRIVE_RUNS_DIR`, pasan a `RUNS_DIR`. El clonado del repositorio se condiciona igual que
en el 04: en local no clona nada porque ya se está dentro del repo.

### 3. Dataset (celda 7)

En Kaggle se recorre `/kaggle/input` buscando la estructura YOLO con la misma
`find_yolo_dataset_dir` que ya existe; si el dataset se agregó con *Add Input →
Datasets*, no se descarga nada. Si no aparece, cae a `kagglehub` como hoy.

### 4. Recuperación del checkpoint (celda nueva, antes de la 10)

Como paso manual previo, y por única vez, el checkpoint de las 19 épocas se descarga de
Drive y se sube a Kaggle como Dataset privado. La celda busca
`last_checkpoint.pth` bajo `/kaggle/input`, lo copia a `RUNS_DIR/<experiment_name>/` y le
restaura permisos de escritura: `shutil.copytree`/`copy2` preservan los del origen, que
es read-only, y el bucle necesita sobreescribir el archivo en cada época.

La celda es idempotente: si ya hay un `last_checkpoint.pth` en `RUNS_DIR`, no lo pisa.
Eso hace que el mismo código sirva para dos casos sin ramificar —el Dataset privado con
el checkpoint de Colab, y *Add Input → Your Work* si hiciera falta una segunda sesión de
Kaggle—.

A partir de ahí la celda 10 funciona sin cambios: encuentra el checkpoint, restaura
estado e historial, reancla el scheduler a las 30 épocas del YAML y reanuda en la 20.

### 5. Publicación de resultados (celda 15)

En Kaggle no hay credenciales de git, así que se bifurca como el notebook 04: se arma un
zip con `reports_results/` y `last.pth` para descargar y commitear desde una máquina con
git configurado. El `last_checkpoint.pth` queda fuera del zip: Quick Save ya lo preserva
como output de la versión y bajar 500 MB no aporta nada. En Colab y local, el commit
directo sigue como está.

### 6. Constancia de la corrida partida

`train_time_min` va a sumar tiempos medidos en dos máquinas distintas. Se deja
constancia en una celda markdown del notebook 03 y en una nota del notebook 05, junto a
la tabla comparativa: 19 épocas en Colab y 11 en la T4 de Kaggle.

No se modifica el esquema de `metrics_summary.csv`. El notebook 05 lo lee para los tres
modelos y agregarle una columna solo a esta fila rompería la comparación.

## Fuera de alcance

Hiperparámetros, el bucle de entrenamiento, el contenido de `src/` y la lógica de
re-anclado del scheduler. El notebook 04 tampoco se toca: ya está adaptado.

## Criterios de éxito

1. El notebook 03 corre de punta a punta en Kaggle y reanuda en la época 20.
2. Las 30 épocas terminan y `reports/results/fasterrcnn_r50fpn/` queda con los mismos
   artefactos que `rtdetr_l` y `yolov8n_baseline`.
3. `metrics_summary.csv` de Faster R-CNN tiene el mismo esquema que los otros dos, y el
   notebook 05 arma la tabla de tres filas sin cambios.
4. El notebook sigue corriendo en Colab y en local sin regresiones.

## Riesgos

- **Tiempo por época desconocido.** RT-DETR-L tardó 18.7 min/época en T4. Faster R-CNN a
  batch 4 con validación completa por época debería rondar 20-35 min, así que 11 épocas
  son unas 4-6.5 h y entran en una sola sesión de 12 h. Si se pasa, el mecanismo de
  recuperación de la celda nueva ya cubre partirla en dos.
- **La P100 de Kaggle no sirve.** Es `sm_60` y el PyTorch de la imagen actual solo trae
  kernels desde `sm_70`. Hay que elegir T4 x2, de la que se usa una sola placa.
