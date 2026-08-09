# Diseño: modelos adicionales de detección de humo y fuego

Fecha: 2026-08-09
Estado: aprobado

## Objetivo

El repositorio ya tiene un baseline entrenado con YOLOv8n sobre el dataset D-Fire, con
sus métricas y figuras versionadas en `reports/results/yolov8n_baseline/`. Este diseño
agrega dos modelos más y una capa de comparación, de modo que el informe final contraste
tres familias arquitectónicas distintas sobre el mismo dato y el mismo protocolo de
evaluación.

## Contexto actual

El pipeline vigente es:

1. `configs/experiments/<nombre>.yaml` define el experimento.
2. Un notebook se ejecuta en Google Colab: clona el repo, descarga el dataset con
   `kagglehub` (`sayedgamal99/smoke-fire-detection-yolo`), entrena y guarda los pesos y
   las corridas en Google Drive (`/content/drive/MyDrive/VCII_DFire/runs`).
3. El notebook copia las salidas principales a `reports/results/<nombre>/` dentro del
   repositorio y hace commit y push.

El dataset está en formato YOLO con dos clases, `0: smoke` y `1: fire`, y los splits
tienen 14 122 imágenes de train, 3 099 de validación y 4 306 de test. Dos hechos del EDA
condicionan el diseño: 6 458 imágenes de train no tienen ninguna caja (son negativos
deliberados) y 8 entradas del split test tienen coordenadas fuera del rango `[0, 1]`.

El baseline YOLOv8n corrió 30 épocas a 640 px y terminó con mAP50 = 0.752 y
mAP50-95 = 0.431.

## Modelos

Se agregan dos experimentos, elegidos para que cada uno represente una familia distinta:

| Experimento | Familia | Librería | Estado |
|---|---|---|---|
| `yolov8n_baseline` | Una etapa, CNN | Ultralytics | ya entrenado |
| `fasterrcnn_r50fpn` | Dos etapas, CNN | torchvision | nuevo |
| `rtdetr_l` | Transformer (DETR real-time) | Ultralytics | nuevo |

**Faster R-CNN** usa `fasterrcnn_resnet50_fpn_v2` con pesos preentrenados en COCO. Se
reemplaza el `box_predictor` por uno de 3 clases, porque torchvision reserva el índice 0
para el fondo: las etiquetas YOLO 0 y 1 se remapean a 1 (smoke) y 2 (fire).

**RT-DETR-l** se entrena con la misma función `train_from_config` que ya existe en el
notebook 02, sustituyendo la clase `YOLO` por `RTDETR`. No requiere código nuevo de
entrenamiento ni de reporte, porque Ultralytics genera los mismos artefactos.

## Arquitectura de código

Hasta ahora todo el código vivía dentro de los notebooks. El entrenamiento de Faster
R-CNN necesita del orden de 600 líneas (dataset, loop, evaluación, gráficos), demasiado
para mantener dentro de una celda. Ese código se ubica en `src/`, que ya existe en el
repositorio pero está vacío, y los notebooks lo importan.

```
src/
  __init__.py
  data/
    __init__.py
    yolo_dataset.py      YoloDetectionDataset, collate_fn
  models/
    __init__.py
    detectors.py         build_fasterrcnn(...)
  engine/
    __init__.py
    trainer.py           train_one_epoch, evaluate, save/load checkpoint
    metrics.py           mAP, barrido de confianza, medición de FPS
  reporting/
    __init__.py
    plots.py             results.png, matrices de confusión, curvas PR/F1/P/R
    summary.py           escritura de metrics_summary.csv
```

Los notebooks corren en Colab, donde el repositorio se clona en
`/content/VpC2---Deteccion-de-humo-y-fuego`. Para importar `src` basta con
`sys.path.insert(0, str(PROJECT_DIR))` después del clonado.

### `data/yolo_dataset.py`

`YoloDetectionDataset` es un `torch.utils.data.Dataset` que recibe el directorio de un
split y devuelve el par `(imagen, target)` que espera la API de detección de torchvision.
Sus responsabilidades:

- Convertir cada caja de YOLO normalizado `cx cy w h` a `xyxy` en píxeles absolutos.
- Clampear las coordenadas a los límites de la imagen, para tolerar las 8 entradas
  inválidas del split test sin abortar.
- Descartar cajas degeneradas, con ancho o alto menor a 1 píxel después del clampeo.
- Remapear las clases de `{0, 1}` a `{1, 2}`.
- Para imágenes sin cajas, devolver `boxes` con forma `(0, 4)` y dtype `float32`, y
  `labels` con forma `(0,)` y dtype `int64`. Sin esto torchvision falla, y afecta al 46 %
  de las imágenes de train.

La augmentación de entrenamiento se limita a volteo horizontal aleatorio, que refleja
también las coordenadas de las cajas. No se hace resize manual: el
`GeneralizedRCNNTransform` interno del modelo redimensiona según `min_size` y `max_size`,
que se configuran en 640 y 1024 para quedar en el mismo orden de magnitud que los 640 px
del baseline YOLO.

`collate_fn` simplemente empaqueta las muestras en listas, porque las imágenes de un
batch pueden tener tamaños distintos.

### `models/detectors.py`

`build_fasterrcnn(num_classes, backbone, trainable_backbone_layers, min_size, max_size)`
construye el modelo y reemplaza el cabezal de clasificación. El parámetro `backbone`
acepta `resnet50_fpn_v2` (opción por defecto) y `mobilenet_v3_large_fpn`, que sirve como
alternativa liviana si el presupuesto de cómputo se vuelve un problema.

### `engine/trainer.py`

- `train_one_epoch` ejecuta una época con precisión mixta opcional y devuelve el promedio
  de cada componente de la pérdida.
- `evaluate` corre el modelo sobre un `DataLoader` en modo inferencia y acumula las
  predicciones en una métrica de torchmetrics.
- `save_checkpoint` y `load_checkpoint` persisten modelo, optimizador, scheduler, número
  de época e historial de métricas.

### `engine/metrics.py`

Las métricas se calculan con `torchmetrics.detection.MeanAveragePrecision`, que
implementa el protocolo COCO, el mismo que usa Ultralytics internamente. De ahí salen
mAP50, mAP50-95 y los desagregados por clase.

Las curvas de precisión, recall y F1 en función de la confianza requieren un barrido
aparte: se cachean las predicciones del split de validación una sola vez y se recalculan
los aciertos para una grilla de umbrales de confianza entre 0 y 1, matcheando por IoU.

`measure_inference_fps` mide el tiempo medio de inferencia sobre un subconjunto fijo de
imágenes de validación, con calentamiento previo y sincronización de CUDA.

### `reporting/plots.py`

Genera los mismos archivos que produce Ultralytics, para que las carpetas de resultados
sean comparables entre sí:

- `results.png`: panel con las pérdidas de entrenamiento y las métricas de validación por
  época.
- `confusion_matrix.png` y `confusion_matrix_normalized.png`: matriz de 3×3 (smoke, fire y
  fondo), construida con el mismo criterio que Ultralytics, matcheando por IoU 0.45 a
  confianza 0.25, donde la fila y la columna de fondo capturan falsos positivos y objetos
  no detectados.
- `PR_curve.png`, `F1_curve.png`, `P_curve.png` y `R_curve.png`, a partir del barrido de
  confianza.

### `reporting/summary.py`

Escribe `metrics_summary.csv`, el archivo que hace posible la comparación entre modelos.

## Configuración

Dos archivos nuevos en `configs/experiments/`, que mantienen el esquema actual de tres
bloques (`experiment`, `training`, `output`):

- `fasterrcnn_r50fpn.yaml`
- `rtdetr_l.yaml`

El bloque `training` se extiende con los campos que necesita torchvision y que YOLO
resolvía solo: `backbone`, `trainable_backbone_layers`, `weight_decay`, `momentum`,
`lr_scheduler`, `amp` y `workers`. Los campos que ya existen conservan su nombre y
significado.

## Salidas

Cada experimento deja en `reports/results/<nombre>/` el mismo conjunto de artefactos:

- `results.csv` con una fila por época
- `results.png`
- `confusion_matrix.png` y `confusion_matrix_normalized.png`
- `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`
- `experiment_config_used.yaml`
- `metrics_summary.csv` (nuevo)

El `results.csv` de Ultralytics tiene columnas propias de esa librería, así que no sirve
para comparar entre modelos. Por eso se agrega `metrics_summary.csv`, de una sola fila y
con un esquema idéntico para los tres experimentos:

```
experiment, family, model, params_M, epochs, imgsz, batch, train_time_min,
mAP50, mAP50_95, precision, recall, f1,
mAP50_smoke, mAP50_fire, mAP50_95_smoke, mAP50_95_fire,
fps, device, split
```

Todas las métricas se calculan sobre el split de validación, con protocolo COCO, para que
las tres filas sean comparables. La columna `split` queda registrada explícitamente para
que quede claro sobre qué datos se midió.

## Notebooks

- **`03_entrenamiento_FasterRCNN.ipynb`**: reproduce el esqueleto del notebook 02 (setup,
  verificación de GPU, montaje de Drive, clonado del repo, descarga con kagglehub, carga
  de la configuración) y luego entrena usando los módulos de `src/`, exporta a `reports/`
  y hace commit.
- **`04_entrenamiento_RTDETR.ipynb`**: clon del notebook 02 con `RTDETR` en lugar de
  `YOLO`, más la celda de exportación de `metrics_summary.csv`.
- **`05_comparacion_modelos.ipynb`**: lee los `metrics_summary.csv` de
  `reports/results/*/`, arma la tabla comparativa y genera las figuras en
  `reports/figures/comparacion/`: barras de mAP50 y mAP50-95 por modelo, mAP por clase, y
  un gráfico de dispersión de mAP50-95 contra FPS que muestra el compromiso entre
  precisión y velocidad.

El notebook 05 no necesita GPU, Drive ni el dataset: trabaja solo con los CSV versionados
en el repositorio, de modo que cualquiera puede regenerar la comparación.

Al notebook 02 se le agrega una celda que exporta el `metrics_summary.csv` del baseline a
partir de su `best.pt`, con `model.val()`. Sin ese paso el baseline no aparecería en la
comparación.

## Presupuesto de cómputo

Es el principal riesgo del plan. El baseline YOLOv8n tardó 2.4 horas en 30 épocas sobre
una GPU T4. Faster R-CNN con ResNet50-FPN es considerablemente más pesado; la estimación
es de 25 a 30 minutos por época con batch 4 y precisión mixta.

Por eso Faster R-CNN corre **12 épocas** (unas 5 a 6 horas) y RT-DETR-l corre **20
épocas**. Partiendo de pesos COCO, 12 épocas alcanzan para converger en un dataset de dos
clases.

Ese tiempo excede la duración de una sesión de Colab, así que el trainer guarda un
checkpoint completo en Drive al terminar cada época y, si al arrancar encuentra uno,
reanuda desde ahí. Es el mismo comportamiento que el notebook 02 ya usa con el
`resume=True` de Ultralytics.

Si el cómputo resulta insuficiente, el campo `backbone` de la configuración permite pasar
a `fasterrcnn_mobilenet_v3_large_fpn` sin tocar código.

## Dependencias

Se agregan a `requirements.txt`:

- `torchmetrics`, para el cálculo de mAP con protocolo COCO
- `pycocotools`, backend de evaluación que usa torchmetrics

## Validación

Los módulos de `src/` se prueban localmente, sin GPU, antes de ejecutar nada en Colab:

- Un test que construye un dataset sintético de pocas imágenes con cajas conocidas y
  verifica la conversión de coordenadas de YOLO a `xyxy`, el clampeo de coordenadas fuera
  de rango, el descarte de cajas degeneradas, el remapeo de clases y la forma de los
  tensores para una imagen sin cajas.
- Un smoke test que corre dos pasos de entrenamiento y una evaluación completa sobre ese
  dataset sintético en CPU, con `mobilenet_v3_large_fpn` para que sea rápido.

El objetivo es que el notebook de Colab no sea el primer lugar donde se descubre un error,
porque allí cada iteración cuesta el tiempo de descargar el dataset y montar el entorno.

## Fuera de alcance

- Búsqueda de hiperparámetros. Cada modelo corre una sola configuración.
- Evaluación sobre el split de test. Las métricas comparativas se calculan sobre
  validación; el test queda reservado para una evaluación final posterior.
- Versionado de pesos. Los `.pt` y `.pth` siguen excluidos por `.gitignore` y viven en
  Drive.
- El contenido de `demo/`, que no forma parte de este trabajo.
