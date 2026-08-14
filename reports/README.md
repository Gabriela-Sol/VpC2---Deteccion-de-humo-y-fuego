# Resultados

## Estructura

- `results/eda/` — tablas del análisis exploratorio (notebook 01).
- `results/<experimento>/` — una carpeta por experimento entrenado.
- `results/comparacion_modelos.csv` — tabla consolidada de todos los experimentos.
- `figures/eda/` — figuras del análisis exploratorio.
- `figures/comparacion/` — figuras comparativas entre modelos (notebook 05).

## Contenido de cada carpeta de experimento

| Archivo | Descripción |
|---|---|
| `results.csv` | métricas por época |
| `results.png` | curvas de pérdida y métricas |
| `confusion_matrix.png` | matriz de confusión (IoU 0.45, confianza 0.25) |
| `confusion_matrix_normalized.png` | la misma, normalizada por columna |
| `PR_curve.png` [^curvas] | curva precisión-recall por clase (IoU 0.5) |
| `F1_curve.png`, `P_curve.png`, `R_curve.png` [^curvas] | métricas en función de la confianza |
| `experiment_config_used.yaml` | configuración exacta con la que se corrió |
| `metrics_summary.csv` | resumen de una fila, con esquema común a todos los modelos |

[^curvas]: Las cuatro figuras de curvas se regeneran al volver a correr el
notebook correspondiente. En esta instantánea del repositorio no están
presentes para `yolov8n_baseline`.

## `metrics_summary.csv`

Es el archivo que hace posible comparar modelos entrenados con librerías
distintas: el `results.csv` de Ultralytics tiene columnas propias que no aplican
a torchvision. Todas las métricas se calculan sobre el split de **validación**
con protocolo COCO. Las columnas `precision`, `recall` y `f1` se reportan en el
umbral de confianza que maximiza F1, promediadas entre las clases (macro), y la
columna `params_M` cuenta los parámetros totales del modelo.

Para que las tres filas sean comparables, Faster R-CNN se construye con el mismo
presupuesto de detección que usa Ultralytics al validar (`conf=0.001`,
`max_det=300`) en vez de los valores por defecto de torchvision.

**Modelo reportado:** los modelos de Ultralytics (`yolov8n_baseline`, `rtdetr_l`)
se evalúan desde su `best.pt`, es decir la mejor época; el modelo de torchvision
(`fasterrcnn_r50fpn`) se evalúa en su **última** época, porque el bucle de
entrenamiento no hace seguimiento de la mejor.

## Experimentos

| Experimento | Familia | Notebook |
|---|---|---|
| `yolov8n_baseline` | Una etapa, CNN | `02_entrenamiento_YOLO.ipynb` |
| `fasterrcnn_r50fpn` | Dos etapas, CNN | `03_entrenamiento_FasterRCNN.ipynb` |
| `rtdetr_l` | Transformer | `04_entrenamiento_RTDETR.ipynb` |
