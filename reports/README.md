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
| `PR_curve.png` | curva precisión-recall por clase (IoU 0.5) |
| `F1_curve.png`, `P_curve.png`, `R_curve.png` | métricas en función de la confianza |
| `experiment_config_used.yaml` | configuración exacta con la que se corrió |
| `metrics_summary.csv` | resumen de una fila, con esquema común a todos los modelos |

## `metrics_summary.csv`

Es el archivo que hace posible comparar modelos entrenados con librerías
distintas: el `results.csv` de Ultralytics tiene columnas propias que no aplican
a torchvision. Todas las métricas se calculan sobre el split de **validación**
con protocolo COCO. Las columnas `precision`, `recall` y `f1` se reportan en el
umbral de confianza que maximiza F1.

## Experimentos

| Experimento | Familia | Notebook |
|---|---|---|
| `yolov8n_baseline` | Una etapa, CNN | `02_entrenamiento_YOLO.ipynb` |
| `fasterrcnn_r50fpn` | Dos etapas, CNN | `03_entrenamiento_FasterRCNN.ipynb` |
| `rtdetr_l` | Transformer | `04_entrenamiento_RTDETR.ipynb` |
