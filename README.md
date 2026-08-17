# Detección de humo y fuego con visión por computadora

Trabajo final para la materia VPCII que compara cinco arquitecturas de detección de
objetos, tres paradigmas distintos, sobre el mismo problema: identificar
humo y fuego en imágenes, usando el dataset público **D-Fire**.

## Objetivo y motivación

La detección temprana de incendios es un problema relevante para sistemas de
monitoreo, seguridad y prevención de riesgos: identificar humo o fuego a
partir de cámaras de vigilancia permite reaccionar antes de que el incendio
se propague, a diferencia de los sensores físicos tradicionales (temperatura,
partículas), que necesitan que el fuego los alcance para dispararse.

Este proyecto entrena y evalúa, bajo un protocolo experimental común (mismo
dataset, mismas particiones, mismas métricas), cinco detectores que
representan tres paradigmas distintos de visión por computadora:

| Paradigma | Modelos |
|---|---|
| Una etapa (YOLO) | YOLOv8n, YOLO11n, YOLO26s |
| Dos etapas | Faster R-CNN (ResNet50-FPN v2) |
| Transformer end-to-end | RT-DETR-L |

La pregunta que se busca responder no es solo "qué modelo da el mejor mAP",
sino cómo se compensan precisión, velocidad de inferencia y costo
computacional entre arquitecturas de naturaleza muy distinta, y qué papel
juegan la calidad de las anotaciones y el tamaño de los objetos en ese
resultado. El análisis completo, con metodología y discusión de resultados,
está en [`paper/paper.pdf`](paper/paper.pdf).

### Resultados

Sobre el split de test, los cinco modelos quedan dentro de un rango angosto
de mAP@0.5 (0.749–0.773), pero con una diferencia de hasta ~66x en velocidad
de inferencia entre el más rápido (YOLOv8n, ~457 FPS) y el más lento (Faster
R-CNN, ~7 FPS):

| Experimento | Familia | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | F1 | FPS | Params (M) |
|---|---|---|---|---|---|---|---|---|
| `yolo26s_baseline` | YOLO26 | 0.773 | 0.443 | 0.786 | 0.700 | 0.740 | 179.7 | 9.95 |
| `yolov8n_baseline` | YOLOv8 | 0.759 | 0.434 | 0.766 | 0.688 | 0.725 | 457.2 | 3.01 |
| `yolo11n_baseline` | YOLO11 | 0.753 | 0.433 | 0.752 | 0.690 | 0.720 | 423.0 | 2.59 |
| `rtdetr_l` | Transformer | 0.751 | 0.415 | 0.757 | 0.690 | 0.722 | 29.5 | 31.99 |
| `fasterrcnn_r50fpn` | Dos etapas | 0.749 | 0.403 | 0.786 | 0.726 | 0.755 | 7.0 | 43.26 |

Tabla completa: [`reports/results/comparacion_modelos.csv`](reports/results/comparacion_modelos.csv).
Todas las corridas usan GPU Tesla T4 (Colab).

## Dataset

**D-Fire** (Venâncio et al., *"An automatic fire detection system based on
deep convolutional neural networks for low-power, resource-constrained
devices"*, Neural Computing and Applications, 2022): 21 527 imágenes con
cuatro tipos de escena (solo humo, solo fuego, humo y fuego simultáneos, y
negativas), anotadas en formato YOLO con dos clases (`smoke`, `fire`).

- Train: 14 122 imágenes · Validación: 3 099 · Test: 4 306.
- Descarga usada en este repo (vía [`kagglehub`](https://pypi.org/project/kagglehub/)):
  **[sayedgamal99/smoke-fire-detection-yolo](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)**.

## Arquitectura del repositorio

```
├── configs/
│   ├── dataset/dfire_colab.yaml     # YAML de dataset (formato Ultralytics) usado en Colab
│   └── experiments/*.yaml           # Un YAML por experimento: modelo + hiperparámetros
├── data/                            # Vacío en el repo (ignorado por git); reservado para datos locales
├── demo/                            # Reservado para materiales de demostración
├── notebooks/
│   ├── 01_exploracion_dfire.ipynb          # EDA: distribución de clases, tamaños de caja, splits
│   ├── 02_entrenamiento_YOLOv8.ipynb       # Entrenamiento y evaluación de YOLOv8n
│   ├── 03_entrenamiento_YOLO11.ipynb       # Entrenamiento y evaluación de YOLO11n
│   ├── 04_entrenamiento_FasterRCNN.ipynb   # Entrenamiento y evaluación de Faster R-CNN (torchvision)
│   ├── 05_entrenamiento_RTDETR.ipynb       # Entrenamiento y evaluación de RT-DETR-L
│   ├── 06_entrenamiento_YOLO26s.ipynb      # Entrenamiento y evaluación de YOLO26s
│   ├── 07_comparacion_modelos.ipynb        # Consolida metrics_summary.csv de todos los modelos
│   └── 08_comparacion_visual.ipynb         # Comparación cualitativa (predicciones sobre imágenes de test)
├── paper/
│   ├── paper.tex                    # Informe (formato IEEE) con estado del arte, metodología y resultados
│   └── paper.pdf
├── reports/
│   ├── figures/                     # Figuras usadas en el paper (EDA y comparación entre modelos)
│   └── results/
│       ├── eda/                             # Tablas del análisis exploratorio (notebook 01)
│       ├── <experimento>/                   # Una carpeta por experimento entrenado (ver detalle abajo)
│       ├── image_comparison/                # Comparación visual por caso (notebook 08)
│       ├── comparacion_modelos.csv          # Tabla consolidada de todos los experimentos
│       └── comparacion_*.png                # Figuras comparativas (mAP, precision/recall/F1)
├── src/
│   ├── data/
│   │   ├── kaggle_inputs.py         # Recupera un checkpoint de una corrida previa en Kaggle
│   │   └── yolo_dataset.py          # Dataset en formato YOLO adaptado a la API de torchvision
│   ├── engine/
│   │   ├── trainer.py               # Bucle de entrenamiento y checkpointing para modelos de torchvision
│   │   ├── metrics.py               # mAP (COCO, vía torchmetrics), curvas P/R/F1, FPS de inferencia
│   │   └── matching.py              # Emparejamiento predicción↔ground truth por IoU y matriz de confusión
│   ├── modeling/detectors.py        # Construcción de Faster R-CNN (torchvision) con cabezal ajustado
│   └── reporting/
│       ├── experiment_report.py     # De un modelo entrenado a la carpeta de resultados completa
│       ├── history.py               # Junta el results.csv de todos los experimentos en un esquema común
│       ├── plots.py                 # Todas las figuras (curvas, matriz de confusión, comparaciones)
│       └── summary.py               # Esquema y validación de metrics_summary.csv
├── tests/                           # Suite de pytest para src/ (fixtures con dataset sintético)
├── requirements.txt
└── pytest.ini
```

- Los notebooks **02, 03, 05 y 06** entrenan con la API de **Ultralytics**
  (YOLOv8n, YOLO11n, RT-DETR-L, YOLO26s respectivamente) y generan sus
  artefactos con las funciones propias de esa librería (el 05 usa además
  `src/reporting` para escribir el resumen con el esquema común).
- El notebook **04** entrena **Faster R-CNN** con `torchvision`, usando el
  código de `src/` (`engine/trainer.py`, `engine/metrics.py`,
  `engine/matching.py`, `modeling/detectors.py`) porque torchvision no trae
  un bucle de entrenamiento ni reporte de métricas prearmados como
  Ultralytics.
- `src/reporting/experiment_report.py` es el punto donde convergen los
  resultados de Faster R-CNN: genera los mismos artefactos
  (`results.csv`, `confusion_matrix.png`, curvas P/R/F1,
  `metrics_summary.csv`, etc.) que produce Ultralytics, para que los modelos
  de ambas librerías queden comparables en `reports/results/`.
- El notebook **07** no entrena nada: junta los `metrics_summary.csv` y
  `results.csv` de todos los experimentos (con `src/reporting/summary.py` y
  `src/reporting/history.py`) y produce la tabla y las figuras comparativas
  del paper.
- El notebook **08** tampoco entrena: carga los pesos entrenados (`best.pt`
  de cada corrida, desde Google Drive) y corre inferencia sobre imágenes del
  split de test agrupadas en cinco casos (solo humo, solo fuego, ambos,
  objetos pequeños, negativas) para la comparación cualitativa.
- `reports/results/<experimento>/` es la carpeta de salida estándar de cada
  entrenamiento. Contiene, entre otros: `args.yaml`/`experiment_config_used.yaml`
  (configuración exacta usada), `results.csv`/`results.png` (curvas por
  época), `confusion_matrix(_normalized).png`, las curvas P/R/F1, y
  `metrics_summary.csv` (una fila con el esquema común definido en
  `src/reporting/summary.py`, que es lo que hace comparables modelos
  entrenados con librerías distintas). Los pesos (`best.pt`/`last.pt`) **no**
  se versionan (ver `.gitignore`): viven en Google Drive durante el
  entrenamiento en Colab.

## Instalación

Python 3.12 (el runtime de Colab) y `ultralytics>=8.3,<8.5`.

```bash
git clone https://github.com/Gabriela-Sol/VpC2---Deteccion-de-humo-y-fuego.git
cd VpC2---Deteccion-de-humo-y-fuego

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Para correr la suite de tests:

```bash
pytest
```

Los tests usan un dataset sintético que se genera en un directorio temporal
(`tests/conftest.py`) y no requieren GPU ni descargar D-Fire.

## Cómo reproducir un entrenamiento

Los notebooks están pensados para correr en **Google Colab** (varios traen
el botón "Open in Colab") y detectan el entorno para adaptarse solos (Colab,
Kaggle Notebooks o GPU local). En términos generales, cada notebook de
entrenamiento:

1. Clona este repositorio (o usa el checkout local si no corre en Colab).
2. Descarga el dataset D-Fire desde Kaggle vía `kagglehub` (queda en el caché
   de esa librería, no en `data/`).
3. Arma o carga la configuración del experimento (`configs/experiments/*.yaml`
   o, en YOLO11/YOLO26s, un diccionario equivalente definido en el propio
   notebook).
4. Entrena (con `RUN_TRAINING = False` por defecto, para no relanzar horas de
   entrenamiento al solo querer inspeccionar resultados ya generados).
5. Evalúa el modelo final sobre el split de test (`best.pt` en los modelos de
   Ultralytics; en Faster R-CNN, la última época, porque su bucle de
   entrenamiento no hace seguimiento de la mejor) y exporta los artefactos a
   `reports/results/<experimento>/`.

En Colab, los notebooks montan **Google Drive** (piden autorización al
ejecutar la celda de `drive.mount`) y guardan las corridas —pesos incluidos—
en `MyDrive/VCII_DFire/runs/`; a `reports/results/` del repo solo se copian
los artefactos livianos (métricas y figuras).

Para entrenar desde cero, cambiar `RUN_TRAINING = True` en la celda
correspondiente. Cada entrenamiento con `epochs=50` (YOLOv8n, YOLO11n,
YOLO26s) tardó entre ~4 y ~5 horas sobre una GPU T4; con `epochs=30`,
RT-DETR-L tardó ~9.4h y Faster R-CNN ~18.5h.

## Inferencia con un modelo entrenado

Los cuatro modelos entrenados con Ultralytics se usan igual, cargando el
checkpoint `best.pt` de la corrida que se quiera: YOLOv8n, YOLO11n y YOLO26s
con la clase `YOLO`, y RT-DETR-L con la clase `RTDETR`:

```python
from ultralytics import YOLO, RTDETR

model = YOLO("reports/results/yolo11n_baseline/weights/best.pt")
# Para RT-DETR-L: model = RTDETR("ruta/a/rtdetr_l/weights/best.pt")

resultados = model.predict(
    source="ruta/a/una/imagen_o_carpeta",
    imgsz=640,
    conf=0.25,
)

for resultado in resultados:
    resultado.show()          # o resultado.save(filename="salida.jpg")
    print(resultado.boxes)    # cajas, clases (0=smoke, 1=fire) y confianza
```

Para Faster R-CNN (torchvision), el checkpoint se carga con
`src/engine/trainer.load_checkpoint` sobre un modelo construido con
`src/modeling/detectors.build_fasterrcnn`:

```python
import torch
from src.modeling.detectors import build_fasterrcnn
from src.engine.trainer import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_fasterrcnn(num_classes=3, pretrained=False)  # 3 = fondo + smoke + fire
load_checkpoint("ruta/al/checkpoint.pt", model, device=device)
model.eval().to(device)
```

> Los pesos (`*.pt`) no están commiteados en el repositorio (ver
> `.gitignore`). Para obtenerlos hay que correr el notebook de entrenamiento
> correspondiente (con `RUN_TRAINING = True`, quedan en Google Drive) o usar
> una copia local que ya se haya generado antes.

## Informe

El análisis completo —estado del arte, metodología, resultados por modelo y
conclusiones— está en [`paper/paper.tex`](paper/paper.tex) /
[`paper/paper.pdf`](paper/paper.pdf).
