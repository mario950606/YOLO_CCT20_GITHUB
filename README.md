# YOLO_CCT20 — Monitoreo de fauna con cámaras trampa

Repositorio reproducible para entrenar, evaluar e interpretar YOLOv8s sobre CCT20 y producir un resumen de actividad por cámara.

## Estructura

```text
YOLO_CCT20/
├── main.py
├── configs/config.example.yaml
├── src/01_prepare_dataset.py
├── src/02_train_yolo.py
├── src/03_evaluate_model.py
├── src/04_generalization.py
├── src/05_gradcam.py
├── src/06_robustness.py
├── src/07_ecological_monitoring.py
├── requirements.txt
└── docs/
```

## 1. Preparación

```bash
git clone <URL_DEL_REPOSITORIO>
cd YOLO_CCT20
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
```

Edite `configs/config.yaml`: indique las rutas del CCT20, dataset YOLO, pesos `best.pt`, resultados, CSV de oclusión y metadatos. `config.yaml` está excluido de Git para no publicar rutas privadas.

El dataset y los pesos no se incluyen en GitHub. Deben estar disponibles localmente o en el almacenamiento del clúster.

## 2. Ejecución segura

Comprobar la fase de entrenamiento sin entrenar:

```bash
python main.py --fase 2
```

Entrenar solo con autorización explícita:

```bash
python main.py --fase 2 --train
```

Reutilizar métricas guardadas (sin volver a ejecutar `model.val`):

```bash
python main.py --fase 3
python main.py --fase 4
```

Autorizar una evaluación nueva:

```bash
python main.py --fase 3 --evaluate
python main.py --fase 4 --evaluate
```

Interpretabilidad y robustez:

```bash
python main.py --fase 5
python main.py --fase 6
```

La fase 6 requiere el CSV manual original de 24 objetos y los CSV de métricas por tamaño; no los genera ni los reemplaza.

Monitoreo ecológico:

```bash
python main.py --fase 7 --target-location <ID_UBICACION_NO_VISTA>
```

La fase 7 comprueba los metadatos de `train` y de la partición elegida. Se detiene si no puede demostrar que la ubicación seleccionada no aparece en entrenamiento, o si no puede ordenar cronológicamente las imágenes. No se debe asumir que `trans_test` significa automáticamente cámara no vista.

## 3. Pipeline

`python main.py --fase all` ejecuta secuencialmente los módulos en modo conservador. No autoriza entrenamiento ni evaluación nueva. La fase 6 puede requerir insumos externos (CSV manual y métricas por tamaño) y la fase 7 exige un ID de ubicación no vista. En la práctica, ejecute y valide cada fase por separado antes de usar `all`.

## 4. Particiones, etiquetas vacías y métricas

Se conservan las 14 clases de fauna. Los archivos de etiqueta vacíos representan imágenes sin fauna y son válidos como fondo; no se convierten en una clase adicional. El reporte de monitoreo distingue imágenes vacías de imágenes con detecciones.

CIS/TRANS se reportan inicialmente como particiones. Solo se deben describir como cámaras/ubicaciones vistas o no vistas después de contrastar los identificadores de ubicación de los JSON originales.

Las cajas detectadas por fotograma no son individuos independientes. El conteo de detecciones no es una estimación de abundancia poblacional.

## 5. Limitaciones conocidas que requieren validación

- `05_gradcam.py` usa una capa interna del modelo y una API de Ultralytics que puede variar entre versiones. Compruebe la capa y revise los mapas generados.
- La fase 6 utiliza una muestra piloto manual de 24 objetos; no es una muestra estadísticamente representativa.
- La fase 7 necesita que los JSON contengan metadatos de ubicación y fecha/hora utilizables. Si esos campos faltan o tienen otros nombres, adapte explícitamente los campos; el módulo falla de forma segura.
- Las versiones del entorno del clúster deben verificarse antes de congelar dependencias para una reproducción exacta.

## 6. Recomendación para GitHub

Suba código, documentación y configuración de ejemplo. No suba el dataset CCT20 completo, imágenes, etiquetas, `best.pt`, pesos iniciales ni resultados pesados. Use Git LFS o un repositorio de artefactos solo si existe una necesidad explícita y autorización para alojar pesos.
