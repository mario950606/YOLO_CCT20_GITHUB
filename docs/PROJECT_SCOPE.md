# Alcance científico y correspondencia con la Pregunta 3

1. Preparación y auditoría del conjunto CCT20.
2. Entrenamiento YOLOv8s con decisiones documentadas.
3. Evaluación mAP@0.5, mAP@0.5:0.95 y métricas por especie.
4. Comparación de particiones; afirmaciones sobre ubicación vista/no vista requieren evidencia de `location_id` en JSON.
5. Grad-CAM exploratorio para inspeccionar atención espacial; no constituye prueba causal.
6. Robustez por tamaño y estudio piloto de oclusión (24 objetos).
7. Posprocesamiento de detecciones por cámara/ubicación, especie y fotografías vacías.

Los conteos de cajas son detecciones, no animales individuales. Las secuencias de cámara pueden contener múltiples fotografías del mismo animal.
