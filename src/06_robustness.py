#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_robustness.py — YOLO_CCT20, Fases 5 y 6.

Integra:
  A) Auditoría estructural de cis_test/trans_test del dataset final.
  B) Reutilización de métricas de robustez por tamaño ya calculadas.
  C) Estudio piloto de oclusión con las 24 anotaciones manuales existentes.
  D) Inferencia YOLO limitada a esas 24 imágenes y matching GT-predicción.
  E) Resúmenes por nivel de oclusión y partición, más informe.

Protecciones:
  - Nunca genera ni sobrescribe anotacion_rapida_24.csv.
  - No utiliza la muestra antigua de 619 objetos.
  - No entrena el modelo.
  - No recalcula las métricas de tamaño.
  - Guarda cada ejecución en un directorio nuevo.
  - Cachea las predicciones de las 24 imágenes; para repetirlas use
    explícitamente --force-inference.

La muestra de 24 objetos es piloto/exploratoria, no representativa.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

PROJECT_DEFAULT = Path(".")
DATASET_DEFAULT = PROJECT_DEFAULT / "data/dataset_yolo"
DATASET_FASE6_DEFAULT = Path("data/dataset_yolo")
MODEL_DEFAULT = Path("weights/best.pt")
BASE_YOLO_DEFAULT = Path("data")
CSV_OCC24_DEFAULT = (
    BASE_YOLO_DEFAULT / "resultados_YOLO_CCT20/fase6_oclusiones/"
    "revision_rapida_24/anotacion_rapida_24.csv"
)
SIZE_METRICS_DEFAULT = (
    BASE_YOLO_DEFAULT / "resultados_YOLO_CCT20/fase5_metricas_por_clase"
)
RESULTS_DEFAULT = PROJECT_DEFAULT / "results"

SPLITS = ("cis_test", "trans_test")
CLASSES = [
    "badger", "bird", "bobcat", "cat", "coyote", "deer", "dog", "fox",
    "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
VALID_OCCLUSION = {"baja", "media", "alta", "no_evaluable"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REQUIRED_OCC_COLUMNS = {
    "id_revision", "conjunto", "file_name", "clase", "categoria_tamano",
    "image_path", "indice_caja_gt", "gt_xc", "gt_yc", "gt_w", "gt_h",
    "nivel_oclusion", "evaluable"
}
SIZE_FILES = {
    "resumen_tamano_objetos.csv": "resumen_tamano_reutilizado.csv",
    "robustez_por_tamano.csv": "robustez_por_tamano_reutilizada.csv",
    "falsos_positivos_globales.csv": "falsos_positivos_globales_reutilizados.csv",
    "recall_por_especie_y_tamano.csv": "recall_por_especie_y_tamano_reutilizado.csv",
}


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(dataset: Path, output: Path) -> list[dict[str, Any]]:
    """Audita imágenes/labels de los dos splits; label vacío es válido."""
    rows = []
    for split in SPLITS:
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            rows.append({
                "split": split, "images": 0, "labels": 0, "empty_labels": 0,
                "images_without_label": "", "labels_without_image": "",
                "malformed_lines": "", "invalid_classes": "",
                "invalid_coordinates": "", "status": "MISSING_DIRECTORY"
            })
            continue

        images = [p for p in image_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        labels = list(label_dir.rglob("*.txt"))
        image_ids = {p.relative_to(image_dir).with_suffix("").as_posix() for p in images}
        label_ids = {p.relative_to(label_dir).with_suffix("").as_posix() for p in labels}
        missing, orphan = image_ids - label_ids, label_ids - image_ids
        empty = malformed = invalid_class = invalid_coord = 0

        for label in labels:
            if label.stat().st_size == 0:
                empty += 1
                continue
            try:
                lines = label.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                malformed += 1
                continue
            for line in lines:
                fields = line.split()
                if len(fields) != 5:
                    malformed += 1
                    continue
                try:
                    cls = int(fields[0])
                    x, y, w, h = map(float, fields[1:])
                    invalid_class += int(cls < 0 or cls >= len(CLASSES))
                    invalid_coord += int(
                        not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h))
                        or w <= 0 or h <= 0
                    )
                except (ValueError, TypeError):
                    malformed += 1

        status = "OK" if not (
            missing or orphan or malformed or invalid_class or invalid_coord
        ) else "REVIEW"
        rows.append({
            "split": split, "images": len(images), "labels": len(labels),
            "empty_labels": empty, "images_without_label": len(missing),
            "labels_without_image": len(orphan), "malformed_lines": malformed,
            "invalid_classes": invalid_class, "invalid_coordinates": invalid_coord,
            "status": status
        })
    write_csv(output / "auditoria_particiones.csv", rows)
    return rows


def load_manual_annotations(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el CSV manual de 24 objetos: {csv_path}\n"
            "Ubique el archivo anotacion_rapida_24.csv original. "
            "No ejecute nuevamente el generador de muestra."
        )
    # sep=None permite detectar coma o punto y coma.
    df = pd.read_csv(csv_path, sep=None, engine="python")
    missing = REQUIRED_OCC_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Al CSV manual le faltan columnas: {sorted(missing)}")
    if len(df) != 24:
        raise ValueError(
            f"Se esperaban exactamente 24 anotaciones manuales; se encontraron {len(df)}. "
            "No se continuará para evitar sustituir la muestra acordada."
        )
    if df["id_revision"].astype(str).duplicated().any():
        raise ValueError("id_revision contiene duplicados.")
    if df["file_name"].astype(str).duplicated().any():
        raise ValueError("file_name contiene duplicados; la asociación imagen-objeto no es única.")

    df["id_revision"] = df["id_revision"].astype(str).str.strip()
    df["conjunto"] = df["conjunto"].astype(str).str.strip()
    df["file_name"] = df["file_name"].astype(str).str.strip()
    df["clase"] = df["clase"].astype(str).str.strip().str.lower()
    df["categoria_tamano"] = df["categoria_tamano"].astype(str).str.strip().str.lower()
    df["nivel_oclusion"] = df["nivel_oclusion"].astype(str).str.strip().str.lower()
    df["evaluable"] = df["evaluable"].astype(str).str.strip().str.lower()

    if not set(df["conjunto"]).issubset(SPLITS):
        raise ValueError(f"El CSV contiene particiones no esperadas: {set(df['conjunto']) - set(SPLITS)}")
    if not set(df["nivel_oclusion"]).issubset(VALID_OCCLUSION):
        raise ValueError("Hay valores no reconocidos en nivel_oclusion.")
    if not set(df["clase"]).issubset(CLASS_TO_ID):
        raise ValueError(f"Clases no reconocidas: {set(df['clase']) - set(CLASS_TO_ID)}")

    # Comprobación de asociación única, si el campo existe en el CSV.
    if "estado_asociacion" in df.columns:
        bad = df["estado_asociacion"].astype(str) != "asociacion_unica_por_clase_y_dimensiones"
        if bad.any():
            raise ValueError(
                "Hay GT cuya asociación no fue declarada única. Revise el CSV manual "
                "antes de comparar con predicciones."
            )

    numeric_cols = ["gt_xc", "gt_yc", "gt_w", "gt_h"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all():
        raise ValueError("Las coordenadas GT contienen valores no finitos.")
    if ((df[["gt_xc", "gt_yc"]] < 0) | (df[["gt_xc", "gt_yc"]] > 1)).any().any():
        raise ValueError("Centro GT fuera del rango normalizado [0,1].")
    if ((df[["gt_w", "gt_h"]] <= 0) | (df[["gt_w", "gt_h"]] > 1)).any().any():
        raise ValueError("Ancho/alto GT fuera del rango (0,1].")
    return df


def resolve_image(row: pd.Series, dataset: Path) -> tuple[Path, str]:
    final_path = dataset / "images" / row["conjunto"] / row["file_name"]
    reference_path = Path(str(row["image_path"])).expanduser()
    if final_path.is_file():
        if reference_path.is_file():
            with Image.open(final_path) as a, Image.open(reference_path) as b:
                if a.size != b.size:
                    raise ValueError(
                        f"Dimensiones diferentes en {row['file_name']}: "
                        f"dataset final={a.size}, referencia Fase 6={b.size}. "
                        "No se transfiere la caja GT."
                    )
                # Comprobación adicional de identidad de píxeles para evitar
                # usar una anotación sobre otra imagen de igual tamaño.
                same = False
                try:
                    same = np.array_equal(np.asarray(a.convert("RGB")),
                                          np.asarray(b.convert("RGB")))
                except Exception:
                    same = False
                if not same:
                    raise ValueError(
                        f"La imagen {row['file_name']} tiene las mismas dimensiones "
                        "pero distinto contenido entre Fase 6 y dataset final. "
                        "No se evalúa esa anotación automáticamente."
                    )
        return final_path, "dataset_final_fase5"
    if reference_path.is_file():
        return reference_path, "dataset_base_fase6"
    raise FileNotFoundError(
        f"No se encuentra {row['file_name']} en el dataset final ni en image_path."
    )


def resolve_all_images(df: pd.DataFrame, dataset: Path) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        path, origin = resolve_image(row, dataset)
        records.append({
            "id_revision": row["id_revision"],
            "image_path_inferencia": str(path),
            "origen_imagen": origin
        })
    return df.merge(pd.DataFrame(records), on="id_revision", validate="one_to_one")


def xywhn_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    xc, yc, w, h = box
    return [
        (xc - w / 2) * width, (yc - h / 2) * height,
        (xc + w / 2) * width, (yc + h / 2) * height
    ]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_or_run_predictions(df: pd.DataFrame, model_path: Path, cache_path: Path,
                            force: bool, imgsz: int, conf: float, iou: float,
                            device: str) -> dict[str, list[dict[str, Any]]]:
    if cache_path.is_file() and not force:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("model_path") != str(model_path):
            raise ValueError(
                "El cache de predicciones fue generado con otra ruta de modelo. "
                "Use --force-inference para recalcular con el modelo actual."
            )
        cache = payload.get("predictions", {})
        expected = {str(p) for p in df["image_path_inferencia"]}
        if set(cache) != expected:
            raise ValueError(
                "El cache no contiene exactamente las 24 imágenes anotadas. "
                "Use --force-inference para regenerarlo."
            )
        print("Reutilizando predicciones cacheadas:", cache_path)
        return cache

    if not model_path.is_file():
        raise FileNotFoundError(f"No se encuentra best.pt: {model_path}")
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    paths = df["image_path_inferencia"].astype(str).tolist()
    # Inferencia limitada a las 24 imágenes; resultados guardados en JSON,
    # sin depender de archivos TXT ausentes para imágenes sin detecciones.
    results = model.predict(
        source=paths, imgsz=imgsz, conf=conf, iou=iou, device=device,
        save=False, verbose=False, stream=False
    )
    if len(results) != len(paths):
        raise RuntimeError(
            f"Ultralytics devolvió {len(results)} resultados para {len(paths)} imágenes."
        )

    predictions: dict[str, list[dict[str, Any]]] = {}
    for path, result in zip(paths, results):
        items = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(xyxy, cls, confs):
                items.append({
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "bbox_xyxy": [float(v) for v in box]
                })
        predictions[path] = items

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "model_path": str(model_path),
        "imgsz": imgsz, "conf": conf, "iou_nms": iou, "device": device,
        "number_images": len(paths),
        "predictions": predictions,
        "created_at": datetime.now().astimezone().isoformat()
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Predicciones de {len(paths)} imágenes guardadas en:", cache_path)
    return predictions


def evaluate_manual_objects(df: pd.DataFrame,
                            predictions: dict[str, list[dict[str, Any]]],
                            iou_threshold: float) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        image_path = Path(row.image_path_inferencia)
        with Image.open(image_path) as im:
            width, height = im.size

        class_id = CLASS_TO_ID[row.clase]
        gt = xywhn_to_xyxy(
            [float(row.gt_xc), float(row.gt_yc), float(row.gt_w), float(row.gt_h)],
            width, height
        )
        image_preds = predictions.get(str(image_path), [])
        candidates = []
        for pred_idx, pred in enumerate(image_preds):
            if int(pred["class_id"]) != class_id:
                continue
            overlap = iou_xyxy(gt, pred["bbox_xyxy"])
            if overlap >= iou_threshold:
                candidates.append((overlap, float(pred["confidence"]), pred_idx))
        # Solo hay una GT anotada por imagen en el CSV manual: se asigna
        # como TP la predicción de misma clase con mayor IoU; desempate conf.
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        matched = bool(candidates)
        best_iou = candidates[0][0] if matched else 0.0
        best_conf = candidates[0][1] if matched else None

        rows.append({
            "id_revision": row.id_revision,
            "conjunto": row.conjunto,
            "file_name": row.file_name,
            "clase": row.clase,
            "categoria_tamano": row.categoria_tamano,
            "nivel_oclusion": row.nivel_oclusion,
            "evaluable": row.evaluable,
            "cuerpo_visible_estimado_pct": getattr(row, "cuerpo_visible_estimado_pct", None),
            "fuera_de_encuadre": getattr(row, "fuera_de_encuadre", None),
            "borrosidad": getattr(row, "borrosidad", None),
            "iluminacion_deficiente": getattr(row, "iluminacion_deficiente", None),
            "origen_imagen": row.origen_imagen,
            "ancho_imagen_px": width,
            "alto_imagen_px": height,
            "numero_predicciones_imagen": len(image_preds),
            "numero_predicciones_misma_clase": sum(
                int(p["class_id"]) == class_id for p in image_preds
            ),
            "detectado": int(matched),
            "falso_negativo": int(not matched),
            "best_iou": float(best_iou),
            "confianza_prediccion": best_conf,
            "iou_umbral": iou_threshold
        })
    return pd.DataFrame(rows)


def summarize_occlusion(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Los tres no_evaluable se conservan en detalle y distribución,
    # pero se excluyen exclusivamente de los resúmenes por nivel.
    valid = detail[detail["nivel_oclusion"].isin(["baja", "media", "alta"])].copy()
    by_split = (valid.groupby(["conjunto", "nivel_oclusion"], dropna=False)
                .agg(objetos=("detectado", "size"),
                     TP=("detectado", "sum"),
                     FN=("falso_negativo", "sum"))
                .reset_index())
    if not by_split.empty:
        by_split["recall_pct"] = 100 * by_split["TP"] / by_split["objetos"]
        by_split["tasa_omision_pct"] = 100 * by_split["FN"] / by_split["objetos"]

    global_level = (valid.groupby("nivel_oclusion", dropna=False)
                    .agg(objetos=("detectado", "size"),
                         TP=("detectado", "sum"),
                         FN=("falso_negativo", "sum"))
                    .reset_index())
    if not global_level.empty:
        global_level["recall_pct"] = 100 * global_level["TP"] / global_level["objetos"]
        global_level["tasa_omision_pct"] = 100 * global_level["FN"] / global_level["objetos"]

    distribution = (detail.groupby(["nivel_oclusion", "evaluable"], dropna=False)
                    .size().rename("objetos").reset_index())
    return by_split, global_level, distribution


def reuse_size_metrics(source_dir: Path, output: Path) -> list[dict[str, Any]]:
    status = []
    size_out = output / "robustez_tamano"
    size_out.mkdir(parents=True, exist_ok=True)
    for source_name, dest_name in SIZE_FILES.items():
        source = source_dir / source_name
        if source.is_file():
            shutil.copy2(source, size_out / dest_name)
            status.append({
                "archivo_origen": str(source),
                "archivo_copia": str(size_out / dest_name),
                "disponible": True
            })
        else:
            status.append({
                "archivo_origen": str(source),
                "archivo_copia": "",
                "disponible": False
            })
    write_csv(output / "disponibilidad_metricas_tamano.csv", status)
    return status


def optional_global_metrics(results: Path, output: Path) -> str | None:
    candidates = [
        results / "02_evaluacion_modelo" / "metricas_globales_cis_trans.csv",
        results / "02_evaluacion_modelo" / "evaluacion" / "metricas_globales_cis_trans.csv",
        results / "metricas_globales_cis_trans.csv",
    ]
    for p in candidates:
        if p.is_file():
            shutil.copy2(p, output / "metricas_globales_fase5_reutilizadas.csv")
            return str(p)
    return None


def make_occlusion_plot(global_level: pd.DataFrame, output: Path) -> None:
    if global_level.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["baja", "media", "alta"]
    df = global_level.set_index("nivel_oclusion").reindex(order).dropna(subset=["recall_pct"])
    if df.empty:
        return
    ax = df["recall_pct"].plot(kind="bar", figsize=(7, 4), ylim=(0, 100))
    ax.set_title("Recall piloto según nivel de oclusión")
    ax.set_ylabel("Recall (%)")
    ax.set_xlabel("Nivel de oclusión")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "recall_por_nivel_oclusion.png", dpi=180)
    plt.close()


def make_size_plot(output: Path) -> None:
    path = output / "robustez_tamano" / "robustez_por_tamano_reutilizada.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    needed = {"conjunto", "tamano", "recall_pct"}
    if not needed.issubset(df.columns):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="tamano", columns="conjunto", values="recall_pct", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(8, 4), ylim=(0, 100))
    ax.set_title("Recall por tamaño del objeto (métricas reutilizadas)")
    ax.set_ylabel("Recall (%)")
    ax.set_xlabel("Categoría de tamaño")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "recall_por_tamano_reutilizado.png", dpi=180)
    plt.close()


def write_report(output: Path, detail: pd.DataFrame,
                 by_split: pd.DataFrame, global_level: pd.DataFrame,
                 size_status: list[dict[str, Any]], global_source: str | None,
                 csv_source: Path, model_path: Path) -> None:
    lines = [
        "# YOLO_CCT20 — Robustez (Fases 5 y 6)", "",
        f"- Fecha: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Modelo: `{model_path}`",
        f"- CSV manual de oclusión: `{csv_source}`",
        f"- Métricas globales Fase 5 reutilizadas: `{global_source or 'No encontradas'}`",
        "",
        "## 1. Robustez según tamaño",
        "",
        "Se reutilizan los CSV existentes de Fase 5/Fase 6. Este módulo no vuelve "
        "a calcular métricas por tamaño.",
        "",
    ]
    for item in size_status:
        lines.append(
            f"- `{Path(item['archivo_origen']).name}`: "
            f"{'disponible y copiado' if item['disponible'] else 'no encontrado'}."
        )
    lines += [
        "", "## 2. Estudio piloto según oclusión", "",
        f"Objetos manuales conservados: **{len(detail)}**.",
        f"Detectados: **{int(detail['detectado'].sum())}**; "
        f"no detectados: **{int(detail['falso_negativo'].sum())}**.",
        "",
        "| Nivel | Objetos | TP | FN | Recall (%) | Omisión (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in global_level.to_dict(orient="records"):
        lines.append(
            f"| {r['nivel_oclusion']} | {r['objetos']} | {r['TP']} | {r['FN']} "
            f"| {r['recall_pct']:.2f} | {r['tasa_omision_pct']:.2f} |"
        )
    lines += [
        "", "### Resultados por partición", "",
        "| Partición | Nivel | Objetos | TP | FN | Recall (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in by_split.to_dict(orient="records"):
        lines.append(
            f"| {r['conjunto']} | {r['nivel_oclusion']} | {r['objetos']} "
            f"| {r['TP']} | {r['FN']} | {r['recall_pct']:.2f} |"
        )
    excluded = int((detail["nivel_oclusion"] == "no_evaluable").sum())
    lines += [
        "", f"Los {excluded} objetos `no_evaluable` permanecen en el detalle y "
        "en la tabla de distribución, pero no se incluyen en las tasas por nivel.",
        "",
        "## 3. Limitaciones metodológicas", "",
        "El análisis de oclusión utiliza 24 objetos anotados manualmente (12 baja, "
        "5 media, 4 alta y 3 no_evaluable, según la muestra acordada). Es una "
        "muestra piloto y no representa necesariamente la distribución real de "
        "oclusiones del conjunto CCT20.",
        "",
        "Una anotación GT se considera detectada cuando existe al menos una "
        "predicción de la misma clase con IoU ≥ 0.50. Cada registro manual "
        "corresponde a una sola caja GT por imagen; se selecciona la predicción "
        "con mayor IoU (desempate por confianza). El resultado es un recall "
        "descriptivo de la muestra, no mAP ni una evaluación completa de todas "
        "las cajas presentes en esas imágenes.",
        "",
        "La ausencia de una detección se registra como FN. Las predicciones "
        "adicionales no se contabilizan aquí como FP; para estudiar FP se deben "
        "comparar todas las predicciones con todas las GT de cada imagen.",
        "",
        "## 4. Archivos", "",
        "- `detalle_evaluacion_24.csv`: resultado por anotación manual.",
        "- `resumen_oclusion_por_conjunto.csv`: métricas por split y nivel.",
        "- `resumen_oclusion_global.csv`: métricas globales por nivel.",
        "- `distribucion_evaluable_oclusion.csv`: conserva `no_evaluable` y estado.",
        "- `predicciones_24.json`: predicciones por imagen (cache).",
        "- `robustez_tamano/`: copias de resultados previos, si están disponibles.",
        "- `informe_robustez.md`: este informe.",
        "",
        "No se modificó el CSV original de anotaciones, el dataset ni los pesos."
    ]
    (output / "informe_robustez.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robustez YOLO CCT20: Fases 5 y 6 + estudio de oclusión."
    )
    parser.add_argument("--project", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--csv-occlusion", type=Path, default=CSV_OCC24_DEFAULT)
    parser.add_argument("--size-metrics", type=Path, default=SIZE_METRICS_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-inference", action="store_true",
                        help="Recalcula predicciones solo para las 24 imágenes.")
    args = parser.parse_args()

    project = args.project.expanduser()
    dataset = args.dataset.expanduser()
    model_path = args.model.expanduser()
    csv_occ = args.csv_occlusion.expanduser()
    results_root = args.results.expanduser()
    size_source = args.size_metrics.expanduser()

    if not project.is_dir():
        raise FileNotFoundError(f"No existe el proyecto: {project}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"No existe el dataset final: {dataset}")
    if not (dataset / "data.yaml").is_file():
        raise FileNotFoundError(f"No existe {dataset / 'data.yaml'}")
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold debe estar en (0,1].")
    if not 0 <= args.conf < 1:
        raise ValueError("--conf debe estar en [0,1).")

    output = results_root / "06_robustez" / f"ejecucion_{stamp()}"
    output.mkdir(parents=True, exist_ok=False)
    print("=" * 76)
    print("YOLO_CCT20 | FASES 5 Y 6 — ROBUSTEZ Y OCLUSIÓN")
    print("=" * 76)
    print("Proyecto:", project)
    print("Dataset final:", dataset)
    print("Modelo:", model_path)
    print("CSV manual (solo lectura):", csv_occ)
    print("Salida:", output)

    audit_rows = audit_dataset(dataset, output)
    print("\nAuditoría de particiones:")
    for row in audit_rows:
        print(row)
    if any(r["status"] != "OK" for r in audit_rows):
        raise RuntimeError(
            "La auditoría de cis_test/trans_test detectó inconsistencias. "
            "Revise auditoria_particiones.csv."
        )

    # Robustez por tamaño: solo copia los CSV preexistentes; no recalcula.
    size_status = reuse_size_metrics(size_source, output)
    print("\nDisponibilidad métricas por tamaño:")
    for row in size_status:
        print(row)
    make_size_plot(output)

    # Lee y valida estrictamente la muestra manual de 24 objetos.
    occ = load_manual_annotations(csv_occ)
    print("\nDistribución original de oclusión:")
    print(occ["nivel_oclusion"].value_counts(dropna=False).to_string())
    if len(occ) != 24:
        raise RuntimeError("Protección: la muestra manual dejó de tener 24 filas.")

    occ = resolve_all_images(occ, dataset)
    print("\nOrigen de imágenes:")
    print(occ["origen_imagen"].value_counts().to_string())
    occ.to_csv(output / "anotaciones_24_enlazadas.csv", index=False, encoding="utf-8-sig")

    cache_path = output / "predicciones_24.json"
    predictions = load_or_run_predictions(
        occ, model_path, cache_path, args.force_inference,
        args.imgsz, args.conf, args.nms_iou, args.device
    )
    detail = evaluate_manual_objects(occ, predictions, args.iou_threshold)
    detail.to_csv(output / "detalle_evaluacion_24.csv", index=False, encoding="utf-8-sig")

    by_split, global_level, distribution = summarize_occlusion(detail)
    by_split.to_csv(output / "resumen_oclusion_por_conjunto.csv", index=False, encoding="utf-8-sig")
    global_level.to_csv(output / "resumen_oclusion_global.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(output / "distribucion_evaluable_oclusion.csv", index=False, encoding="utf-8-sig")
    make_occlusion_plot(global_level, output)

    global_source = optional_global_metrics(results_root, output)
    write_report(output, detail, by_split, global_level, size_status,
                 global_source, csv_occ, model_path)

    (output / "metadatos.json").write_text(json.dumps({
        "project": str(project),
        "dataset_final_fase5": str(dataset),
        "model": str(model_path),
        "manual_occlusion_csv_read_only": str(csv_occ),
        "manual_object_count": len(occ),
        "occlusion_counts": occ["nivel_oclusion"].value_counts().to_dict(),
        "splits": list(SPLITS),
        "prediction_images": len(predictions),
        "confidence_threshold": args.conf,
        "nms_iou": args.nms_iou,
        "matching_iou_threshold": args.iou_threshold,
        "imgsz": args.imgsz,
        "device": args.device,
        "size_metrics_source": str(size_source),
        "global_metrics_source": global_source,
        "generated_at": datetime.now().astimezone().isoformat()
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResumen global de oclusión:")
    print(global_level.to_string(index=False) if not global_level.empty else "Sin filas evaluables.")
    print(f"\nObjetos manuales: {len(detail)}")
    print(f"Detectados: {int(detail['detectado'].sum())}")
    print(f"No detectados: {int(detail['falso_negativo'].sum())}")
    print("\nResultados:", output)
    print("No se entrenó el modelo ni se modificó el dataset, best.pt o CSV manual.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
