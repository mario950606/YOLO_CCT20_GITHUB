#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO_CCT20 - Fase 4 + Fase 5: evaluación del modelo.

Modo seguro por defecto:
  * audita la estructura del dataset;
  * reutiliza metricas_globales_cis_trans.csv si existe;
  * NO ejecuta model.val, NO entrena y NO modifica pesos.

Para ejecutar una nueva evaluación completa de cis_test y trans_test:
  python src/03_evaluate_model.py --evaluate

Las nuevas evaluaciones se guardan en un directorio nuevo con timestamp.
No se sobrescriben resultados anteriores.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROJECT = Path(".")
DEFAULT_DATASET = DEFAULT_PROJECT / "data/dataset_yolo"
DEFAULT_MODEL = Path("weights/best.pt")
DEFAULT_RESULTS = DEFAULT_PROJECT / "results"
SPLITS = ("train", "cis_val", "trans_val", "cis_test", "trans_test")
TEST_SPLITS = ("cis_test", "trans_test")
CLASS_NAMES = [
    "badger", "bird", "bobcat", "cat", "coyote", "deer", "dog",
    "fox", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_names(cfg: dict[str, Any]) -> list[str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        try:
            names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
        except (KeyError, TypeError):
            raise ValueError("El campo 'names' del data.yaml no tiene IDs consecutivos.")
    if not isinstance(names, list):
        raise ValueError("El data.yaml debe contener 'names' como lista o diccionario.")
    return [str(x) for x in names]


def audit_dataset(dataset: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Auditoría de solo lectura; las etiquetas vacías son válidas."""
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

        images = [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        labels = list(label_dir.rglob("*.txt"))
        image_keys = {p.relative_to(image_dir).with_suffix("").as_posix() for p in images}
        label_keys = {p.relative_to(label_dir).with_suffix("").as_posix() for p in labels}
        missing = image_keys - label_keys
        orphan = label_keys - image_keys
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
                parts = line.split()
                if len(parts) != 5:
                    malformed += 1
                    continue
                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    if cls < 0 or cls >= len(CLASS_NAMES):
                        invalid_class += 1
                    if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        invalid_coord += 1
                except (TypeError, ValueError):
                    malformed += 1

        ok = not (missing or orphan or malformed or invalid_class or invalid_coord)
        rows.append({
            "split": split, "images": len(images), "labels": len(labels),
            "empty_labels": empty, "images_without_label": len(missing),
            "labels_without_image": len(orphan), "malformed_lines": malformed,
            "invalid_classes": invalid_class, "invalid_coordinates": invalid_coord,
            "status": "OK" if ok else "REVIEW"
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "auditoria_estructura_dataset.csv"
    write_csv(csv_path, rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_existing_metrics(results_dir: Path, project: Path) -> Path | None:
    candidates = [
        results_dir / "02_evaluacion_modelo" / "metricas_globales_cis_trans.csv",
        results_dir / "metricas_globales_cis_trans.csv",
        project / "resultados_fase5" / "02_evaluacion_modelo" / "metricas_globales_cis_trans.csv",
        project / "resultados_fase5" / "metricas_globales_cis_trans.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_existing_metrics(path: Path) -> list[dict[str, Any]]:
    import pandas as pd
    df = pd.read_csv(path)
    required = {"split", "precision", "recall", "mAP50", "mAP50_95"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV existente no contiene columnas requeridas: {sorted(missing)}")
    df = df[df["split"].isin(TEST_SPLITS)]
    if set(df["split"]) != set(TEST_SPLITS):
        raise ValueError("El CSV existente no contiene cis_test y trans_test.")
    return df.to_dict(orient="records")


def make_split_yaml(dataset: Path, base_cfg: dict[str, Any], split: str, out_dir: Path) -> Path:
    cfg = dict(base_cfg)
    cfg["path"] = str(dataset)
    cfg["val"] = f"images/{split}"
    cfg.pop("test", None)
    path = out_dir / f"data_{split}.yaml"
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )
    return path


def evaluate_model(model_path: Path, dataset: Path, base_cfg: dict[str, Any],
                   out_dir: Path, imgsz: int, batch: int, device: str,
                   workers: int, conf: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ejecuta model.val únicamente cuando el usuario especifica --evaluate."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    global_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for split in TEST_SPLITS:
        print(f"\nEvaluando {split} ...")
        split_yaml = make_split_yaml(dataset, base_cfg, split, out_dir)
        kwargs = dict(
            data=str(split_yaml), split="val", imgsz=imgsz, batch=batch,
            device=device, workers=workers, plots=True,
            project=str(out_dir / "ultralytics"), name=split,
            exist_ok=False, verbose=True
        )
        if conf is not None:
            kwargs["conf"] = conf
        metrics = model.val(**kwargs)
        box = metrics.box
        global_rows.append({
            "split": split,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        })

        # Ultralytics expone AP por clase en box.ap50 y box.ap.
        ap50_values = getattr(box, "ap50", None)
        ap_values = getattr(box, "ap", None)
        maps = getattr(box, "maps", None)
        names = model.names
        class_ids = list(range(len(names)))
        for idx, cid in enumerate(class_ids):
            ap50 = float(ap50_values[idx]) if ap50_values is not None and idx < len(ap50_values) else None
            ap = float(ap_values[idx]) if ap_values is not None and idx < len(ap_values) else (
                float(maps[cid]) if maps is not None and cid < len(maps) else None
            )
            class_rows.append({
                "split": split, "class_id": cid, "class_name": names[cid],
                "AP50": ap50, "AP50_95": ap
            })

    return global_rows, class_rows


def save_comparison(global_rows: list[dict[str, Any]], out_dir: Path) -> None:
    write_csv(out_dir / "metricas_globales_cis_trans.csv", global_rows)
    by_split = {r["split"]: r for r in global_rows}
    if set(by_split) == set(TEST_SPLITS):
        rows = []
        for metric in ("precision", "recall", "mAP50", "mAP50_95"):
            cis = float(by_split["cis_test"][metric])
            trans = float(by_split["trans_test"][metric])
            rows.append({
                "metrica": metric, "cis_test": cis, "trans_test": trans,
                "brecha_cis_menos_trans": cis - trans,
                "brecha_puntos_porcentuales": (cis - trans) * 100
            })
        write_csv(out_dir / "brecha_generalizacion.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auditoría y evaluación YOLO CCT20 (Fases 4 y 5)."
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--evaluate", action="store_true",
                        help="Ejecuta model.val en cis_test y trans_test. Puede tardar.")
    parser.add_argument("--from-existing", action="store_true",
                        help="Reutiliza métricas CSV existentes (comportamiento predeterminado).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=None,
                        help="Confianza opcional para model.val; por defecto usa Ultralytics.")
    args = parser.parse_args()

    project = args.project.expanduser()
    dataset = args.dataset.expanduser()
    model_path = args.model.expanduser()
    results_root = args.results.expanduser()
    output_dir = results_root / "02_evaluacion_modelo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("YOLO_CCT20 | EVALUACIÓN FASE 4 + FASE 5")
    print("=" * 72)
    print("Proyecto :", project)
    print("Dataset  :", dataset)
    print("Modelo   :", model_path)
    print("Salida   :", output_dir)
    print("Modo     :", "EVALUACIÓN NUEVA" if args.evaluate else "REUTILIZAR RESULTADOS")

    if not project.is_dir():
        raise FileNotFoundError(f"No existe el proyecto: {project}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset}")
    yaml_path = dataset / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No existe: {yaml_path}")

    base_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(base_cfg, dict):
        raise ValueError("data.yaml no contiene un diccionario válido.")
    names = load_names(base_cfg)
    if names != CLASS_NAMES:
        raise ValueError(
            "Las clases u orden del data.yaml no coinciden con las 14 clases "
            f"esperadas por el proyecto.\nEncontradas: {names}"
        )

    audit_rows = audit_dataset(dataset, output_dir / "auditoria")
    print("\nAUDITORÍA DE PARTICIONES")
    for row in audit_rows:
        print(row)
    bad = [r for r in audit_rows if r["split"] in TEST_SPLITS and r["status"] != "OK"]
    if bad:
        raise RuntimeError(
            "La auditoría de cis_test/trans_test detectó inconsistencias. "
            "Revise auditoria/auditoria_estructura_dataset.csv antes de evaluar."
        )

    if args.evaluate:
        if not model_path.is_file():
            raise FileNotFoundError(f"No se encuentra best.pt: {model_path}")
        run_dir = output_dir / f"evaluacion_{timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        global_rows, class_rows = evaluate_model(
            model_path, dataset, base_cfg, run_dir,
            args.imgsz, args.batch, args.device, args.workers, args.conf
        )
        save_comparison(global_rows, run_dir)
        write_csv(run_dir / "metricas_por_clase.csv", class_rows)
        (run_dir / "metadatos_evaluacion.json").write_text(
            json.dumps({
                "model": str(model_path), "dataset": str(dataset),
                "splits": list(TEST_SPLITS), "imgsz": args.imgsz,
                "batch": args.batch, "device": args.device,
                "workers": args.workers, "conf": args.conf,
                "created": datetime.now().astimezone().isoformat()
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print("\nEvaluación nueva guardada en:", run_dir)
    else:
        existing = find_existing_metrics(output_dir, project)
        if existing is None:
            print("\nNo se encontró metricas_globales_cis_trans.csv.")
            print("No se ejecutó model.val para evitar repetir cómputo.")
            print("Cuando decida ejecutar una evaluación nueva, use --evaluate.")
            return 2
        global_rows = read_existing_metrics(existing)
        # Copia de consulta a una carpeta de resultados sin modificar el original.
        reuse_dir = output_dir / f"reutilizados_{timestamp()}"
        reuse_dir.mkdir(parents=True, exist_ok=False)
        save_comparison(global_rows, reuse_dir)
        (reuse_dir / "fuente_metricas.txt").write_text(
            f"Métricas reutilizadas desde: {existing}\n"
            "No se ejecutó inferencia ni model.val.\n",
            encoding="utf-8"
        )
        print("\nMétricas reutilizadas desde:", existing)
        print("Copia de trabajo:", reuse_dir)
        for row in global_rows:
            print(row)

    print("\nProceso terminado. No se entrenó el modelo ni se modificó el dataset.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
