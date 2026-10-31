#!/usr/bin/env python3
"""
02_train_yolo.py
Entrenamiento reproducible de YOLOv8s para CCT20.

Por seguridad, el entrenamiento NO se inicia por defecto.
Primero ejecutar:
    python 02_train_yolo.py --check
Para autorizar explícitamente el entrenamiento:
    python 02_train_yolo.py --train

Las particiones de test no se utilizan durante el entrenamiento.
La validación se realiza con VAL_SPLIT (por defecto: cis_val).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get("YOLO_PROJECT", "."))
DATASET = Path(os.environ.get("YOLO_DATASET", str(PROJECT / "data/dataset_yolo")))
SOURCE_DATA_YAML = DATASET / "data.yaml"
WEIGHTS = Path(os.environ.get("YOLO_INITIAL_WEIGHTS", str(PROJECT / "weights/yolov8s.pt")))

OUTPUT_DIR = Path(os.environ.get("YOLO_TRAIN_RESULTS", str(PROJECT / "results/entrenamiento")))
LOG_DIR = OUTPUT_DIR / "logs"
RUN_NAME = "yolov8s_50ep_fase4"
RUN_DIR = OUTPUT_DIR / RUN_NAME

VAL_SPLIT = os.environ.get("YOLO_VAL_SPLIT", "cis_val")
SPLITS = ("train", "cis_val", "trans_val", "cis_test", "trans_test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_NAMES = [
    "badger", "bird", "bobcat", "cat", "coyote", "deer", "dog",
    "fox", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel",
]

# Parámetros conservados de la Fase 4 original.
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "50"))
IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
BATCH = int(os.environ.get("YOLO_BATCH", "8"))
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")
WORKERS = int(os.environ.get("YOLO_WORKERS", "2"))
SEED = int(os.environ.get("YOLO_SEED", "0"))


def audit_split(split: str) -> dict:
    """Comprueba correspondencia entre imágenes y TXT de etiquetas."""
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    if not image_dir.is_dir() or not label_dir.is_dir():
        return {
            "split": split,
            "ok": False,
            "error": "Falta el directorio images/ o labels/",
        }

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(label_dir.rglob("*.txt"))

    image_keys = {
        p.relative_to(image_dir).with_suffix("").as_posix()
        for p in images
    }
    label_keys = {
        p.relative_to(label_dir).with_suffix("").as_posix()
        for p in labels
    }

    missing = sorted(image_keys - label_keys)
    orphan = sorted(label_keys - image_keys)
    empty = sum(p.stat().st_size == 0 for p in labels)

    return {
        "split": split,
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty,
        "images_without_label": len(missing),
        "labels_without_image": len(orphan),
        "missing_examples": missing[:5],
        "orphan_examples": orphan[:5],
        "ok": not missing and not orphan and len(images) == len(labels),
    }


def load_yaml_config():
    """Lee el YAML de Fase 3 y comprueba clases y rutas esperadas."""
    if importlib.util.find_spec("yaml") is None:
        raise RuntimeError("Falta PyYAML. Instálelo en el entorno de entrenamiento.")
    import yaml

    if not SOURCE_DATA_YAML.is_file():
        raise FileNotFoundError(f"No existe el YAML del dataset: {SOURCE_DATA_YAML}")

    cfg = yaml.safe_load(SOURCE_DATA_YAML.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("El data.yaml original no contiene un diccionario válido.")

    names = cfg.get("names")
    if isinstance(names, dict):
        try:
            keys = sorted(int(k) for k in names)
            if keys != list(range(14)):
                raise ValueError("Los IDs de clase deben ser consecutivos de 0 a 13.")
            names = [names[k] if k in names else names[str(k)] for k in range(14)]
        except (TypeError, KeyError):
            raise ValueError("No se pudo interpretar el campo names del data.yaml.")
    elif isinstance(names, list):
        names = list(names)
    else:
        raise ValueError("El campo names debe ser una lista o diccionario.")

    if names != EXPECTED_NAMES:
        raise ValueError(
            "Los nombres/orden de clases del dataset no coinciden con las 14 clases "
            f"esperadas.\nEncontrado: {names}"
        )

    if cfg.get("train") != "images/train":
        raise ValueError("La ruta train del data.yaml no coincide con images/train.")

    if VAL_SPLIT not in ("cis_val", "trans_val"):
        raise ValueError("VAL_SPLIT debe ser cis_val o trans_val.")

    # YAML independiente: no se modifica el data.yaml validado de Fase 3.
    phase_cfg = {
        "path": str(DATASET),
        "train": "images/train",
        "val": f"images/{VAL_SPLIT}",
        "test": "images/cis_test",
        "nc": len(EXPECTED_NAMES),
        "names": EXPECTED_NAMES,
    }
    return phase_cfg


def write_phase4_yaml(phase_cfg: dict) -> Path:
    import yaml

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = OUTPUT_DIR / f"data_fase4_val_{VAL_SPLIT}.yaml"

    if yaml_path.exists():
        existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if existing != phase_cfg:
            raise FileExistsError(
                f"El YAML ya existe con configuración diferente: {yaml_path}. "
                "No se sobrescribirá."
            )
    else:
        yaml_path.write_text(
            yaml.safe_dump(phase_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    return yaml_path


def check_environment() -> dict:
    modules = ("torch", "ultralytics", "yaml", "PIL", "numpy")
    availability = {
        name: importlib.util.find_spec(name) is not None for name in modules
    }
    missing = [name for name, present in availability.items() if not present]
    if missing:
        raise RuntimeError("Dependencias no disponibles: " + ", ".join(missing))

    import torch
    import ultralytics

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def preflight() -> tuple[list[dict], dict, Path]:
    errors = []

    for label, path in {
        "Proyecto": PROJECT,
        "Dataset": DATASET,
        "Pesos YOLOv8s": WEIGHTS,
    }.items():
        if not path.exists():
            errors.append(f"No existe {label}: {path}")

    if not DATASET.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {DATASET}")

    results = [audit_split(split) for split in SPLITS]
    for row in results:
        if not row.get("ok", False):
            errors.append(f"Auditoría no superada en {row['split']}: {row}")

    phase_cfg = load_yaml_config()
    yaml_path = write_phase4_yaml(phase_cfg)

    if RUN_DIR.exists():
        errors.append(
            f"El directorio de entrenamiento ya existe; se protege y no se sobrescribe: {RUN_DIR}"
        )

    env = check_environment()
    return results, {"errors": errors, "environment": env}, yaml_path


def print_preflight(results: list[dict], report: dict, yaml_path: Path) -> None:
    print("=" * 72)
    print("YOLO CCT20 — FASE 4: COMPROBACIÓN PREVIA")
    print("=" * 72)
    print(f"Proyecto:       {PROJECT}")
    print(f"Dataset:        {DATASET}")
    print(f"Pesos iniciales:{WEIGHTS}")
    print(f"YAML Fase 4:    {yaml_path}")
    print(f"Validación:     {VAL_SPLIT}")
    print(f"Parámetros:     epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, device={DEVICE}")
    print("\nAUDITORÍA DE PARTICIONES")
    for row in results:
        print(json.dumps(row, ensure_ascii=False))
    print("\nENTORNO")
    print(json.dumps(report["environment"], ensure_ascii=False, indent=2))
    if report["errors"]:
        print("\nPREFLIGHT NO SUPERADO:")
        for error in report["errors"]:
            print(" -", error)
    else:
        print("\nPREFLIGHT SUPERADO. No se ha iniciado entrenamiento.")


def train_model(yaml_path: Path) -> None:
    from ultralytics import YOLO

    if RUN_DIR.exists():
        raise FileExistsError(
            f"Se cancela para proteger resultados existentes: {RUN_DIR}"
        )

    start = time.time()
    start_iso = datetime.now().astimezone().isoformat()
    print(f"Inicio: {start_iso}")
    print(f"Salida: {RUN_DIR}")

    try:
        model = YOLO(str(WEIGHTS))
        model.train(
            data=str(yaml_path),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            fraction=1.0,
            val=True,
            patience=50,
            pretrained=True,
            optimizer="auto",
            cache=False,
            seed=SEED,
            deterministic=True,
            save=True,
            save_period=10,
            plots=True,
            project=str(OUTPUT_DIR),
            name=RUN_NAME,
            exist_ok=False,
            verbose=True,
        )
        print("Entrenamiento finalizado.")
    finally:
        elapsed = time.time() - start
        print(f"Fin: {datetime.now().astimezone().isoformat()}")
        print(f"Duración: {elapsed / 3600:.2f} horas")


def inspect_artifacts() -> None:
    print("\nARTEFACTOS DEL ENTRENAMIENTO")
    expected = (
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "results.png",
        "confusion_matrix.png",
    )
    for relative in expected:
        path = RUN_DIR / relative
        print(f"{'PRESENTE' if path.is_file() else 'NO DISPONIBLE':16s} {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preparación y entrenamiento YOLOv8s para CCT20."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Ejecuta auditorías y preflight; no entrena (modo predeterminado).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Autoriza explícitamente el entrenamiento completo.",
    )
    args = parser.parse_args()

    if args.check and args.train:
        parser.error("Use --check o --train, no ambos.")

    try:
        results, report, yaml_path = preflight()
        print_preflight(results, report, yaml_path)

        if report["errors"]:
            return 1

        if args.train:
            train_model(yaml_path)
            inspect_artifacts()
        else:
            print("\nEntrenamiento NO iniciado. Para autorizarlo use --train.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
