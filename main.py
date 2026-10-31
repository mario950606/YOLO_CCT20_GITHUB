#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Punto de entrada seguro para el proyecto YOLO_CCT20."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
PHASE_SCRIPTS = {
    "1": ROOT / "src" / "01_prepare_dataset.py",
    "2": ROOT / "src" / "02_train_yolo.py",
    "3": ROOT / "src" / "03_evaluate_model.py",
    "4": ROOT / "src" / "04_generalization.py",
    "5": ROOT / "src" / "05_gradcam.py",
    "6": ROOT / "src" / "06_robustness.py",
    "7": ROOT / "src" / "07_ecological_monitoring.py",
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Copie configs/config.example.yaml a "
            "configs/config.yaml y ajuste las rutas locales."
        )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("La configuración debe contener una sección 'paths'.")
    return config


def resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def run(command: list[str | Path], env: dict[str, str] | None = None) -> None:
    printable = [str(item) for item in command]
    print("\n>>> " + " ".join(printable), flush=True)
    subprocess.run(printable, cwd=ROOT, env=env, check=True)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"No se encuentra {description}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecutor por fases de YOLO_CCT20. No entrena ni recalcula evaluaciones por defecto."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "config.yaml",
        help="Configuración local YAML (por defecto: configs/config.yaml)."
    )
    parser.add_argument(
        "--fase", required=True, choices=list(PHASE_SCRIPTS),
        help="Fase que se ejecutará: 1 auditoría, 2 entrenamiento/check, 3 evaluación, "
             "4 generalización, 5 Grad-CAM, 6 robustez/oclusión, 7 monitoreo."
    )
    parser.add_argument("--train", action="store_true",
                        help="Autoriza explícitamente un entrenamiento nuevo en fase 2.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Autoriza ejecutar model.val en fases 3 o 4; sin esto se reutilizan resultados.")
    parser.add_argument("--force-inference", action="store_true",
                        help="Fuerza únicamente la inferencia de las 24 imágenes de fase 6.")
    parser.add_argument("--target-location", default=None,
                        help="ID de ubicación no vista, requerido por fase 7.")
    args = parser.parse_args()

    config_path = resolve(args.config)
    cfg = load_config(config_path)
    paths = cfg["paths"]
    training = cfg.get("training", {})
    evaluation = cfg.get("evaluation", {})
    monitoring = cfg.get("monitoring", {})

    required_keys = ("project", "dataset", "model", "results")
    missing_keys = [key for key in required_keys if not paths.get(key)]
    if missing_keys:
        raise ValueError(f"Faltan claves en paths: {missing_keys}")

    project = resolve(paths["project"])
    dataset = resolve(paths["dataset"])
    model = resolve(paths["model"])
    results = resolve(paths["results"])
    cct_root = resolve(paths["cct20_root"]) if paths.get("cct20_root") else None
    initial_weights = resolve(paths.get("initial_weights", "weights/yolov8s.pt"))
    csv_occ = resolve(paths["manual_occlusion_csv"]) if paths.get("manual_occlusion_csv") else None
    size_metrics = resolve(paths["size_metrics"]) if paths.get("size_metrics") else None
    json_dir = resolve(paths["annotations_json"]) if paths.get("annotations_json") else None

    print("=" * 72)
    print(f"YOLO_CCT20 | FASE {args.fase}")
    print("=" * 72)
    print("Proyecto :", project)
    print("Dataset  :", dataset)
    print("Modelo   :", model)
    print("Resultados:", results)

    script = PHASE_SCRIPTS[args.fase]
    require_file(script, f"script de fase {args.fase}")

    # Variables leídas por 02_train_yolo.py; no se inicia entrenamiento sin --train.
    phase2_env = os.environ.copy()
    phase2_env.update({
        "YOLO_PROJECT": str(project),
        "YOLO_DATASET": str(dataset),
        "YOLO_INITIAL_WEIGHTS": str(initial_weights),
        "YOLO_TRAIN_RESULTS": str(results / "entrenamiento"),
        "YOLO_VAL_SPLIT": str(training.get("val_split", "cis_val")),
        "YOLO_EPOCHS": str(training.get("epochs", 50)),
        "YOLO_IMGSZ": str(training.get("imgsz", 640)),
        "YOLO_BATCH": str(training.get("batch", 8)),
        "YOLO_DEVICE": str(training.get("device", "cpu")),
        "YOLO_WORKERS": str(training.get("workers", 2)),
        "YOLO_SEED": str(training.get("seed", 0)),
    })

    if args.fase == "1":
        if cct_root is None:
            raise ValueError("Fase 1 requiere paths.cct20_root.")
        require_dir(cct_root, "directorio fuente CCT20")
        # Solo auditoría. Nunca reconstruye ni sobrescribe el dataset preparado.
        run([
            PYTHON, script, "--dataset-dir", cct_root,
            "--project-dir", project, "--output-dir", dataset, "--mode", "audit"
        ])

    elif args.fase == "2":
        require_dir(dataset, "dataset YOLO")
        # La comprobación no entrena. --train es una autorización explícita.
        command = [PYTHON, script, "--train"] if args.train else [PYTHON, script, "--check"]
        run(command, env=phase2_env)

    elif args.fase in ("3", "4"):
        require_dir(dataset, "dataset YOLO")
        if args.evaluate:
            require_file(model, "modelo best.pt")
        section = evaluation
        command = [
            PYTHON, script, "--project", project, "--dataset", dataset,
            "--model", model, "--results", results,
            "--imgsz", str(section.get("imgsz", 640)),
            "--batch", str(section.get("batch", 8)),
            "--device", str(section.get("device", "cpu")),
            "--workers", str(section.get("workers", 2)),
        ]
        if args.evaluate:
            command.append("--evaluate")
        run(command)

    elif args.fase == "5":
        require_dir(dataset, "dataset YOLO")
        require_file(model, "modelo best.pt")
        run([
            PYTHON, script, "--project", project, "--dataset", dataset,
            "--model", model, "--output", results / "04_gradcam",
            "--imgsz", str(evaluation.get("imgsz", 640)),
            "--device", str(evaluation.get("device", "cpu")),
        ])

    elif args.fase == "6":
        require_dir(dataset, "dataset YOLO")
        require_file(model, "modelo best.pt")
        if csv_occ is None or size_metrics is None:
            raise ValueError("Fase 6 requiere paths.manual_occlusion_csv y paths.size_metrics.")
        require_file(csv_occ, "CSV manual de las 24 anotaciones")
        require_dir(size_metrics, "directorio de métricas por tamaño")
        command = [
            PYTHON, script, "--project", project, "--dataset", dataset,
            "--model", model, "--csv-occlusion", csv_occ,
            "--size-metrics", size_metrics, "--results", results,
            "--imgsz", str(evaluation.get("imgsz", 640)),
            "--device", str(evaluation.get("device", "cpu")),
        ]
        if args.force_inference:
            command.append("--force-inference")
        run(command)

    elif args.fase == "7":
        require_dir(dataset, "dataset YOLO")
        require_file(model, "modelo best.pt")
        if json_dir is None:
            raise ValueError("Fase 7 requiere paths.annotations_json.")
        require_dir(json_dir, "directorio de anotaciones JSON")
        target = args.target_location or monitoring.get("target_location")
        if not target:
            raise ValueError(
                "Fase 7 requiere --target-location con un ID de ubicación "
                "demostrada como no vista durante el entrenamiento."
            )
        command = [
            PYTHON, script, "--json-dir", json_dir, "--dataset", dataset,
            "--model", model, "--results", results,
            "--split", str(monitoring.get("split", "trans_test")),
            "--target-location", str(target),
            "--max-images", str(monitoring.get("max_images", 500)),
            "--imgsz", str(monitoring.get("image_size", 640)),
            "--conf", str(monitoring.get("confidence", 0.25)),
            "--iou", str(monitoring.get("iou", 0.70)),
            "--device", str(monitoring.get("device", "cpu")),
        ]
        if monitoring.get("location_field"):
            command += ["--location-field", str(monitoring["location_field"])]
        if monitoring.get("datetime_field"):
            command += ["--datetime-field", str(monitoring["datetime_field"])]
        run(command)

    print("\nFase solicitada finalizada.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
