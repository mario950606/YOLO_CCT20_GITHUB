#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_generalization.py
YOLO_CCT20 — Fase 5: análisis de generalización entre cis_test y trans_test.

Este módulo NO entrena el modelo. Por defecto reutiliza los CSV de evaluación
generados por 03_evaluate_model.py o por el notebook integrado de Fase 5/6.
Solo con --evaluate ejecuta model.val en ambos conjuntos.

IMPORTANTE:
- cis_test y trans_test se comparan como particiones definidas por el proyecto.
- La diferencia observada es una brecha entre particiones. No demuestra por sí
  sola que la ubicación geográfica sea la causa.
- Para afirmar "ubicaciones vistas/no vistas" hay que verificar los metadatos
  camera/location_id y demostrar la correspondencia exacta con los splits.
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

PROJECT_DEFAULT = Path(".")
DATASET_DEFAULT = PROJECT_DEFAULT / "data/dataset_yolo"
MODEL_DEFAULT = Path("weights/best.pt")
RESULTS_DEFAULT = PROJECT_DEFAULT / "results"
SPLITS = ("cis_test", "trans_test")
EXPECTED_CLASSES = [
    "badger", "bird", "bobcat", "cat", "coyote", "deer", "dog",
    "fox", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"
]
REQUIRED_GLOBAL = {"split", "precision", "recall", "mAP50", "mAP50_95"}


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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd
    return pd.read_csv(path).to_dict(orient="records")


def candidate_metric_files(results: Path, project: Path) -> tuple[list[Path], list[Path]]:
    global_candidates = [
        results / "02_evaluacion_modelo" / "metricas_globales_cis_trans.csv",
        results / "metricas_globales_cis_trans.csv",
        project / "resultados_fase5" / "02_evaluacion_modelo" / "metricas_globales_cis_trans.csv",
        project / "resultados_fase5" / "metricas_globales_cis_trans.csv",
    ]
    class_candidates = [
        results / "02_evaluacion_modelo" / "metricas_por_clase.csv",
        results / "metricas_por_clase.csv",
        project / "resultados_fase5" / "02_evaluacion_modelo" / "metricas_por_clase.csv",
        project / "resultados_fase5" / "metricas_por_clase.csv",
    ]
    return global_candidates, class_candidates


def first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.is_file()), None)


def load_dataset_config(dataset: Path) -> dict[str, Any]:
    path = dataset / "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"El contenido de {path} no es un diccionario YAML.")
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if not isinstance(names, list):
        raise ValueError("data.yaml debe declarar names como lista o diccionario.")
    names = [str(n) for n in names]
    if names != EXPECTED_CLASSES:
        raise ValueError(
            "El orden/número de clases del data.yaml no coincide con el esperado.\n"
            f"Esperado: {EXPECTED_CLASSES}\nEncontrado: {names}"
        )
    return cfg


def audit_test_splits(dataset: Path, output: Path) -> list[dict[str, Any]]:
    """Comprueba imágenes, etiquetas y geometría; etiquetas vacías son válidas."""
    from PIL import Image

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
                  if p.is_file() and p.suffix.lower() in
                  {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}]
        labels = list(label_dir.rglob("*.txt"))
        image_ids = {p.relative_to(image_dir).with_suffix("").as_posix() for p in images}
        label_ids = {p.relative_to(label_dir).with_suffix("").as_posix() for p in labels}
        missing = image_ids - label_ids
        orphan = label_ids - image_ids
        empty = malformed = invalid_class = invalid_coord = 0

        for label in labels:
            if label.stat().st_size == 0:
                empty += 1
                continue
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    malformed += 1
                    continue
                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    invalid_class += int(cls < 0 or cls >= len(EXPECTED_CLASSES))
                    invalid_coord += int(
                        not all(0.0 <= v <= 1.0 for v in (x, y, w, h))
                        or w <= 0 or h <= 0
                    )
                except (ValueError, TypeError):
                    malformed += 1

        valid = not (missing or orphan or malformed or invalid_class or invalid_coord)
        rows.append({
            "split": split, "images": len(images), "labels": len(labels),
            "empty_labels": empty, "images_without_label": len(missing),
            "labels_without_image": len(orphan), "malformed_lines": malformed,
            "invalid_classes": invalid_class, "invalid_coordinates": invalid_coord,
            "status": "OK" if valid else "REVIEW"
        })

    write_csv(output / "auditoria_particiones.csv", rows)
    return rows


def run_evaluation(model_path: Path, dataset: Path, cfg: dict[str, Any],
                   output: Path, imgsz: int, batch: int, device: str,
                   workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(f"No se encuentra el modelo: {model_path}")
    model = YOLO(str(model_path))
    global_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        split_cfg = dict(cfg)
        split_cfg["path"] = str(dataset)
        split_cfg["val"] = f"images/{split}"
        split_cfg.pop("test", None)
        yaml_path = output / f"data_{split}.yaml"
        yaml_path.write_text(
            yaml.safe_dump(split_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8"
        )

        print(f"\n[model.val] Evaluando {split} ...")
        metrics = model.val(
            data=str(yaml_path), split="val", imgsz=imgsz, batch=batch,
            device=device, workers=workers, plots=True,
            project=str(output / "ultralytics"), name=split,
            exist_ok=False, verbose=True
        )
        box = metrics.box
        global_rows.append({
            "split": split,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        })
        # En Ultralytics, maps contiene AP50:95 por clase. AP50 se obtiene
        # de ap50 cuando la versión instalada lo expone.
        maps_5095 = getattr(box, "maps", None)
        ap50 = getattr(box, "ap50", None)
        for class_id, class_name in enumerate(model.names.values()
                                               if isinstance(model.names, dict)
                                               else model.names):
            class_rows.append({
                "split": split,
                "class_id": class_id,
                "class_name": str(class_name),
                "AP50": float(ap50[class_id]) if ap50 is not None and class_id < len(ap50) else None,
                "mAP50_95": float(maps_5095[class_id])
                if maps_5095 is not None and class_id < len(maps_5095) else None
            })
    return global_rows, class_rows


def validate_global(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("El CSV de métricas globales está vacío.")
    missing = REQUIRED_GLOBAL - set(rows[0])
    if missing:
        raise ValueError(f"Faltan columnas en métricas globales: {sorted(missing)}")
    by_split = {str(r["split"]): r for r in rows}
    if not set(SPLITS).issubset(by_split):
        raise ValueError(f"Se requieren ambos splits: {SPLITS}")
    return [by_split[s] for s in SPLITS]


def calculate_gap(global_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_split = {r["split"]: r for r in global_rows}
    output = []
    for metric in ("precision", "recall", "mAP50", "mAP50_95"):
        cis = float(by_split["cis_test"][metric])
        trans = float(by_split["trans_test"][metric])
        output.append({
            "metrica": metric,
            "cis_test": cis,
            "trans_test": trans,
            "brecha_cis_menos_trans": cis - trans,
            "brecha_puntos_porcentuales": (cis - trans) * 100.0,
            "caida_relativa_trans_vs_cis_pct": (
                (cis - trans) / cis * 100.0 if cis != 0 else None
            )
        })
    return output


def compare_classes(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not class_rows:
        return []
    by_key = {(r["split"], str(r["class_name"])): r for r in class_rows}
    names = sorted({str(r["class_name"]) for r in class_rows})
    output = []
    for name in names:
        a = by_key.get(("cis_test", name), {})
        b = by_key.get(("trans_test", name), {})
        ap_a = a.get("mAP50_95")
        ap_b = b.get("mAP50_95")
        ap50_a = a.get("AP50")
        ap50_b = b.get("AP50")
        output.append({
            "class_name": name,
            "cis_test_mAP50_95": ap_a,
            "trans_test_mAP50_95": ap_b,
            "gap_mAP50_95": (
                float(ap_a) - float(ap_b)
                if ap_a is not None and ap_b is not None else None
            ),
            "cis_test_AP50": ap50_a,
            "trans_test_AP50": ap50_b,
            "gap_AP50": (
                float(ap50_a) - float(ap50_b)
                if ap50_a is not None and ap50_b is not None else None
            )
        })
    return output


def make_plots(global_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]],
               output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(global_rows).set_index("split")
    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    ax = df.loc[list(SPLITS), metrics].plot(kind="bar", figsize=(10, 5), ylim=(0, 1))
    ax.set_title("Comparación de desempeño: cis_test vs trans_test")
    ax.set_ylabel("Métrica (0–1)")
    ax.set_xlabel("Partición")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "comparacion_metricas_globales.png", dpi=180)
    plt.close()

    if class_rows:
        cdf = pd.DataFrame(class_rows)
        if "mAP50_95" in cdf and cdf["mAP50_95"].notna().any():
            pivot = cdf.pivot(index="class_name", columns="split", values="mAP50_95")
            pivot = pivot.reindex(columns=list(SPLITS))
            ax = pivot.plot(kind="bar", figsize=(13, 6), ylim=(0, 1))
            ax.set_title("mAP@0.5:0.95 por especie y partición")
            ax.set_ylabel("mAP@0.5:0.95")
            ax.set_xlabel("Especie")
            ax.grid(axis="y", alpha=0.25)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output / "map5095_por_especie.png", dpi=180)
            plt.close()


def write_report(output: Path, global_rows: list[dict[str, Any]],
                 gap_rows: list[dict[str, Any]], class_gap: list[dict[str, Any]],
                 audit_rows: list[dict[str, Any]], source: str) -> None:
    by_split = {r["split"]: r for r in global_rows}
    lines = [
        "# YOLO_CCT20 — Fase 5: análisis de generalización",
        "",
        f"- Fecha de generación: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Fuente de métricas: `{source}`",
        "- Particiones comparadas: `cis_test` y `trans_test`.",
        "",
        "## 1. Métricas globales",
        "",
        "| Métrica | cis_test | trans_test |",
        "|---|---:|---:|",
    ]
    for metric, label in [
        ("precision", "Precision"), ("recall", "Recall"),
        ("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95")
    ]:
        lines.append(
            f"| {label} | {float(by_split['cis_test'][metric]):.4f} "
            f"| {float(by_split['trans_test'][metric]):.4f} |"
        )
    lines += ["", "## 2. Brecha entre particiones", "",
              "| Métrica | cis − trans | Brecha (puntos porcentuales) | Caída relativa (%) |",
              "|---|---:|---:|---:|"]
    for r in gap_rows:
        rel = "N/D" if r["caida_relativa_trans_vs_cis_pct"] is None else f"{r['caida_relativa_trans_vs_cis_pct']:.2f}"
        lines.append(
            f"| {r['metrica']} | {r['brecha_cis_menos_trans']:.4f} "
            f"| {r['brecha_puntos_porcentuales']:.2f} | {rel} |"
        )
    lines += [
        "", "## 3. Interpretación y límites",
        "",
        "La comparación cuantifica diferencias de desempeño entre las particiones "
        "`cis_test` y `trans_test`. Una brecha positiva indica que la métrica fue "
        "mayor en `cis_test` que en `trans_test`.",
        "",
        "Esta diferencia no demuestra por sí sola que la ubicación geográfica sea "
        "la causa. Para describir formalmente el resultado como generalización a "
        "ubicaciones no vistas, se debe verificar en los metadatos originales la "
        "identidad de las cámaras/ubicaciones de entrenamiento y prueba y confirmar "
        "que la definición de `cis`/`trans` corresponde a ubicaciones conocidas "
        "y no conocidas durante el entrenamiento.",
        "",
        "Las métricas por especie deben interpretarse considerando el número de "
        "objetos/anotaciones de cada clase en cada partición. Una métrica baja o "
        "no disponible en una clase con pocos ejemplos puede ser inestable.",
        "",
        "## 4. Auditoría de particiones",
        "",
        "| Partición | Imágenes | Etiquetas | Etiquetas vacías | Estado |",
        "|---|---:|---:|---:|---|",
    ]
    for r in audit_rows:
        lines.append(
            f"| {r['split']} | {r['images']} | {r['labels']} | "
            f"{r['empty_labels']} | {r['status']} |"
        )
    lines += [
        "", "## 5. Salidas",
        "",
        "- `metricas_globales_cis_trans.csv`",
        "- `brecha_generalizacion.csv`",
        "- `comparacion_por_especie.csv`",
        "- `comparacion_metricas_globales.png`",
        "- `map5095_por_especie.png` (si hay métricas por clase)",
        "- `auditoria_particiones.csv`",
        "",
        "Este módulo no entrena el modelo ni modifica el dataset original."
    ]
    (output / "informe_generalizacion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fase 5: análisis de generalización cis_test vs trans_test."
    )
    parser.add_argument("--project", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--evaluate", action="store_true",
                        help="Ejecuta model.val en ambos splits; requiere tiempo/cómputo.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    project = args.project.expanduser()
    dataset = args.dataset.expanduser()
    results = args.results.expanduser()
    if not project.is_dir():
        raise FileNotFoundError(f"No existe el proyecto: {project}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset}")

    cfg = load_dataset_config(dataset)
    output = results / "04_generalizacion" / (
        f"evaluacion_{stamp()}" if args.evaluate else f"analisis_{stamp()}"
    )
    output.mkdir(parents=True, exist_ok=False)

    print("=" * 72)
    print("YOLO_CCT20 | FASE 5 — GENERALIZACIÓN")
    print("=" * 72)
    print("Dataset:", dataset)
    print("Salida :", output)
    print("Modo   :", "EVALUACIÓN NUEVA" if args.evaluate else "REUTILIZAR MÉTRICAS")

    audit_rows = audit_test_splits(dataset, output)
    print("\nAuditoría:")
    for row in audit_rows:
        print(row)
    if any(r["status"] != "OK" for r in audit_rows):
        raise RuntimeError(
            "La auditoría de cis_test/trans_test detectó problemas. "
            "Revise auditoria_particiones.csv; no se continúa con la comparación."
        )

    if args.evaluate:
        global_rows, class_rows = run_evaluation(
            args.model.expanduser(), dataset, cfg, output,
            args.imgsz, args.batch, args.device, args.workers
        )
        source = f"model.val ejecutado con {args.model}"
    else:
        global_candidates, class_candidates = candidate_metric_files(results, project)
        global_file = first_existing(global_candidates)
        if global_file is None:
            raise FileNotFoundError(
                "No se encontró metricas_globales_cis_trans.csv. "
                "Ejecute primero 03_evaluate_model.py o use --evaluate."
            )
        global_rows = read_csv_rows(global_file)
        class_file = first_existing(class_candidates)
        class_rows = read_csv_rows(class_file) if class_file else []
        source = str(global_file)
        print("\nMétricas globales reutilizadas:", global_file)
        if class_file:
            print("Métricas por clase reutilizadas:", class_file)
        else:
            print("Aviso: no se encontró metricas_por_clase.csv; "
                  "se omite el análisis por especie.")

    global_rows = validate_global(global_rows)
    gap_rows = calculate_gap(global_rows)
    class_gap = compare_classes(class_rows)

    write_csv(output / "metricas_globales_cis_trans.csv", global_rows)
    write_csv(output / "brecha_generalizacion.csv", gap_rows)
    write_csv(output / "comparacion_por_especie.csv", class_gap)
    if class_rows:
        write_csv(output / "metricas_por_clase_fuente.csv", class_rows)

    make_plots(global_rows, class_rows, output)
    write_report(output, global_rows, gap_rows, class_gap, audit_rows, source)

    (output / "metadatos.json").write_text(json.dumps({
        "project": str(project),
        "dataset": str(dataset),
        "model": str(args.model.expanduser()),
        "splits": list(SPLITS),
        "source_metrics": source,
        "evaluation_executed": bool(args.evaluate),
        "imgsz": args.imgsz if args.evaluate else None,
        "batch": args.batch if args.evaluate else None,
        "device": args.device if args.evaluate else None,
        "workers": args.workers if args.evaluate else None,
        "generated_at": datetime.now().astimezone().isoformat()
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBrecha entre particiones:")
    for row in gap_rows:
        print(row)
    print("\nResultados guardados en:", output)
    print("No se entrenó el modelo ni se modificó el dataset.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
