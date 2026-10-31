#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_dataset.py
Consolida auditoría de anotaciones (Fase 2) y conversión/construcción YOLO (Fase 3).

Modos:
  --mode audit      auditoría de JSON, categorías, bbox y archivos de imagen (solo lectura)
  --mode build      construye dataset YOLO si la salida aún no existe; si existe, audita
  --mode all        ejecuta auditoría y luego construcción/auditoría final

Los JSON e imágenes originales nunca se modifican. La salida existente nunca se sobrescribe.
"""
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import argparse
import csv
import json
import math
import shutil
import yaml

PARTICIONES = {
    "train": "train_annotations.json",
    "cis_val": "cis_val_annotations.json",
    "trans_val": "trans_val_annotations.json",
    "cis_test": "cis_test_annotations.json",
    "trans_test": "trans_test_annotations.json",
}
CLASES_EXCLUIDAS = {"empty", "car"}

def audit_source(dataset_dir: Path, json_dir: Path, image_dir: Path):
    """Auditoría de Fase 2: estructura JSON, IDs, categorías, bbox e imágenes."""
    print("\n" + "=" * 72)
    print("FASE 2 — AUDITORÍA DE DATOS Y ANOTACIONES")
    print("=" * 72)
    errores = []
    resumen = []
    for split, json_name in PARTICIONES.items():
        path = json_dir / json_name
        if not path.is_file():
            errores.append(f"{split}: falta {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"images", "annotations", "categories"}
        if not required.issubset(data):
            errores.append(f"{split}: JSON sin claves {required - set(data)}")
            continue
        image_ids = [str(x.get("id")) for x in data["images"]]
        duplicate_image_ids = len(image_ids) - len(set(image_ids))
        category_ids = {c.get("id") for c in data["categories"]}
        image_id_set = set(image_ids)
        image_by_id = {str(im.get("id")): im for im in data["images"]}
        missing_image_refs = sum(1 for a in data["annotations"] if str(a.get("image_id")) not in image_id_set)
        bad_bbox = 0
        outside_bbox = 0
        for ann in data["annotations"]:
            bbox = ann.get("bbox")
            if ann.get("category_id") not in category_ids:
                bad_bbox += 1
                continue
            if ann.get("category_id") not in category_ids or bbox is None:
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bad_bbox += 1
                continue
            try:
                x, y, w, h = map(float, bbox)
                if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    bad_bbox += 1
                    continue
                im = image_by_id.get(str(ann.get("image_id")))
                if im:
                    Wj, Hj = float(im["width"]), float(im["height"])
                    if x < 0 or y < 0 or x + w > Wj or y + h > Hj:
                        outside_bbox += 1
            except (TypeError, ValueError, KeyError):
                bad_bbox += 1
        missing_files = sum(1 for im in data["images"] if not (image_dir / im["file_name"]).is_file())
        row = {
            "split": split, "images_json": len(data["images"]),
            "annotations": len(data["annotations"]), "categories": len(data["categories"]),
            "duplicate_image_ids": duplicate_image_ids,
            "annotations_without_image": missing_image_refs,
            "invalid_bbox": bad_bbox, "bbox_outside_declared_bounds": outside_bbox,
            "physical_images_missing": missing_files,
        }
        resumen.append(row)
        print(f"\\n[{split}]")
        for key, value in row.items():
            if key != "split":
                print(f"{key}: {value}")
    report_dir = dataset_dir.parent / "auditoria_fase2"
    report_dir.mkdir(parents=True, exist_ok=True)
    if resumen:
        with (report_dir / "resumen_auditoria_fase2.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(resumen[0]))
            writer.writeheader()
            writer.writerows(resumen)
    if errores:
        raise RuntimeError("Errores en auditoría de origen:\\n" + "\\n".join(errores))
    print(f"\\nResumen de auditoría guardado en: {report_dir}")
    print("Nota: las cajas fuera de límites se reportan; no se alteran los JSON originales.")

def build_or_audit(dataset_dir: Path, json_dir: Path, image_dir: Path, project_dir: Path, out_dir: Path):
    DATASET = dataset_dir
    JSON_DIR = json_dir
    IMG_DIR = image_dir
    PROJECT = project_dir
    OUT_DIR = out_dir
    # Si la salida ya existe, NO reconstruir ni sobrescribir.
    MODO_AUDITORIA = OUT_DIR.exists() and any(OUT_DIR.iterdir())

    if MODO_AUDITORIA:
        print("Se encontró un dataset existente.")
        print("Se realizará únicamente la auditoría.")
        print("No se modificarán imágenes ni etiquetas.")
    else:
        print("No existe el dataset de salida.")
        print("Se realizará la construcción completa.")

    # Comprobar entradas.
    if not MODO_AUDITORIA:
        if not JSON_DIR.is_dir():
            raise FileNotFoundError(f"No existe: {JSON_DIR}")

        if not IMG_DIR.is_dir():
            raise FileNotFoundError(f"No existe: {IMG_DIR}")


    # ============================================================
    # 2. DEFINIR LAS 14 CLASES DE FAUNA
    # ============================================================

    if not MODO_AUDITORIA:

        with (JSON_DIR / PARTICIONES["train"]).open(
            "r", encoding="utf-8"
        ) as f:
            train_data = json.load(f)

        nombres_categoria = {
            c["id"]: c["name"]
            for c in train_data["categories"]
        }

        CLASES = sorted(
            nombre
            for nombre in nombres_categoria.values()
            if nombre not in CLASES_EXCLUIDAS
        )

        if len(CLASES) != 14:
            raise ValueError(
                f"Se esperaban 14 clases; se encontraron "
                f"{len(CLASES)}: {CLASES}"
            )

        ID_CLASE = {
            nombre: i for i, nombre in enumerate(CLASES)
        }

        print("\nClases identificadas:", len(CLASES))

        for nombre, indice in ID_CLASE.items():
            print(indice, nombre)

    else:

        ruta_classes = OUT_DIR / "classes.json"

        if not ruta_classes.is_file():
            raise FileNotFoundError(
                f"No existe el archivo: {ruta_classes}"
            )

        with ruta_classes.open("r", encoding="utf-8") as f:
            clases_guardadas = json.load(f)

        CLASES = [
            clases_guardadas[str(i)]
            for i in range(len(clases_guardadas))
        ]

        if len(CLASES) != 14:
            raise ValueError(
                f"El dataset contiene {len(CLASES)} clases, "
                "se esperaban 14."
            )

        ID_CLASE = {
            nombre: i for i, nombre in enumerate(CLASES)
        }


    # ============================================================
    # 3. CONSTRUCCIÓN DEL DATASET
    #    Solo se ejecuta si la salida no existe.
    # ============================================================

    if not MODO_AUDITORIA:

        for particion in PARTICIONES:

            (OUT_DIR / "images" / particion).mkdir(
                parents=True, exist_ok=True
            )

            (OUT_DIR / "labels" / particion).mkdir(
                parents=True, exist_ok=True
            )

        manifiesto = []
        registro_cajas = []
        resumen = {}

        # --------------------------------------------------------
        # Procesar cada partición
        # --------------------------------------------------------

        for particion, nombre_json in PARTICIONES.items():

            ruta_json = JSON_DIR / nombre_json

            with ruta_json.open("r", encoding="utf-8") as f:
                data = json.load(f)

            categorias = {
                c["id"]: c["name"]
                for c in data["categories"]
            }

            imagenes = {
                str(im["id"]): im
                for im in data["images"]
            }

            anotaciones_por_imagen = {}

            for ann in data["annotations"]:
                iid = str(ann["image_id"])
                anotaciones_por_imagen.setdefault(
                    iid, []
                ).append(ann)

            estad = Counter()
            conteo_clases = Counter()

            # ----------------------------------------------------
            # Recorrer imágenes
            # ----------------------------------------------------

            for iid, im in imagenes.items():

                estad["imagenes_procesadas"] += 1

                nombre_imagen = im["file_name"]
                ruta_relativa = Path(nombre_imagen)

                origen = IMG_DIR / ruta_relativa

                destino_img = (
                    OUT_DIR / "images" / particion / ruta_relativa
                )

                destino_txt = (
                    OUT_DIR / "labels" / particion /
                    ruta_relativa.with_suffix(".txt")
                )

                # Verificar existencia.
                if not origen.is_file():

                    manifiesto.append({
                        "particion": particion,
                        "imagen": nombre_imagen,
                        "estado": "excluida",
                        "motivo": "imagen_fisica_no_encontrada",
                        "cajas_validas": 0,
                    })

                    estad["imagenes_excluidas"] += 1
                    estad["imagen_no_encontrada"] += 1

                    continue

                # Leer dimensiones físicas.
                try:
                    with Image.open(origen) as img:
                        W, H = img.size

                    Wj = float(im["width"])
                    Hj = float(im["height"])

                    if (
                        not all(
                            math.isfinite(v)
                            for v in (W, H, Wj, Hj)
                        )
                        or W <= 0 or H <= 0
                        or Wj <= 0 or Hj <= 0
                    ):
                        raise ValueError("Dimensiones inválidas")

                    sx = W / Wj
                    sy = H / Hj

                except Exception:

                    manifiesto.append({
                        "particion": particion,
                        "imagen": nombre_imagen,
                        "estado": "excluida",
                        "motivo": "error_lectura_o_dimensiones",
                        "cajas_validas": 0,
                    })

                    estad["imagenes_excluidas"] += 1
                    estad["error_lectura"] += 1

                    continue

                etiquetas = []
                motivos = set()

                cajas_corregidas = 0
                cajas_invalidas = 0
                anotacion_incierta = False

                anotaciones = anotaciones_por_imagen.get(
                    iid, []
                )

                # ------------------------------------------------
                # Procesar anotaciones de la imagen
                # ------------------------------------------------

                for ann in anotaciones:

                    categoria = categorias.get(
                        ann.get("category_id")
                    )

                    # Excluir empty y car como clases.
                    # La imagen se conservará como negativa
                    # si no tiene otras cajas de fauna.
                    if categoria in CLASES_EXCLUIDAS:

                        motivos.add(
                            f"anotacion_{categoria}_excluida"
                        )

                        continue

                    # Categoría desconocida.
                    if categoria not in ID_CLASE:

                        motivos.add("categoria_no_reconocida")
                        anotacion_incierta = True

                        continue

                    bbox = ann.get("bbox")

                    # Verificar formato bbox.
                    if (
                        not isinstance(bbox, (list, tuple))
                        or len(bbox) != 4
                    ):

                        motivos.add("anotacion_sin_bbox")
                        anotacion_incierta = True

                        registro_cajas.append({
                            "particion": particion,
                            "imagen": nombre_imagen,
                            "anotacion_id": ann.get("id", ""),
                            "categoria": categoria,
                            "decision": "excluida",
                            "motivo": "anotacion_sin_bbox",
                        })

                        continue

                    # Conversión numérica.
                    try:

                        x, y, w, h = map(float, bbox)

                        if not all(
                            math.isfinite(v)
                            for v in (x, y, w, h)
                        ):
                            raise ValueError(
                                "Coordenadas no finitas"
                            )

                    except Exception:

                        motivos.add("bbox_no_numerica_o_no_finita")
                        anotacion_incierta = True
                        cajas_invalidas += 1

                        registro_cajas.append({
                            "particion": particion,
                            "imagen": nombre_imagen,
                            "anotacion_id": ann.get("id", ""),
                            "categoria": categoria,
                            "decision": "excluida",
                            "motivo": "bbox_no_numerica_o_no_finita",
                        })

                        continue

                    if w <= 0 or h <= 0:

                        motivos.add("caja_no_positiva")
                        anotacion_incierta = True
                        cajas_invalidas += 1

                        registro_cajas.append({
                            "particion": particion,
                            "imagen": nombre_imagen,
                            "anotacion_id": ann.get("id", ""),
                            "categoria": categoria,
                            "decision": "excluida",
                            "motivo": "anchura_o_altura_no_positiva",
                        })

                        continue

                    # Convertir coordenadas al tamaño físico.
                    x1_original = x * sx
                    y1_original = y * sy
                    x2_original = (x + w) * sx
                    y2_original = (y + h) * sy

                    # Recortar a los límites de la imagen.
                    x1 = max(0.0, min(float(W), x1_original))
                    y1 = max(0.0, min(float(H), y1_original))
                    x2 = max(0.0, min(float(W), x2_original))
                    y2 = max(0.0, min(float(H), y2_original))

                    if x2 <= x1 or y2 <= y1:

                        motivos.add(
                            "caja_vacia_despues_del_recorte"
                        )

                        anotacion_incierta = True
                        cajas_invalidas += 1

                        registro_cajas.append({
                            "particion": particion,
                            "imagen": nombre_imagen,
                            "anotacion_id": ann.get("id", ""),
                            "categoria": categoria,
                            "decision": "excluida",
                            "motivo": "caja_vacia_despues_del_recorte",
                        })

                        continue

                    modificada = (
                        abs(x1 - x1_original) > 1e-6
                        or abs(y1 - y1_original) > 1e-6
                        or abs(x2 - x2_original) > 1e-6
                        or abs(y2 - y2_original) > 1e-6
                    )

                    if modificada:

                        cajas_corregidas += 1
                        motivos.add("caja_recortada")

                    # Formato YOLO.
                    xc = ((x1 + x2) / 2) / W
                    yc = ((y1 + y2) / 2) / H
                    wn = (x2 - x1) / W
                    hn = (y2 - y1) / H

                    etiquetas.append(
                        f"{ID_CLASE[categoria]} "
                        f"{xc:.8f} {yc:.8f} "
                        f"{wn:.8f} {hn:.8f}"
                    )

                    conteo_clases[categoria] += 1

                    if modificada:

                        registro_cajas.append({
                            "particion": particion,
                            "imagen": nombre_imagen,
                            "anotacion_id": ann.get("id", ""),
                            "categoria": categoria,
                            "decision": "recortada",
                            "motivo": "limites_de_la_imagen",
                        })

                # ------------------------------------------------
                # Decidir el estado de la imagen
                # ------------------------------------------------

                if etiquetas:

                    estado = "incluida"

                    if anotacion_incierta:
                        motivos.add(
                            "otras_anotaciones_incompletas_o_dudosas"
                        )

                elif anotacion_incierta:

                    manifiesto.append({
                        "particion": particion,
                        "imagen": nombre_imagen,
                        "estado": "excluida_revision",
                        "motivo": ";".join(sorted(motivos)),
                        "cajas_validas": 0,
                    })

                    estad["imagenes_excluidas"] += 1
                    estad["imagenes_pendientes_revision"] += 1
                    estad["cajas_invalidas_excluidas"] += cajas_invalidas

                    continue

                else:

                    estado = "incluida_negativa"

                    if not anotaciones:
                        motivos.add("sin_anotaciones")

                    motivos.add("sin_cajas_validas_de_fauna")

                # ------------------------------------------------
                # Copiar imagen y crear etiqueta
                # ------------------------------------------------

                destino_img.parent.mkdir(
                    parents=True, exist_ok=True
                )

                destino_txt.parent.mkdir(
                    parents=True, exist_ok=True
                )

                shutil.copy2(origen, destino_img)

                contenido_txt = (
                    "\n".join(etiquetas) + "\n"
                    if etiquetas else ""
                )

                destino_txt.write_text(
                    contenido_txt,
                    encoding="utf-8"
                )

                manifiesto.append({
                    "particion": particion,
                    "imagen": nombre_imagen,
                    "estado": estado,
                    "motivo": ";".join(sorted(motivos)),
                    "cajas_validas": len(etiquetas),
                })

                if estado == "incluida":

                    estad["imagenes_incluidas"] += 1
                    estad["cajas_validas"] += len(etiquetas)

                else:

                    estad["imagenes_negativas"] += 1

                estad["cajas_recortadas"] += cajas_corregidas
                estad["cajas_invalidas_excluidas"] += cajas_invalidas

            # ----------------------------------------------------
            # Resumen de partición
            # ----------------------------------------------------

            resumen[particion] = {
                "estadisticas": dict(estad),
                "cajas_por_clase": dict(conteo_clases),
            }

            print(f"\n--- {particion} ---")

            for clave, valor in estad.items():
                print(f"{clave}: {valor}")

        # ========================================================
        # 4. GUARDAR ARCHIVOS DE CONTROL
        # ========================================================

        with (OUT_DIR / "classes.json").open(
            "w", encoding="utf-8"
        ) as f:

            json.dump(
                {
                    str(i): nombre
                    for i, nombre in enumerate(CLASES)
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        with (OUT_DIR / "manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:

            campos = [
                "particion",
                "imagen",
                "estado",
                "motivo",
                "cajas_validas",
            ]

            escritor = csv.DictWriter(
                f, fieldnames=campos
            )

            escritor.writeheader()
            escritor.writerows(manifiesto)

        with (OUT_DIR / "registro_cajas.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:

            campos = [
                "particion",
                "imagen",
                "anotacion_id",
                "categoria",
                "decision",
                "motivo",
            ]

            escritor = csv.DictWriter(
                f, fieldnames=campos
            )

            escritor.writeheader()
            escritor.writerows(registro_cajas)

        with (OUT_DIR / "resumen.json").open(
            "w", encoding="utf-8"
        ) as f:

            json.dump(
                resumen,
                f,
                ensure_ascii=False,
                indent=2
            )

        data_yaml = {
            "path": str(OUT_DIR),
            "train": "images/train",
            "val": "images/cis_val",
            "test": "images/cis_test",
            "names": {
                i: nombre
                for i, nombre in enumerate(CLASES)
            },
        }

        with (OUT_DIR / "data.yaml").open(
            "w", encoding="utf-8"
        ) as f:

            yaml.safe_dump(
                data_yaml,
                f,
                sort_keys=False,
                allow_unicode=True
            )

        print("\nConstrucción del dataset terminada.")


    # ============================================================
    # 5. AUDITORÍA FINAL
    #    Se ejecuta tanto después de construir como cuando
    #    el dataset ya existe.
    # ============================================================

    print("\n==============================================")
    print("AUDITORÍA FINAL DEL DATASET")
    print("==============================================")

    errores_validacion = []

    totales = Counter()

    for particion in PARTICIONES:

        carpeta_img = OUT_DIR / "images" / particion
        carpeta_lbl = OUT_DIR / "labels" / particion

        if not carpeta_img.is_dir() or not carpeta_lbl.is_dir():

            errores_validacion.append(
                f"{particion}: falta directorio de imágenes o etiquetas"
            )

            continue

        imagenes_salida = [
            p for p in carpeta_img.rglob("*")
            if p.is_file()
        ]

        etiquetas_salida = list(
            carpeta_lbl.rglob("*.txt")
        )

        # --------------------------------------------------------
        # 5.1 Comparación por ruta relativa y nombre base.
        # No depende de JPG, JPEG, PNG u otra extensión.
        # --------------------------------------------------------

        indices_imagenes = {
            (
                p.relative_to(carpeta_img).parent,
                p.stem
            )
            for p in imagenes_salida
        }

        indices_etiquetas = {
            (
                p.relative_to(carpeta_lbl).parent,
                p.stem
            )
            for p in etiquetas_salida
        }

        imagenes_sin_etiqueta = (
            indices_imagenes - indices_etiquetas
        )

        etiquetas_sin_imagen = (
            indices_etiquetas - indices_imagenes
        )

        if imagenes_sin_etiqueta:

            errores_validacion.append(
                f"{particion}: {len(imagenes_sin_etiqueta)} "
                "imágenes sin etiqueta"
            )

        if etiquetas_sin_imagen:

            errores_validacion.append(
                f"{particion}: {len(etiquetas_sin_imagen)} "
                "etiquetas sin imagen"
            )

        # --------------------------------------------------------
        # 5.2 Comprobar etiquetas YOLO
        # --------------------------------------------------------

        etiquetas_vacias = 0
        cajas_particion = 0

        for ruta_lbl in etiquetas_salida:

            contenido = ruta_lbl.read_text(
                encoding="utf-8"
            ).strip()

            # Etiqueta vacía: imagen negativa válida.
            if not contenido:

                etiquetas_vacias += 1
                continue

            for numero_linea, linea in enumerate(
                contenido.splitlines(), start=1
            ):

                partes = linea.split()

                if len(partes) != 5:

                    errores_validacion.append(
                        f"{particion}/{ruta_lbl.name}: "
                        f"formato incorrecto en línea {numero_linea}"
                    )

                    continue

                try:

                    clase = int(partes[0])

                    valores = [
                        float(v) for v in partes[1:]
                    ]

                    xc, yc, wn, hn = valores

                    if not 0 <= clase < len(CLASES):
                        raise ValueError("ID de clase fuera de rango")

                    if not all(
                        math.isfinite(v) for v in valores
                    ):
                        raise ValueError("Valor no finito")

                    if not all(
                        0 <= v <= 1 for v in valores
                    ):
                        raise ValueError("Coordenada fuera de rango")

                    if wn <= 0 or hn <= 0:
                        raise ValueError("Ancho o alto no positivo")

                    cajas_particion += 1

                except Exception as exc:

                    errores_validacion.append(
                        f"{particion}/{ruta_lbl.name}: "
                        f"línea {numero_linea}: {exc}"
                    )

        # --------------------------------------------------------
        # 5.3 Comparar etiquetas vacías con manifiesto
        # --------------------------------------------------------

        ruta_manifest = OUT_DIR / "manifest.csv"

        if ruta_manifest.is_file():

            with ruta_manifest.open(
                "r", newline="", encoding="utf-8"
            ) as f:

                filas_manifest = [
                    fila for fila in csv.DictReader(f)
                    if fila["particion"] == particion
                ]

            negativas_manifest = sum(
                1 for fila in filas_manifest
                if fila["estado"] == "incluida_negativa"
            )

            if etiquetas_vacias != negativas_manifest:

                errores_validacion.append(
                    f"{particion}: etiquetas vacías "
                    f"({etiquetas_vacias}) no coinciden con "
                    f"negativas del manifiesto "
                    f"({negativas_manifest})"
                )

        else:

            errores_validacion.append(
                "No se encontró manifest.csv"
            )

        # --------------------------------------------------------
        # 5.4 Estadísticas
        # --------------------------------------------------------

        totales["imagenes"] += len(imagenes_salida)
        totales["etiquetas"] += len(etiquetas_salida)
        totales["negativas"] += etiquetas_vacias
        totales["cajas"] += cajas_particion

        print(f"\n--- {particion} ---")
        print("Imágenes:", len(imagenes_salida))
        print("Etiquetas:", len(etiquetas_salida))
        print("Etiquetas vacías:", etiquetas_vacias)
        print("Cajas YOLO:", cajas_particion)
        print("Imágenes sin etiqueta:", len(imagenes_sin_etiqueta))
        print("Etiquetas sin imagen:", len(etiquetas_sin_imagen))


    # ============================================================
    # 6. COMPROBACIÓN GLOBAL
    # ============================================================

    print("\n==============================================")
    print("RESULTADO GLOBAL")
    print("==============================================")

    print("Clases:", len(CLASES))
    print("Imágenes:", totales["imagenes"])
    print("Etiquetas:", totales["etiquetas"])
    print("Imágenes negativas:", totales["negativas"])
    print("Cajas YOLO:", totales["cajas"])

    if totales["imagenes"] != totales["etiquetas"]:

        errores_validacion.append(
            "El total de imágenes no coincide con el total de etiquetas."
        )

    # ------------------------------------------------------------
    # 6.1 Mostrar errores de forma controlada
    # ------------------------------------------------------------

    if errores_validacion:

        ruta_errores = OUT_DIR / "errores_validacion.txt"

        ruta_errores.write_text(
            "\n".join(errores_validacion),
            encoding="utf-8"
        )

        print("\nVALIDACIÓN: SE ENCONTRARON ERRORES")
        print("Total de errores:", len(errores_validacion))

        for error in errores_validacion[:20]:
            print("-", error)

        if len(errores_validacion) > 20:
            print(
                f"... y {len(errores_validacion) - 20} "
                "errores adicionales."
            )

        print("\nDetalle completo:", ruta_errores)

        raise RuntimeError(
            f"Auditoría final: {len(errores_validacion)} "
            "errores. Revisa errores_validacion.txt."
        )

    else:

        print("\nVALIDACIÓN FINAL: CORRECTA")
        print("Correspondencia imágenes-etiquetas: correcta.")
        print("Formato de etiquetas YOLO: correcto.")
        print("Coordenadas normalizadas: correctas.")
        print("Etiquetas negativas: correctas.")
        print("Integridad de las cinco particiones: correcta.")

    print("\nDirectorio del dataset:")
    print(OUT_DIR)

    print("\nOriginales modificados: NO")


def main():
    parser = argparse.ArgumentParser(description="Preparación y auditoría del dataset CCT20 para YOLO")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/CCT20"))
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["audit", "build", "all"], default="all")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    json_dir = dataset_dir / "eccv_18_annotation_files"
    image_dir = dataset_dir / "eccv_18_all_images_sm"
    out_dir = args.output_dir or (args.project_dir / "dataset_yolo_fase3_negativos")
    if not json_dir.is_dir():
        raise FileNotFoundError(f"No existe directorio JSON: {json_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No existe directorio de imágenes: {image_dir}")
    if args.mode in ("audit", "all"):
        audit_source(dataset_dir, json_dir, image_dir)
    if args.mode in ("build", "all"):
        build_or_audit(dataset_dir, json_dir, image_dir, args.project_dir, out_dir)

if __name__ == "__main__":
    main()
