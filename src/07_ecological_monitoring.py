\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monitoreo ecológico por cámara/ubicación a partir de metadatos CCT20.

Fail-closed: no infiere si no puede demostrar ubicación y orden temporal.
Las detecciones por imagen NO equivalen a individuos ni abundancia poblacional.
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import yaml

CLASSES = ["badger","bird","bobcat","cat","coyote","deer","dog","fox",
           "opossum","rabbit","raccoon","rodent","skunk","squirrel"]

def first_value(obj, candidates):
    for key in candidates:
        value = obj.get(key)
        if value not in (None, ""):
            return key, value
    return None, None

def parse_time(value):
    if value is None: return None
    if isinstance(value, (int,float)):
        # Unix timestamp if plausible
        try: return datetime.fromtimestamp(float(value))
        except Exception: return None
    raw=str(value).strip().replace("Z","+00:00")
    for fmt in (None,"%Y-%m-%d %H:%M:%S","%Y-%m-%d","%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.fromisoformat(raw) if fmt is None else datetime.strptime(raw,fmt)
        except Exception: pass
    return None

def load_json(path):
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or not isinstance(data.get("images"),list):
        raise ValueError(f"JSON COCO no válido: {path}")
    return data

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-dir",type=Path,required=True)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--split",default="trans_test")
    ap.add_argument("--location-field",default=None)
    ap.add_argument("--datetime-field",default=None)
    ap.add_argument("--target-location",default=None)
    ap.add_argument("--max-images",type=int,default=500)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--conf",type=float,default=.25)
    ap.add_argument("--iou",type=float,default=.70)
    ap.add_argument("--device",default="cpu")
    args=ap.parse_args()

    train=load_json(args.json_dir/"train_annotations.json")
    test=load_json(args.json_dir/f"{args.split}_annotations.json")
    sample=(train.get("images") or [])+(test.get("images") or [])
    if not sample: raise RuntimeError("No hay imágenes en los JSON.")
    loc_candidates=[args.location_field] if args.location_field else [
        "location","location_id","camera_id","camera","seq_id"]
    time_candidates=[args.datetime_field] if args.datetime_field else [
        "datetime","date","timestamp","capture_time","time"]
    loc_key=next((k for k in loc_candidates if any(im.get(k) not in (None,"") for im in sample)),None)
    time_key=next((k for k in time_candidates if any(im.get(k) not in (None,"") for im in sample)),None)
    if not loc_key or not time_key:
        keys=sorted({k for im in sample for k in im.keys()})
        raise RuntimeError("No se pudo identificar metadata de cámara/fecha de forma segura. "
                           f"Campos disponibles: {keys}. No se realizará inferencia.")
    train_locs={str(im[loc_key]) for im in train["images"] if im.get(loc_key) not in (None,"")}
    test_locs=sorted({str(im[loc_key]) for im in test["images"] if im.get(loc_key) not in (None,"")})
    unseen=sorted(set(test_locs)-train_locs)
    print("Campo ubicación:",loc_key,"| Campo fecha:",time_key)
    print("Ubicaciones en train:",len(train_locs),"| en",args.split,":",len(test_locs))
    print("Ubicaciones no presentes en train:",unseen[:50])
    if not unseen:
        raise RuntimeError("No hay ubicaciones no vistas verificables en esta partición. "
                           "No se etiquetará CIS/TRANS como generalización geográfica.")
    target=str(args.target_location) if args.target_location else None
    if target is None:
        raise RuntimeError("Configure --target-location con un ID de la lista de ubicaciones no vistas.")
    if target not in unseen:
        raise ValueError(f"La ubicación {target} no está demostrada como no vista en train.")
    frames=[]
    for im in test["images"]:
        if str(im.get(loc_key))!=target: continue
        dt=parse_time(im.get(time_key))
        if dt is None: continue
        fname=Path(str(im["file_name"])).name
        path=args.dataset/"images"/args.split/fname
        if not path.is_file():
            # preserve nested relative path if it exists
            rel=args.dataset/"images"/args.split/Path(str(im["file_name"]))
            if rel.is_file(): path=rel
            else: continue
        frames.append((dt,path,im))
    frames.sort(key=lambda x:x[0])
    if not frames: raise RuntimeError("No hay imágenes legibles con fecha válida para esa ubicación.")
    frames=frames[:args.max_images]
    from ultralytics import YOLO
    model=YOLO(str(args.model))
    out=args.results/"07_monitoreo_ecologico"/f"location_{target}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True,exist_ok=False)
    image_rows=[]; detection_rows=[]; by_species=Counter()
    for idx,(dt,path,meta) in enumerate(frames,1):
        pred=model.predict(str(path),imgsz=args.imgsz,conf=args.conf,iou=args.iou,
                           device=args.device,verbose=False,save=False)[0]
        boxes=pred.boxes
        n=0 if boxes is None else len(boxes)
        species=Counter()
        if boxes is not None:
            for b in boxes:
                cid=int(b.cls.item()); conf=float(b.conf.item())
                name=CLASSES[cid] if 0<=cid<len(CLASSES) else f"class_{cid}"
                species[name]+=1; by_species[name]+=1
                xyxy=[float(v) for v in b.xyxy[0].tolist()]
                detection_rows.append({"frame":idx,"datetime":dt.isoformat(),
                    "location_id":target,"image":path.name,"class_id":cid,
                    "species":name,"confidence":conf,"x1":xyxy[0],"y1":xyxy[1],
                    "x2":xyxy[2],"y2":xyxy[3]})
        image_rows.append({"frame":idx,"datetime":dt.isoformat(),"location_id":target,
            "image":path.name,"detections":n,"has_fauna":int(n>0),
            "species_detected":";".join(sorted(species))})
    def write_csv(path,rows):
        if not rows: path.write_text("",encoding="utf-8"); return
        with path.open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_csv(out/"actividad_por_imagen.csv",image_rows)
    write_csv(out/"detalle_detecciones.csv",detection_rows)
    species_rows=[{"species":k,"detections":v} for k,v in sorted(by_species.items())]
    write_csv(out/"conteo_por_especie.csv",species_rows)
    n=len(image_rows); positives=sum(r["has_fauna"] for r in image_rows)
    summary={"location_id":target,"split":args.split,"location_field":loc_key,
      "datetime_field":time_key,"images_processed":n,"images_with_fauna":positives,
      "empty_images":n-positives,"empty_image_rate":(n-positives)/n,
      "images_with_fauna_rate":positives/n,"raw_box_detections":len(detection_rows),
      "distinct_predicted_species":len(by_species),"species_detection_counts":dict(by_species),
      "period_start":image_rows[0]["datetime"],"period_end":image_rows[-1]["datetime"],
      "warning":"Box detections and trigger images are not independent individuals or population abundance."}
    (out/"resumen_monitoreo.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"informe_monitoreo.md").write_text(
      "# Informe de monitoreo ecológico\n\n"
      f"- Ubicación: `{target}` (no presente en train)\n- Partición: `{args.split}`\n"
      f"- Imágenes procesadas: {n}\n- Imágenes con detecciones: {positives}\n"
      f"- Imágenes vacías: {n-positives} ({summary['empty_image_rate']:.1%})\n"
      f"- Cajas detectadas: {len(detection_rows)}\n- Especies detectadas: {len(by_species)}\n"
      f"- Periodo: {summary['period_start']} a {summary['period_end']}\n\n"
      "**Limitación:** las detecciones por fotograma no representan individuos independientes "
      "ni permiten estimar abundancia poblacional.\n",encoding="utf-8")
    print("Resultados:",out)

if __name__=="__main__": main()
