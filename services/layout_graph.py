from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any

def _iou(b1, b2) -> float:
    x1,y1,x2,y2 = b1
    X1,Y1,X2,Y2 = b2
    ix1, iy1 = max(x1,X1), max(y1,Y1)
    ix2, iy2 = min(x2,X2), min(y2,Y2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    a1 = max(0,(x2-x1))*max(0,(y2-y1))
    a2 = max(0,(X2-X1))*max(0,(Y2-Y1))
    union = a1 + a2 - inter
    return inter/union if union else 0.0

def _dedupe_by_iou(items: List[Dict[str, Any]], iou_thr: float = 0.9) -> List[Dict[str, Any]]:
    out = []
    by_label = {}
    for it in items:
        by_label.setdefault(it["label"], []).append(it)

    for label, group in by_label.items():
        group = sorted(group, key=lambda x: x["conf"], reverse=True)
        kept = []
        for cand in group:
            if any(_iou(cand["bbox"], k["bbox"]) >= iou_thr for k in kept):
                continue
            kept.append(cand)
        out.extend(kept)

    return sorted(out, key=lambda x: x["idx"])

def _x_overlap_ratio(a, b) -> float:
    ax1,_,ax2,_ = a
    bx1,_,bx2,_ = b
    inter = max(0, min(ax2,bx2) - max(ax1,bx1))
    denom = max(1e-6, min(ax2-ax1, bx2-bx1))
    return inter / denom

def parse_detections_json(dets: list[dict], crops_dir: Path) -> List[Dict[str, Any]]:
    items = []
    for d in dets:
        idx = int(d["i"])
        label = str(d["label"])
        conf = float(d.get("conf", 0.0))
        x1, y1, x2, y2 = map(int, d["box"])

        crop_path = crops_dir / f"{idx:03d}_{label}.png"

        it = {
            "idx": idx,
            "label": label,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            "crop_path": str(crop_path).replace("\\", "/"),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "w": (x2 - x1),
            "h": (y2 - y1),
        }
        items.append(it)

    return items

def reading_order(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda it: (it["y1"], it["x1"]))

def link_headers(items: List[Dict[str, Any]], slack: int = 20) -> List[Dict[str, Any]]:
    headers = [it for it in items if it["label"] == "section_header"]
    targets = [it for it in items if it["label"] in ("table", "plain_text")]

    for t in targets:
        best = None
        best_score = -1e18
        for h in headers:
            if h["y2"] > t["y1"] + slack:
                continue
            dy = t["y1"] - h["y2"]
            xov = _x_overlap_ratio(h["bbox"], t["bbox"])
            score = (xov * 1000.0) - float(dy)
            if score > best_score:
                best_score = score
                best = h

        t["linked_header_idx"] = best["idx"] if best else None
        t["linked_header_crop"] = best["crop_path"] if best else None

    return items

def build_and_save_layout(page_dir: Path) -> Dict[str, Any]:
    crops_dir = page_dir / "crops"

    detections_json = next(page_dir.glob("*_detections.json"), None)
    if detections_json is None:
        raise FileNotFoundError(
            f"[LAYOUT] No *_detections.json found in {page_dir}"
        )

    dets = json.loads(detections_json.read_text(encoding="utf-8"))

    items = parse_detections_json(dets, crops_dir)
    items = _dedupe_by_iou(items, iou_thr=0.9)
    items = reading_order(items)
    items = link_headers(items, slack=20)

    layout = {
        "page_dir": str(page_dir).replace("\\", "/"),
        "detections_txt": str(detections_json).replace("\\", "/"),
        "num_items": len(items),
        "items": items,
    }

    out_path = page_dir / "layout.json"
    out_path.write_text(
        json.dumps(layout, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return layout
