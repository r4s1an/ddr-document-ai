from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RE_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("\n", " ")).strip()

def norm(s: Any) -> str:
    t = clean(s).lower()
    t = t.replace("：", ":").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"\([^)]*\)", "", t)
    t = t.replace("()", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_float(s: Any) -> Optional[float]:
    m = RE_NUM.search(clean(s))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def median_int(vals: List[int]) -> int:
    if not vals:
        return 0
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) // 2

@dataclass
class ColModel:
    x_ranges: Dict[str, Tuple[float, float]]
    header_max_y: float
    row_thresh: int

def _box_to_xyxy(box: Any) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        if isinstance(box, list) and len(box) >= 4 and all(isinstance(v, (int, float)) for v in box[:4]):
            x1, y1, x2, y2 = map(float, box[:4])
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return x1, y1, x2, y2

        if isinstance(box, list) and box and isinstance(box[0], (list, tuple)) and len(box[0]) == 2:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            return min(xs), min(ys), max(xs), max(ys)

        if isinstance(box, list) and len(box) == 8 and all(isinstance(v, (int, float)) for v in box):
            xs = list(map(float, box[0::2]))
            ys = list(map(float, box[1::2]))
            return min(xs), min(ys), max(xs), max(ys)

        if hasattr(box, "tolist"):
            return _box_to_xyxy(box.tolist())
    except Exception:
        return None, None, None, None
    return None, None, None, None

def _load_items(json_path: str) -> Tuple[List[dict], int]:
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8-sig", errors="ignore"))

    rec_texts = []
    rec_boxes = []

    if isinstance(data, dict):
        rec_texts = data.get("rec_texts") or []
        rec_boxes = data.get("rec_boxes") or data.get("dt_boxes") or data.get("boxes") or data.get("rec_polys") or []
    if not rec_texts or not rec_boxes:
        raise ValueError("JSON has no rec_texts/rec_boxes")

    items = []
    heights = []
    n = min(len(rec_texts), len(rec_boxes))

    for i in range(n):
        t = rec_texts[i]
        b = rec_boxes[i]
        if not t or not b:
            continue

        x1, y1, x2, y2 = _box_to_xyxy(b)
        if x1 is None:
            continue

        txt = clean(t)
        if not txt:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        h = max(1, int(y2 - y1))
        items.append({"text": txt, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy})
        heights.append(h)

    mh = median_int(heights) or 18
    row_thresh = max(14, int(mh * 0.85))
    return items, row_thresh
def _detect_headers_and_xranges(items: List[dict], row_thresh: int) -> Tuple[Dict[str, Tuple[float, float]], float]:
    # First, find the header row by looking for depth-related keywords
    header_candidates = []
    for it in items:
        nt = norm(it["text"])
        if any(kw in nt for kw in ["depth", "mmd", "mtvd", "tvd", "shows", "lithology", "description"]):
            header_candidates.append((it["y1"], it))
    
    if not header_candidates:
        raise RuntimeError("Header not found")
    
    header_candidates.sort(key=lambda z: z[0])
    header_y = header_candidates[0][0]
    
    # Get all items in the header row
    header_items = [it for y, it in header_candidates if abs(y - header_y) <= row_thresh * 2]
    header_items.sort(key=lambda it: it["x1"])
    
    # Group nearby boxes horizontally (within 20px) to form compound headers
    groups = []
    for it in header_items:
        nt = norm(it["text"])
        if not groups or it["x1"] - groups[-1]["x2"] > 20:
            groups.append({
                "texts": [nt], 
                "x1": it["x1"], 
                "x2": it["x2"], 
                "y1": it["y1"], 
                "y2": it["y2"], 
                "cx": it["cx"]
            })
        else:
            groups[-1]["texts"].append(nt)
            groups[-1]["x2"] = it["x2"]
            groups[-1]["cx"] = (groups[-1]["x1"] + it["x2"]) / 2.0
            groups[-1]["y2"] = max(groups[-1]["y2"], it["y2"])
    
    # Now match compound headers
    header_boxes: Dict[str, Tuple[float, float, float, float, float]] = {}
    
    for g in groups:
        combined = " ".join(g["texts"])
        x1, y1, x2, y2, cx = g["x1"], g["y1"], g["x2"], g["y2"], g["cx"]
        
        if "start" in combined and "depth" in combined and "mmd" in combined:
            header_boxes["start_depth_mmd"] = (x1, y1, x2, y2, cx)
        if "end" in combined and "depth" in combined and "mmd" in combined:
            header_boxes["end_depth_mmd"] = (x1, y1, x2, y2, cx)
        if "start" in combined and "depth" in combined and ("mtvd" in combined or "tvd" in combined):
            header_boxes["start_depth_mtvd"] = (x1, y1, x2, y2, cx)
        if "end" in combined and "depth" in combined and ("mtvd" in combined or "tvd" in combined):
            header_boxes["end_depth_mtvd"] = (x1, y1, x2, y2, cx)
        if "shows" in combined and "description" in combined:
            header_boxes["shows_description"] = (x1, y1, x2, y2, cx)
        if "lithology" in combined and "description" in combined:
            header_boxes["lithology_description"] = (x1, y1, x2, y2, cx)
    
    if "start_depth_mmd" not in header_boxes or "lithology_description" not in header_boxes:
        raise RuntimeError(f"Header incomplete: {list(header_boxes.keys())}")
    
    col_order = [
        "start_depth_mmd",
        "end_depth_mmd",
        "start_depth_mtvd",
        "end_depth_mtvd",
        "shows_description",
        "lithology_description",
    ]
    present = [(k, header_boxes[k][4]) for k in col_order if k in header_boxes]
    present.sort(key=lambda z: z[1])
    
    centers = [cx for _, cx in present]
    keys = [k for k, _ in present]
    
    PAD = 170.0
    x_ranges: Dict[str, Tuple[float, float]] = {}
    for i, k in enumerate(keys):
        left = (centers[i - 1] + centers[i]) / 2.0 if i > 0 else (centers[i] - PAD)
        right = (centers[i] + centers[i + 1]) / 2.0 if i < len(keys) - 1 else (centers[i] + PAD)
        x_ranges[k] = (left, right)
    
    header_max_y = max(v[3] for v in header_boxes.values())
    return x_ranges, float(header_max_y)

def _assign_col(cm: ColModel, x1: float, x2: float, cx: float) -> Optional[str]:
    best_k = None
    best_ov = 0.0
    w = max(1.0, x2 - x1)
    for k, (lo, hi) in cm.x_ranges.items():
        ov = max(0.0, min(x2, hi) - max(x1, lo))
        if ov > best_ov:
            best_ov = ov
            best_k = k
    if best_k is None:
        return None
    if best_ov / w >= 0.25:
        return best_k
    for k, (lo, hi) in cm.x_ranges.items():
        if lo <= cx < hi:
            return k
    return best_k

def _band_rows_by_start_depth_mmd(body: List[dict], cm: ColModel) -> List[Tuple[float, float, float]]:
    lo, hi = cm.x_ranges["start_depth_mmd"]
    anchors = []
    for it in body:
        if lo <= it["cx"] < hi:
            v = parse_float(it["text"])
            if v is not None:
                anchors.append((it["cy"], v))

    anchors.sort(key=lambda z: z[0])

    dedup = []
    for y, d in anchors:
        if not dedup or abs(y - dedup[-1][0]) > cm.row_thresh:
            dedup.append((y, d))

    if not dedup:
        return []

    bands = []
    for i, (y, d) in enumerate(dedup):
        y_next = dedup[i + 1][0] if i < len(dedup) - 1 else 10**9
        bands.append((y, y_next, d))
    return bands

def _has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)

def _fix_wrapped_shows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i in range(1, len(rows)):
        cur = rows[i]
        prev = rows[i - 1]

        cur_sh = clean(cur.get("shows_description"))
        prev_sh = clean(prev.get("shows_description"))
        cur_li = clean(cur.get("lithology_description"))

        if not cur_sh:
            continue
        if len(cur_sh) > 25:
            continue
        if _has_digit(cur_sh):
            continue
        if not prev_sh:
            continue
        if not cur_li:
            continue

        prev["shows_description"] = clean(prev_sh + " " + cur_sh)
        cur["shows_description"] = None

    return rows

def parse_lithology_information_fragment(json_path: str, col_model: Optional[ColModel] = None) -> Tuple[List[Dict[str, Any]], Optional[ColModel]]:
    items, row_thresh = _load_items(json_path)
    items.sort(key=lambda it: (it["y1"], it["x1"]))

    inferred = None
    try:
        x_ranges, header_max_y = _detect_headers_and_xranges(items, row_thresh)
        inferred = ColModel(x_ranges=x_ranges, header_max_y=header_max_y, row_thresh=row_thresh)
    except Exception:
        inferred = None

    if inferred is not None:
        col_model = inferred

    if col_model is None:
        return [], None

    if inferred is not None:
        body = [it for it in items if it["y1"] > (col_model.header_max_y - 2)]
    else:
        body = list(items)

    body.sort(key=lambda it: (it["y1"], it["x1"]))

    if "start_depth_mmd" not in col_model.x_ranges:
        return [], col_model

    bands = _band_rows_by_start_depth_mmd(body, col_model)
    if not bands:
        return [], col_model

    out_rows: List[Dict[str, Any]] = []

    for y0, y1, start_mmd in bands:
        band_items = [it for it in body if (y0 - 2) <= it["cy"] < (y1 - 2)]
        col_items: Dict[str, List[dict]] = {k: [] for k in col_model.x_ranges.keys()}

        for it in band_items:
            k = _assign_col(col_model, it["x1"], it["x2"], it["cx"])

            if k:
                col_items[k].append(it)

        for k in col_items:
            col_items[k].sort(key=lambda it: (it["y1"], it["x1"]))

        end_mmd = None
        if "end_depth_mmd" in col_items:
            end_mmd = parse_float(clean(" ".join(it["text"] for it in col_items["end_depth_mmd"])))

        start_mtvd = None
        if "start_depth_mtvd" in col_items:
            start_mtvd = parse_float(clean(" ".join(it["text"] for it in col_items["start_depth_mtvd"])))

        end_mtvd = None
        if "end_depth_mtvd" in col_items:
            end_mtvd = parse_float(clean(" ".join(it["text"] for it in col_items["end_depth_mtvd"])))

        shows_desc = None
        if "shows_description" in col_items:
            blob = clean(" ".join(it["text"] for it in col_items["shows_description"]))
            shows_desc = blob or None

        lith_desc = None
        if "lithology_description" in col_items:
            blob = clean(" ".join(it["text"] for it in col_items["lithology_description"]))
            lith_desc = blob or None

        out_rows.append(
            {
                "start_depth_mmd": float(start_mmd) if start_mmd is not None else None,
                "end_depth_mmd": end_mmd,
                "start_depth_mtvd": start_mtvd,
                "end_depth_mtvd": end_mtvd,
                "shows_description": shows_desc,
                "lithology_description": lith_desc,
            }
        )
    out_rows = _fix_wrapped_shows(out_rows)
    return out_rows, col_model

def merge_lithology_information_rows(base: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[int], Dict[str, Any]] = {}
    order: List[Tuple[int]] = []

    def key(r: Dict[str, Any]) -> Optional[Tuple[int]]:
        v = r.get("start_depth_mmd")
        if v is None:
            return None
        try:
            return (int(round(float(v) * 1000.0)),)
        except Exception:
            return None

    for r in base:
        k = key(r)
        if k and k not in by_key:
            by_key[k] = dict(r)
            order.append(k)

    for r in incoming:
        k = key(r)
        if not k:
            continue
        if k not in by_key:
            by_key[k] = dict(r)
            order.append(k)
            continue

        b = by_key[k]
        for f in (
            "end_depth_mmd",
            "start_depth_mtvd",
            "end_depth_mtvd",
            "shows_description",
            "lithology_description",
        ):
            nv = r.get(f)
            ov = b.get(f)

            if (ov is None or clean(ov) == "") and (nv is not None and clean(nv) != ""):
                b[f] = nv
                continue

            if nv is None or clean(nv) == "":
                continue

            if f in ("shows_description", "lithology_description"):
                if len(clean(nv)) > len(clean(ov)) + 2:
                    b[f] = nv

    return [by_key[k] for k in order]

def extract_lithology_information_from_jsons(json_paths: List[str], debug: bool = False) -> List[Dict[str, Any]]:
    cm: Optional[ColModel] = None
    merged: List[Dict[str, Any]] = []

    for jp in json_paths:
        rows, cm = parse_lithology_information_fragment(jp, col_model=cm)
        if rows:
            merged = merge_lithology_information_rows(merged, rows)

    if debug:
        for i, r in enumerate(merged, 1):
            print(f"{i:02d}. {r}")

    return merged

def _sort_key_from_path(p: Path) -> Tuple[int, int, str]:
    s = str(p).lower()
    pm = re.search(r"page_(\d+)", s)
    tm = re.search(r"table_(\d+)", s)
    page = int(pm.group(1)) if pm else 0
    table = int(tm.group(1)) if tm else 0
    return page, table, str(p)

def parse_lithology_information_rows(processed_ddr_dir: Path, document_id: int) -> List[Dict[str, Any]]:
    frags: List[Path] = []
    for page_dir in sorted(processed_ddr_dir.glob("page_*")):
        li_dir = page_dir / "section_tables" / "lithology_information"
        if li_dir.exists():
            frags.extend(sorted(li_dir.rglob("*_table_res.json"), key=_sort_key_from_path))

    if not frags:
        return []

    rows = extract_lithology_information_from_jsons([str(p) for p in frags], debug=False)

    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["document_id"] = document_id
        out.append(rr)

    return out