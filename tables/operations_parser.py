from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RE_TIME = re.compile(r"\b([01]?\d|2[0-3])\s*[:.]\s*([0-5]\d)\b")
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")

def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("\n", " ")).strip()

def norm(s: Any) -> str:
    return clean(s).lower().replace("：", ":").replace("\u2013", "-").replace("\u2014", "-")

def parse_time(s: Any) -> Optional[str]:
    m = RE_TIME.search(clean(s))
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    return f"{hh:02d}:{mm:02d}"

def parse_depth(s: Any) -> Optional[float]:
    m = RE_NUM.search(clean(s))
    if not m:
        return None
    v = m.group(0)
    return float(v)

def median_int(vals: List[int]) -> int:
    if not vals:
        return 0
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) // 2

def split_main_sub(activity: str) -> Tuple[str, str]:
    a = clean(activity).replace("—", "-").replace("–", "-")
    if not a:
        return "", ""
    parts = [clean(p) for p in re.split(r"\s*-\s*", a, maxsplit=1)]
    if len(parts) == 2:
        return parts[0], re.sub(r"^[\s\-–—]+", "", parts[1]).strip()
    return a, ""

def _looks_like_time_leak(v: Any) -> bool:
    t = clean(v)
    return parse_time(t) is not None and len(t) <= 5

def _more_complete(a: Any, b: Any) -> bool:
    return len(clean(b)) > len(clean(a)) + 2

@dataclass
class ColModel:
    x_ranges: Dict[str, Tuple[float, float]]
    header_max_y: float
    row_thresh: int

def _load_items(json_path: str) -> Tuple[List[dict], int]:
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8-sig", errors="ignore"))

    rec_texts = []
    rec_boxes = []

    if isinstance(data, dict):
        rec_texts = data.get("rec_texts") or []
        rec_boxes = data.get("rec_boxes") or data.get("dt_boxes") or data.get("boxes") or []
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

def _detect_headers_and_xranges(items: List[dict], row_thresh: int) -> Tuple[Dict[str, Tuple[float, float]], float]:
    cands = []
    for it in items:
        nt = norm(it["text"])
        hit = []
        if ("start" in nt and "time" in nt) or nt == "start":
            hit.append("start_time")
        if ("end" in nt and "time" in nt) or nt == "end":
            hit.append("end_time")
        if "depth" in nt:
            hit.append("end_depth_mmd")
        if "activity" in nt or "main sub" in nt or "main - sub" in nt:
            hit.append("activity")
        if "state" in nt:
            hit.append("state")
        if "remark" in nt:
            hit.append("remark")
        if hit:
            cands.append((it["y1"], it["x1"], it["x2"], it["y2"], it["cx"], it["text"], hit))

    if not cands:
        raise RuntimeError("Header not found")

    cands.sort(key=lambda z: z[0])
    header_y = cands[0][0]
    header_band = [c for c in cands if abs(c[0] - header_y) <= row_thresh * 2]

    header_boxes: Dict[str, Tuple[float, float, float, float, float]] = {}

    def set_if_better(k: str, x1: float, y1: float, x2: float, y2: float, cx: float):
        prev = header_boxes.get(k)
        if prev is None:
            header_boxes[k] = (x1, y1, x2, y2, cx)
            return
        px1, py1, px2, py2, pcx = prev
        if (x2 - x1) > (px2 - px1) + 2 or (y1 < py1 and abs((x2 - x1) - (px2 - px1)) <= 10):
            header_boxes[k] = (x1, y1, x2, y2, cx)

    for y1, x1, x2, y2, cx, text, hits in header_band:
        nt = norm(text)
        if ("start" in nt and "time" in nt) or nt == "start":
            set_if_better("start_time", x1, y1, x2, y2, cx)
        if ("end" in nt and "time" in nt) or nt == "end":
            set_if_better("end_time", x1, y1, x2, y2, cx)
        if "depth" in nt:
            set_if_better("end_depth_mmd", x1, y1, x2, y2, cx)
        if "activity" in nt or "main sub" in nt or "main - sub" in nt:
            set_if_better("activity", x1, y1, x2, y2, cx)
        if "state" in nt:
            set_if_better("state", x1, y1, x2, y2, cx)
        if "remark" in nt:
            set_if_better("remark", x1, y1, x2, y2, cx)

    if "start_time" not in header_boxes or "activity" not in header_boxes or "remark" not in header_boxes:
        raise RuntimeError(f"Header incomplete: {list(header_boxes.keys())}")

    col_order = ["start_time", "end_time", "end_depth_mmd", "activity", "state", "remark"]
    present = [(k, header_boxes[k][4]) for k in col_order if k in header_boxes]
    present.sort(key=lambda z: z[1])

    centers = [cx for _, cx in present]
    keys = [k for k, _ in present]

    PAD = 140.0
    x_ranges: Dict[str, Tuple[float, float]] = {}
    for i, k in enumerate(keys):
        left = (centers[i - 1] + centers[i]) / 2.0 if i > 0 else (centers[i] - PAD)
        right = (centers[i] + centers[i + 1]) / 2.0 if i < len(keys) - 1 else (centers[i] + PAD)
        x_ranges[k] = (left, right)

    header_max_y = max(v[3] for v in header_boxes.values())
    return x_ranges, float(header_max_y)

def _band_rows_by_start_time(body: List[dict], cm: ColModel) -> List[Tuple[float, float, str]]:
    def in_col(k: str, cx: float) -> bool:
        lo, hi = cm.x_ranges[k]
        return lo <= cx < hi

    anchors = []
    for it in body:
        if "start_time" in cm.x_ranges and in_col("start_time", it["cx"]):
            t = parse_time(it["text"])
            if t:
                anchors.append((it["y1"], t))

    anchors.sort(key=lambda z: z[0])

    dedup = []
    for y, t in anchors:
        if not dedup or abs(y - dedup[-1][0]) > cm.row_thresh:
            dedup.append((y, t))

    if not dedup:
        return []

    bands = []
    for i, (y, t) in enumerate(dedup):
        y_next = dedup[i + 1][0] if i < len(dedup) - 1 else 10**9
        bands.append((y, y_next, t))
    return bands

def parse_operations_fragment(json_path: str, col_model: Optional[ColModel] = None) -> Tuple[List[Dict[str, Any]], Optional[ColModel]]:
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

    body = [it for it in items if it["y1"] > (col_model.header_max_y - 2 if inferred is not None else -10**9)]
    body.sort(key=lambda it: (it["y1"], it["x1"]))

    bands = _band_rows_by_start_time(body, col_model)
    if not bands:
        return [], col_model

    def assign_col(cx: float) -> Optional[str]:
        for k, (lo, hi) in col_model.x_ranges.items():
            if lo <= cx < hi:
                return k
        return None

    out_rows: List[Dict[str, Any]] = []

    for y0, y1, start_time in bands:
        band_items = [it for it in body if y0 - 2 <= it["y1"] < y1 - 2]
        col_items: Dict[str, List[dict]] = {k: [] for k in col_model.x_ranges.keys()}

        for it in band_items:
            k = assign_col(it["cx"])
            if k:
                col_items[k].append(it)

        for k in col_items:
            col_items[k].sort(key=lambda it: (it["y1"], it["x1"]))

        end_time = None
        for it in col_items.get("end_time", []):
            tt = parse_time(it["text"])
            if tt and tt != start_time:
                end_time = tt
                break

        end_depth = None
        depth_blob = clean(" ".join(it["text"] for it in col_items.get("end_depth_mmd", [])))
        if depth_blob:
            end_depth = parse_depth(depth_blob)

        activity_blob = clean(" ".join(it["text"] for it in col_items.get("activity", [])))
        main_activity, sub_activity = split_main_sub(activity_blob)

        state_text = clean(" ".join(it["text"] for it in col_items.get("state", [])[:1])).lower()
        state = None
        if state_text:
            m = re.search(r"[a-z0-9/]+", state_text)
            state = m.group(0) if m else None

        remark_blob = clean(" ".join(it["text"] for it in col_items.get("remark", [])))

        full_state_col = clean(" ".join(it["text"] for it in col_items.get("state", []))).lower()
        if full_state_col and state:
            if full_state_col.startswith(state):
                spill = clean(full_state_col[len(state):])
                if spill:
                    remark_blob = clean(" ".join([remark_blob, spill])) if remark_blob else spill

        out_rows.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "end_depth_mmd": end_depth,
                "main_activity": main_activity or None,
                "sub_activity": sub_activity or None,
                "state": state or None,
                "remark": remark_blob or None,
            }
        )

    return out_rows, col_model

def merge_operations_rows(base: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def key(r: Dict[str, Any]) -> str:
        return clean(r.get("start_time") or "")

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
        for f in ("end_time", "end_depth_mmd", "main_activity", "sub_activity", "state", "remark"):
            nv = r.get(f)
            ov = b.get(f)

            if (ov is None or clean(ov) == "") and (nv is not None and clean(nv) != ""):
                b[f] = nv
                continue

            if nv is None or clean(nv) == "":
                continue

            if f in ("end_time", "end_depth_mmd") and _looks_like_time_leak(ov) and not _looks_like_time_leak(nv):
                b[f] = nv
                continue

            if _more_complete(ov, nv):
                b[f] = nv

    return [by_key[k] for k in order]

def extract_operations_from_jsons(json_paths: List[str], debug: bool = False) -> List[Dict[str, Any]]:
    cm: Optional[ColModel] = None
    merged: List[Dict[str, Any]] = []

    for jp in json_paths:
        rows, cm = parse_operations_fragment(jp, col_model=cm)
        if rows:
            merged = merge_operations_rows(merged, rows)

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

def extract_operations_from_operations_dir(operations_dir: str, debug: bool = False) -> List[Dict[str, Any]]:
    root = Path(operations_dir)
    paths = sorted(root.rglob("*_table_res.json"), key=_sort_key_from_path)
    return extract_operations_from_jsons([str(p) for p in paths], debug=debug)

def parse_operations_rows(processed_ddr_dir: Path, document_id: int) -> List[Dict[str, Any]]:
    frags: List[Path] = []
    for page_dir in sorted(processed_ddr_dir.glob("page_*")):
        ops_dir = page_dir / "section_tables" / "operations"
        if ops_dir.exists():
            frags.extend(sorted(ops_dir.rglob("*_table_res.json"), key=_sort_key_from_path))

    if not frags:
        return []

    rows = extract_operations_from_jsons([str(p) for p in frags], debug=False)

    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["document_id"] = document_id
        out.append(rr)

    return out