from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RE_TIME = re.compile(r"\b([01]?\d|2[0-3])\s*[:.]\s*([0-5]\d)\b")
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")

CANON_KEYS = [
    "time",
    "class",
    "depth_to_top_mmd",
    "depth_to_bottom_md",
    "depth_to_top_mtvd",
    "depth_to_bottom_tvd",
    "highest_gas_percent",
    "lowest_gas_percent",
    "c1_ppm",
    "c2_ppm",
    "c3_ppm",
    "ic4_ppm",
    "ic5_ppm",
]


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


def parse_num(s: Any) -> Optional[float]:
    m = RE_NUM.search(clean(s))
    if not m:
        return None
    try:
        v = float(m.group(0))
    except Exception:
        return None
    return v


def median_int(vals: List[int]) -> int:
    if not vals:
        return 0
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) // 2


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

    rec_texts: List[Any] = []
    rec_boxes: List[Any] = []

    if isinstance(data, dict):
        rec_texts = data.get("rec_texts") or []
        rec_boxes = data.get("rec_boxes") or data.get("dt_boxes") or data.get("boxes") or []
    elif isinstance(data, list):
        for el in data:
            if (
                isinstance(el, list)
                and len(el) >= 2
                and isinstance(el[0], (list, tuple))
                and isinstance(el[1], (list, tuple))
                and len(el[1]) >= 1
            ):
                rec_boxes.append(el[0])
                rec_texts.append(el[1][0])

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
        items.append({"text": txt, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy, "w": x2 - x1})
        heights.append(h)

    mh = median_int(heights) or 18
    row_thresh = max(14, int(mh * 0.85))
    return items, row_thresh


def _is_headerish(t: str) -> bool:
    nt = norm(t)
    if "time" in nt:
        return True
    if nt == "class" or "class" in nt:
        return True
    if "depth" in nt or "top" in nt or "bottom" in nt:
        return True
    if nt in ("mmd", "md", "mtvd", "tvd"):
        return True
    if "highest" in nt or "lowest" in nt:
        return True
    if nt in ("c1", "c2", "c3", "ic4", "ic5"):
        return True
    if "ppm" in nt or nt in ("(ppm)", "%", "(%)"):
        return True
    return False


def _cluster_by_x(tokens: List[dict], gap: float) -> List[List[dict]]:
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda it: it["cx"])
    clusters: List[List[dict]] = []
    cur = [toks[0]]
    for it in toks[1:]:
        if abs(it["cx"] - cur[-1]["cx"]) <= gap:
            cur.append(it)
        else:
            clusters.append(cur)
            cur = [it]
    clusters.append(cur)
    return clusters


def _cluster_label(col: List[dict]) -> str:
    col2 = sorted(col, key=lambda it: (it["y1"], it["x1"]))
    s = clean(" ".join(it["text"] for it in col2))
    return s


def _map_header_label_to_key(label: str) -> Optional[str]:
    nt = norm(label)

    if "time" in nt:
        return "time"
    if "class" in nt:
        return "class"

    if "depth" in nt and "top" in nt and "mmd" in nt:
        return "depth_to_top_mmd"
    if "depth" in nt and "bottom" in nt and ("md" in nt and "mmd" not in nt):
        return "depth_to_bottom_md"
    if "depth" in nt and "top" in nt and "mtvd" in nt:
        return "depth_to_top_mtvd"
    if "depth" in nt and "bottom" in nt and "tvd" in nt and "mtvd" not in nt:
        return "depth_to_bottom_tvd"

    if "highest" in nt and "gas" in nt:
        return "highest_gas_percent"
    if "lowest" in nt and "gas" in nt:
        return "lowest_gas_percent"

    if re.search(r"\bc1\b", nt) and "ppm" in nt:
        return "c1_ppm"
    if re.search(r"\bc2\b", nt) and "ppm" in nt:
        return "c2_ppm"
    if re.search(r"\bc3\b", nt) and "ppm" in nt:
        return "c3_ppm"
    if re.search(r"\bic4\b", nt) and "ppm" in nt:
        return "ic4_ppm"
    if re.search(r"\bic5\b", nt) and "ppm" in nt:
        return "ic5_ppm"

    if nt == "c1":
        return "c1_ppm"
    if nt == "c2":
        return "c2_ppm"
    if nt == "c3":
        return "c3_ppm"
    if nt == "ic4":
        return "ic4_ppm"
    if nt == "ic5":
        return "ic5_ppm"

    return None


@dataclass
class ColModel:
    x_ranges: Dict[str, Tuple[float, float]]
    header_max_y: float
    row_thresh: int


def _detect_headers_and_xranges(items: List[dict], row_thresh: int) -> Tuple[Dict[str, Tuple[float, float]], float]:
    header_cands = [it for it in items if _is_headerish(it["text"])]
    if not header_cands:
        raise RuntimeError("Header not found")

    header_cands.sort(key=lambda it: it["y1"])
    y0 = header_cands[0]["y1"]
    band = [it for it in header_cands if it["y1"] <= y0 + row_thresh * 4]

    widths = [max(1, int(it.get("w") or 0)) for it in band]
    mw = median_int(widths) or 40
    gap = max(40.0, float(mw) * 0.9)

    clusters = _cluster_by_x(band, gap=gap)

    cols: List[Tuple[str, float, float, float]] = []
    for col in clusters:
        label = _cluster_label(col)
        key = _map_header_label_to_key(label)
        if not key:
            continue
        cx = sum(it["cx"] for it in col) / max(1, len(col))
        x1 = min(it["x1"] for it in col)
        x2 = max(it["x2"] for it in col)
        cols.append((key, cx, x1, x2))

    by_key: Dict[str, Tuple[float, float, float]] = {}
    for k, cx, x1, x2 in cols:
        prev = by_key.get(k)
        if prev is None:
            by_key[k] = (cx, x1, x2)
        else:
            pcx, px1, px2 = prev
            if (x2 - x1) > (px2 - px1) + 2:
                by_key[k] = (cx, x1, x2)

    needed = {"time", "class", "depth_to_top_mmd"}
    if not needed.issubset(set(by_key.keys())):
        raise RuntimeError(f"Header incomplete: {sorted(by_key.keys())}")

    ordered = [(k, by_key[k][0]) for k in CANON_KEYS if k in by_key]
    ordered.sort(key=lambda z: z[1])

    centers = [cx for _, cx in ordered]
    keys = [k for k, _ in ordered]

    PAD = 120.0
    x_ranges: Dict[str, Tuple[float, float]] = {}
    for i, k in enumerate(keys):
        left = (centers[i - 1] + centers[i]) / 2.0 if i > 0 else (centers[i] - PAD)
        right = (centers[i] + centers[i + 1]) / 2.0 if i < len(keys) - 1 else (centers[i] + PAD)
        x_ranges[k] = (left, right)

    header_max_y = max(it["y2"] for it in band)
    return x_ranges, float(header_max_y)


def _assign_col(cm: ColModel, cx: float) -> Optional[str]:
    for k, (lo, hi) in cm.x_ranges.items():
        if lo <= cx < hi:
            return k
    return None

def _band_rows_by_depth_top_mmd(body: List[dict], cm: ColModel) -> List[Tuple[float, float, float]]:
    if "depth_to_top_mmd" not in cm.x_ranges:
        return []
    lo, hi = cm.x_ranges["depth_to_top_mmd"]

    anchors: List[Tuple[float, float]] = []
    for it in body:
        if lo <= it["cx"] < hi:
            v = parse_num(it["text"])
            if v is not None:
                anchors.append((it["y1"], v))

    anchors.sort(key=lambda z: z[0])
    dedup: List[Tuple[float, float]] = []
    for y, v in anchors:
        if not dedup or abs(y - dedup[-1][0]) > cm.row_thresh:
            dedup.append((y, v))

    if not dedup:
        return []

    bands = []
    for i, (y, v) in enumerate(dedup):
        y_next = dedup[i + 1][0] if i < len(dedup) - 1 else 10**9
        bands.append((y, y_next, v))
    return bands

def _join_text(items: List[dict]) -> Optional[str]:
    s = clean(" ".join(clean(it.get("text")) for it in items if clean(it.get("text"))))
    return s or None


def _looks_like_header_leak(v: Any) -> bool:
    nt = norm(v)
    if not nt:
        return False
    return _is_headerish(nt) and (len(nt) <= 20 or "depth" in nt or "highest" in nt or "lowest" in nt)


def _num_or_none(items: List[dict]) -> Optional[float]:
    for it in items:
        v = parse_num(it.get("text"))
        if v is not None:
            return v
    return None


def _key_from_row(r: Dict[str, Any]) -> Optional[str]:
    v = r.get("depth_to_top_mmd")
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return clean(v) or None
    if abs(fv - round(fv)) < 1e-6:
        return str(int(round(fv)))
    s = f"{fv:.3f}".rstrip("0").rstrip(".")
    return s or None


def parse_gas_reading_information_fragment(json_path: str, col_model: Optional[ColModel] = None) -> Tuple[List[Dict[str, Any]], Optional[ColModel]]:
    items, row_thresh = _load_items(json_path)
    items.sort(key=lambda it: (it["y1"], it["x1"]))

    inferred: Optional[ColModel] = None
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

    bands = _band_rows_by_depth_top_mmd(body, col_model)
    if not bands:
        return [], col_model

    out_rows: List[Dict[str, Any]] = []

    for y0, y1, depth_top_mmd in bands:
        band_items = [it for it in body if y0 - 2 <= it["y1"] < y1 - 2]
        col_items: Dict[str, List[dict]] = {k: [] for k in col_model.x_ranges.keys()}

        for it in band_items:
            k = _assign_col(col_model, it["cx"])
            if k:
                col_items[k].append(it)

        for k in col_items:
            col_items[k].sort(key=lambda it: (it["y1"], it["x1"]))

        t = None
        if "time" in col_items:
            for it in col_items["time"]:
                tt = parse_time(it["text"])
                if tt:
                    t = tt
                    break

        cls = None
        if "class" in col_items:
            cls = _join_text(col_items["class"])

        row: Dict[str, Any] = {
            "time": t,
            "class": cls,
            "depth_to_top_mmd": float(depth_top_mmd) if depth_top_mmd is not None else None,
            "depth_to_bottom_md": _num_or_none(col_items.get("depth_to_bottom_md", [])),
            "depth_to_top_mtvd": _num_or_none(col_items.get("depth_to_top_mtvd", [])),
            "depth_to_bottom_tvd": _num_or_none(col_items.get("depth_to_bottom_tvd", [])),
            "highest_gas_percent": _num_or_none(col_items.get("highest_gas_percent", [])),
            "lowest_gas_percent": _num_or_none(col_items.get("lowest_gas_percent", [])),
            "c1_ppm": _num_or_none(col_items.get("c1_ppm", [])),
            "c2_ppm": _num_or_none(col_items.get("c2_ppm", [])),
            "c3_ppm": _num_or_none(col_items.get("c3_ppm", [])),
            "ic4_ppm": _num_or_none(col_items.get("ic4_ppm", [])),
            "ic5_ppm": _num_or_none(col_items.get("ic5_ppm", [])),
        }

        out_rows.append(row)

    return out_rows, col_model


def merge_gas_reading_information_rows(base: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for r in base:
        k = _key_from_row(r)
        if k and k not in by_key:
            by_key[k] = dict(r)
            order.append(k)

    for r in incoming:
        k = _key_from_row(r)
        if not k:
            continue
        if k not in by_key:
            by_key[k] = dict(r)
            order.append(k)
            continue

        b = by_key[k]
        for f in CANON_KEYS:
            nv = r.get(f)
            ov = b.get(f)

            if ov is None and nv is not None:
                b[f] = nv
                continue

            if nv is None:
                continue

            if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                if (ov <= -900 or ov != ov) and (nv > -900 and nv == nv):
                    b[f] = nv
                continue

            if isinstance(ov, str) or isinstance(nv, str):
                ovs = clean(ov)
                nvs = clean(nv)

                if not ovs and nvs:
                    b[f] = nv
                    continue

                if _looks_like_header_leak(ov) and not _looks_like_header_leak(nv):
                    b[f] = nv
                    continue

                if len(nvs) > len(ovs) + 2:
                    b[f] = nv

    return [by_key[k] for k in order]


def extract_gas_reading_information_from_jsons(json_paths: List[str]) -> List[Dict[str, Any]]:
    cm: Optional[ColModel] = None
    merged: List[Dict[str, Any]] = []
    for jp in json_paths:
        rows, cm = parse_gas_reading_information_fragment(jp, col_model=cm)
        if rows:
            merged = merge_gas_reading_information_rows(merged, rows)
    return merged


def _sort_key_from_path(p: Path) -> Tuple[int, int, str]:
    s = str(p).lower()
    pm = re.search(r"page_(\d+)", s)
    tm = re.search(r"table_(\d+)", s)
    page = int(pm.group(1)) if pm else 0
    table = int(tm.group(1)) if tm else 0
    return page, table, str(p)


def parse_gas_reading_information_rows(processed_ddr_dir: Path, document_id: int) -> List[Dict[str, Any]]:
    frags: List[Path] = []
    for page_dir in sorted(processed_ddr_dir.glob("page_*")):
        sec_dir = page_dir / "section_tables" / "gas_reading_information"
        if sec_dir.exists():
            frags.extend(sorted(sec_dir.rglob("*_table_res.json"), key=_sort_key_from_path))

    if not frags:
        return []

    rows = extract_gas_reading_information_from_jsons([str(p) for p in frags])

    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["document_id"] = document_id
        out.append(rr)
    return out