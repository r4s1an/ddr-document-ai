from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RE_TIME = re.compile(r"^\s*(\d{1,2})\s*[:.]\s*(\d{2})\s*$")
RE_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

TARGET_FIELDS = [
    "sample_point",
    "sample_depth_mmd",
    "fluid_type",
    "fluid_density_g_cm3",
    "funnel_visc_s",
    "plastic_visc_mpas",
    "yield_point_pa",
    "test_temp_hpht_degc",
]

LABEL_MAP = {
    "sample point": "sample_point",
    "sample depth mmd": "sample_depth_mmd",
    "fluid type": "fluid_type",
    "fluid density": "fluid_density_g_cm3",
    "funnel visc": "funnel_visc_s",
    "plastic visc": "plastic_visc_mpas",
    "yield point": "yield_point_pa",
    "test temp hpht": "test_temp_hpht_degc",
}

ROW_HEADERS_TO_IGNORE = {
    "solids",
    "viscometer tests",
    "filtration tests",
    "comment",
}


@dataclass(frozen=True)
class OCRCell:
    text: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass
class ColumnModel:
    sample_times: List[str]
    col_x: List[float]
    label_right_x: float


def parse_drilling_fluid_rows(processed_ddr_dir: Path, document_id: int) -> List[Dict[str, Any]]:
    fragments = _find_drilling_fluid_fragments(processed_ddr_dir)
    if not fragments:
        return []

    col_model: Optional[ColumnModel] = None
    merged_by_time: Dict[str, Dict[str, Any]] = {}

    for frag_path in fragments:
        frag = _parse_fragment(frag_path, col_model)
        if frag["col_model"] is not None and col_model is None:
            col_model = frag["col_model"]
        merged_by_time = _merge_rows(merged_by_time, frag["rows_by_time"])

    if not merged_by_time:
        return []

    out: List[Dict[str, Any]] = []
    for t in sorted(merged_by_time.keys(), key=_time_sort_key):
        row = merged_by_time[t]
        row["document_id"] = document_id
        row["sample_time"] = t
        for f in TARGET_FIELDS:
            row.setdefault(f, None)
        out.append(row)

    return out


def _find_drilling_fluid_fragments(root: Path) -> List[Path]:
    out: List[Path] = []
    for page_dir in sorted(root.glob("page_*")):
        df_dir = page_dir / "section_tables" / "drilling_fluid"
        if df_dir.exists():
            out.extend(sorted(df_dir.rglob("*_table_res.json")))
    return out


def _parse_fragment(table_res_json_path: Path, col_model: Optional[ColumnModel]) -> Dict[str, Any]:
    cells = _load_paddle_cells(table_res_json_path)
    if not cells:
        return {"rows_by_time": {}, "col_model": col_model}

    inferred = _infer_columns_from_sample_time_row(cells)
    if inferred is not None:
        col_model = inferred
    else:
        if col_model is None:
            return {"rows_by_time": {}, "col_model": None}

    assert col_model is not None
    rows_by_time: Dict[str, Dict[str, Any]] = {t: {} for t in col_model.sample_times}

    for label_norm, field in LABEL_MAP.items():
        if label_norm in ROW_HEADERS_TO_IGNORE:
            continue

        y = _find_row_y(cells, label_norm, col_model)
        if y is None:
            continue

        vals = _extract_values_for_row(cells, row_y=y, col_model=col_model)
        for t, raw_val in vals.items():
            val = _normalize_value(field, raw_val)
            if _looks_like_header_hour(field, val, t):
                continue
            if not _is_plausible(field, val):
                continue
            rows_by_time[t][field] = val

    return {"rows_by_time": rows_by_time, "col_model": col_model}


def _merge_rows(base: Dict[str, Dict[str, Any]], incoming: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    for t in incoming.keys():
        base.setdefault(t, {})

    for t, inc_row in incoming.items():
        b = base.setdefault(t, {})
        for k, inc_val in inc_row.items():
            if inc_val is None or inc_val == "":
                continue

            if k not in b or b[k] is None or b[k] == "":
                b[k] = inc_val
                continue

            if str(b[k]).strip() != str(inc_val).strip():
                existing = b[k]
                incoming_val = inc_val

                existing_ok = _is_plausible(k, existing) and not _looks_like_header_hour(k, existing, t)
                incoming_ok = _is_plausible(k, incoming_val) and not _looks_like_header_hour(k, incoming_val, t)

                if (not existing_ok) and incoming_ok:
                    b[k] = incoming_val

    return base


def _infer_columns_from_sample_time_row(cells: List[OCRCell]) -> Optional[ColumnModel]:
    label_cell = _find_best_label_cell(cells, "sample time", col_model=None)
    if label_cell is None:
        return None

    y = label_cell.cy
    cand: List[Tuple[float, str]] = []
    for c in cells:
        if c.cx <= label_cell.x2 + 5:
            continue
        if abs(c.cy - y) <= max(12, label_cell.h * 0.8):
            t = _normalize_time(c.text)
            if t:
                cand.append((c.cx, t))

    if not cand:
        return None

    cand.sort(key=lambda x: x[0])
    times = [t for _, t in cand]
    col_x = [x for x, _ in cand]

    return ColumnModel(sample_times=times, col_x=col_x, label_right_x=label_cell.x2)


def _find_row_y(cells: List[OCRCell], label_norm: str, col_model: ColumnModel) -> Optional[float]:
    c = _find_best_label_cell(cells, label_norm, col_model=col_model)
    return None if c is None else c.cy


def _extract_values_for_row(cells: List[OCRCell], row_y: float, col_model: ColumnModel) -> Dict[str, str]:
    out: Dict[str, str] = {}
    y_tol = 18
    x_tol = 90

    row_cands = [
        c for c in cells
        if c.cx > (col_model.label_right_x + 8) and abs(c.cy - row_y) <= y_tol
    ]
    if not row_cands:
        return out

    for t, cx in zip(col_model.sample_times, col_model.col_x):
        best: Optional[OCRCell] = None
        best_dx = 1e18
        for c in row_cands:
            dx = abs(c.cx - cx)
            if dx <= x_tol and dx < best_dx:
                best = c
                best_dx = dx
        if best is not None:
            val = best.text.strip()
            if val != "":
                out[t] = val

    return out


def _find_best_label_cell(
    cells: List[OCRCell],
    target_label: str,
    col_model: Optional[ColumnModel],
) -> Optional[OCRCell]:
    target = _normalize_text(target_label)

     
    if col_model is not None and col_model.col_x:
        label_region_right = min(col_model.col_x) - 10
    else:
        xs = sorted([c.cx for c in cells])
        if not xs:
            return None
        label_region_right = xs[int(len(xs) * 0.50)]

    for c in sorted(cells, key=lambda x: (x.cy, x.cx)):
        if c.cx > label_region_right:
            continue

        txt = _normalize_text(c.text)
        txt = re.sub(r"\([^)]*\)", "", txt)
        txt = txt.replace("()", "").strip()

        if target in txt:
            return c

    return None


def _load_paddle_cells(path: Path) -> List[OCRCell]:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    if isinstance(data, dict) and "rec_texts" in data:
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores")
        boxes = data.get("dt_boxes") or data.get("rec_boxes") or data.get("boxes") or data.get("rec_polys")
        if not boxes:
            return []
        n = min(len(texts), len(boxes))
        out: List[OCRCell] = []
        for i in range(n):
            text = str(texts[i]).strip()
            conf = float(scores[i]) if (scores and i < len(scores)) else 0.0
            x1, y1, x2, y2 = _box_to_xyxy(boxes[i])
            if x1 is None:
                continue
            out.append(OCRCell(text=text, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))
        return out

    raw = _unwrap_paddle_result(data)
    out: List[OCRCell] = []
    for item in raw:
        try:
            box, tc = item[0], item[1]
            text = str(tc[0]).strip()
            conf = float(tc[1]) if len(tc) > 1 else 0.0
            x1, y1, x2, y2 = _box_to_xyxy(box)
            if x1 is None:
                continue
            out.append(OCRCell(text=text, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))
        except Exception:
            continue
    return out


def _box_to_xyxy(box: Any) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        if isinstance(box, list) and box and isinstance(box[0], (list, tuple)) and len(box[0]) == 2:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            return min(xs), min(ys), max(xs), max(ys)

        if isinstance(box, list) and len(box) >= 4 and all(isinstance(v, (int, float)) for v in box[:4]):
            vals = list(map(float, box))
            xs = vals[0::2]
            ys = vals[1::2]
            return min(xs), min(ys), max(xs), max(ys)

        if hasattr(box, "tolist"):
            return _box_to_xyxy(box.tolist())
    except Exception:
        pass

    return None, None, None, None


def _unwrap_paddle_result(data: Any) -> List[Any]:
    if isinstance(data, list):
        if data and isinstance(data[0], list) and len(data[0]) == 2:
            return data
        out: List[Any] = []
        for part in data:
            if isinstance(part, list):
                out.extend(part)
        return out

    if isinstance(data, dict):
        for k in ("ocr_result", "result", "data", "items", "boxes"):
            if k in data:
                return _unwrap_paddle_result(data[k])
        for k in ("pages", "page_results"):
            if k in data:
                return _unwrap_paddle_result(data[k])

    return []


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(".", " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_time(s: str) -> Optional[str]:
    m = RE_TIME.match(s.strip())
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}"


def _time_sort_key(t: str) -> int:
    m = RE_TIME.match(t)
    if not m:
        return 10**9
    return int(m.group(1)) * 60 + int(m.group(2))


def _normalize_value(field: str, raw: str) -> Any:
    s = raw.strip()
    if field in ("sample_point", "fluid_type"):
        return s

    m = RE_NUM.search(s)
    if not m:
        return s

    try:
        return float(m.group(0))
    except Exception:
        return s


def _hour_from_sample_time(sample_time: str) -> Optional[int]:
    m = RE_TIME.match(sample_time.strip())
    if not m:
        return None
    return int(m.group(1))


def _looks_like_header_hour(field: str, value: Any, sample_time: str) -> bool:
    if field in ("sample_point", "fluid_type"):
        return False

    hr = _hour_from_sample_time(sample_time)
    if hr is None:
        return False

    try:
        v = float(value)
    except Exception:
        return False

    return abs(v - float(hr)) < 1e-6


def _is_plausible(field: str, value: Any) -> bool:
    if value is None or value == "":
        return False

    if field in ("sample_point", "fluid_type"):
        return True

    try:
        v = float(value)
    except Exception:
        return False

     
    if abs(v + 999.99) < 1e-6:
        return True

    if field == "plastic_visc_mpas":
        return 0 <= v <= 200
    if field == "yield_point_pa":
        return 0 <= v <= 300
    if field == "test_temp_hpht_degc":
        return 0 <= v <= 300
    if field == "fluid_density_g_cm3":
        return 0.5 <= v <= 3.0
    if field == "funnel_visc_s":
        return -2000 <= v <= 2000
    if field == "sample_depth_mmd":
        return 0 <= v <= 20000

    return True
