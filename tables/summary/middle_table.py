from pathlib import Path
from typing import Dict, Any
from tables.utils import clean, normalize_label, parse_val, split_label_value_row
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


TABLE_2_FIELDS = [
    "dist_drilled_m",
    "penetration_rate_mph",
    "hole_dia_in",
    "pressure_test_type",
    "formation_strength_g_cm3",
    "dia_last_casing",
]

FIELD_MAP = {
    "dist drilled": "dist_drilled_m",
    "penetration rate": "penetration_rate_mph",
    "hole dia": "hole_dia_in",
    "pressure test type": "pressure_test_type",
    "formation strength": "formation_strength_g_cm3",
    "dia last casing": "dia_last_casing",
}


def group_rows(
    rec_texts: List[str],
    rec_boxes: List[List[int]],
    row_threshold: int = 18,
) -> List[Tuple[int, List[Tuple[int, str]]]]:
    """
    Group OCR items into rows by y1 proximity.
    Returns: list of (row_y, [(x1, text), ...]) sorted left->right per row.
    """
    items = []
    for t, b in zip(rec_texts, rec_boxes):
        if not t or not b or len(b) < 4:
            continue
        x1, y1, x2, y2 = map(int, b[:4])
        items.append((y1, x1, clean(t)))

    items.sort(key=lambda z: (z[0], z[1]))

    rows: List[Tuple[int, List[Tuple[int, str]]]] = []
    cur: List[Tuple[int, str]] = []
    cur_y = None

    for y1, x1, txt in items:
        if not txt:
            continue

        if cur_y is None or abs(y1 - cur_y) <= row_threshold:
            cur.append((x1, txt))
            if cur_y is None:
                cur_y = y1
        else:
            cur.sort(key=lambda z: z[0])
            rows.append((cur_y, cur))
            cur = [(x1, txt)]
            cur_y = y1

    if cur:
        cur.sort(key=lambda z: z[0])
        rows.append((cur_y if cur_y is not None else 0, cur))

    return rows

class SummaryMiddleTableParser:
    def extract(self, json_path: Path,
        row_threshold: int = 18,
        debug: bool = False) -> Dict[str, Any]:
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")

        with open(p, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        rec_texts = data.get("rec_texts", []) or []
        rec_boxes = data.get("rec_boxes", []) or []

        if not rec_texts or not rec_boxes:
            raise ValueError("JSON has no rec_texts/rec_boxes (not a PaddleOCR table JSON?)")

        rows = group_rows(rec_texts, rec_boxes, row_threshold=row_threshold)

         
        lines: List[Tuple[int, str]] = []
        for y, parts in rows:
            row_str = " ".join(t for _, t in parts)
            lines.append((y, clean(row_str)))

        if debug:
            print("=" * 70)
            print("RECONSTRUCTED ROWS:")
            print("=" * 70)
            for y, s in lines:
                print(f"y={y:4d} | {s}")

        payload: Dict[str, Any] = {k: None for k in TABLE_2_FIELDS}
        pending_field = None   

        if debug:
            print("\n" + "=" * 70)
            print("PARSING:")
            print("=" * 70)

        for _, txt in sorted(lines, key=lambda z: z[0]):
            if not txt:
                continue

            label_txt, value_txt = split_label_value_row(txt)

             
            if pending_field and value_txt == "" and ":" not in txt:
                payload[pending_field] = parse_val(txt)
                if debug:
                    print(f"✓ attached pending {pending_field} <- {payload[pending_field]!r}   (from '{txt}')")
                pending_field = None
                continue

             
            if ":" in label_txt:
                norm = normalize_label(label_txt)

                field = None
                 
                for k, v in FIELD_MAP.items():
                    if k in norm:
                        field = v
                        break

                if not field:
                    if debug:
                        print(f"· skip unknown label: '{label_txt}' (norm='{norm}')")
                    continue

                if value_txt:
                    payload[field] = parse_val(value_txt)
                    pending_field = None
                    if debug:
                        print(f"✓ {field} <- {payload[field]!r}   (from '{txt}')")
                else:
                     
                    payload[field] = None
                    pending_field = field
                    if debug:
                        print(f"✓ {field} pending value (saw '{label_txt}')")
            else:
                 
                if pending_field:
                    payload[pending_field] = parse_val(txt)
                    if debug:
                        print(f"✓ attached pending {pending_field} <- {payload[pending_field]!r}   (from '{txt}')")
                    pending_field = None

        return payload