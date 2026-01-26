from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import json
from tables.utils  import clean, parse_val, group_rows, split_label_value_row

TABLE_3_FIELDS = [
    "depth_kickoff_mmd",
    "depth_kickoff_mtvd",
    "depth_mmd",
    "depth_mtvd",
    "plug_back_depth_mmd",
    "depth_formation_strength_mmd",
    "depth_formation_strength_mtvd",
    "depth_last_casing_mmd",
    "depth_last_casing_mtvd",
]

FIELD_MAP = {
    "depth at kick off mmd": "depth_kickoff_mmd",
    "depth at kick off mtvd": "depth_kickoff_mtvd",
    "depth mmd": "depth_mmd",
    "depth mtvd": "depth_mtvd",
    "plug back depth mmd": "plug_back_depth_mmd",
    "depth at formation strength mmd": "depth_formation_strength_mmd",
    "depth at formation strength mtvd": "depth_formation_strength_mtvd",
    "depth at last casing mmd": "depth_last_casing_mmd",
    "depth at last casing mtvd": "depth_last_casing_mtvd",
}

def normalize_label(s: str) -> str:
    s = clean(s).replace("：", ":")
    s = re.sub(r"\([^)]*\)", "", s)   
    s = s.rstrip(":").strip().lower()

     
    s = re.sub(r"\bm\s*md\b", "mmd", s)      
    s = re.sub(r"\bmmd\b", "mmd", s)        
    s = re.sub(r"\bm\s*tvd\b", "mtvd", s)   
    s = re.sub(r"\bmtvd\b", "mtvd", s)

     
    s = s.replace("kick off", "kick off")   
    return s

def match_field(norm_label: str) -> str | None:
    """
    Robust match:
      1) exact match on FIELD_MAP keys
      2) containment match (helps OCR noise like extra spaces / casing)
      3) a few tolerant patterns (kick off / formation strength / last casing)
    """
    if norm_label in FIELD_MAP:
        return FIELD_MAP[norm_label]

     
    for k, v in FIELD_MAP.items():
        if k in norm_label:
            return v

     
    if "depth" in norm_label and "kick" in norm_label and "off" in norm_label and "mmd" in norm_label:
        return "depth_kickoff_mmd"
    if "depth" in norm_label and "kick" in norm_label and "off" in norm_label and "mtvd" in norm_label:
        return "depth_kickoff_mtvd"

    if "plug" in norm_label and "back" in norm_label and "depth" in norm_label and "mmd" in norm_label:
        return "plug_back_depth_mmd"

    if "formation" in norm_label and "strength" in norm_label and "mmd" in norm_label:
        return "depth_formation_strength_mmd"
    if "formation" in norm_label and "strength" in norm_label and "mtvd" in norm_label:
        return "depth_formation_strength_mtvd"

    if "last" in norm_label and "casing" in norm_label and "mmd" in norm_label:
        return "depth_last_casing_mmd"
    if "last" in norm_label and "casing" in norm_label and "mtvd" in norm_label:
        return "depth_last_casing_mtvd"

     
    if norm_label.startswith("depth") and "mmd" in norm_label and "kick" not in norm_label and "formation" not in norm_label and "last" not in norm_label:
        return "depth_mmd"
    if norm_label.startswith("depth") and "mtvd" in norm_label and "kick" not in norm_label and "formation" not in norm_label and "last" not in norm_label:
        return "depth_mtvd"

    return None

class SummaryRightTableParser:
    def extract(self,
            json_path: str,
            row_threshold: int = 18,
            debug: bool = False,
        ) -> Dict[str, Any]:
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
            lines.append((y, clean(" ".join(t for _, t in parts))))

        if debug:
            print("=" * 80)
            print("RECONSTRUCTED ROWS:")
            print("=" * 80)
            for y, s in sorted(lines, key=lambda z: z[0]):
                print(f"y={y:4d} | {s}")

        payload: Dict[str, Any] = {k: None for k in TABLE_3_FIELDS}
        pending_field: str | None = None

        if debug:
            print("\n" + "=" * 80)
            print("PARSING:")
            print("=" * 80)

        for _, txt in sorted(lines, key=lambda z: z[0]):
            if not txt:
                continue

            label_txt, value_txt = split_label_value_row(txt)

             
            if pending_field and (":" not in txt):
                payload[pending_field] = parse_val(txt)
                if debug:
                    print(f"✓ attached pending {pending_field} <- {payload[pending_field]!r}   (from '{txt}')")
                pending_field = None
                continue

            if ":" in label_txt:
                norm = normalize_label(label_txt)
                field = match_field(norm)

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

        return payload