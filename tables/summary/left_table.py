import json
import re
from pathlib import Path
from typing import Dict, Any
from tables.utils import clean, parse_val, normalize_label, split_label_value_row
# Add the parent directory (DDR Processor) to sys.path
FIELD_MAP = {
    "status": "status",
    "report creation time": "report_creation_ts",
    "report number": "report_number",
    "days ahead/behind": "days_ahead_behind",
    "operator": "operator",
    "rig name": "rig_name",
    "drilling contractor": "drilling_contractor",
    "spud date": "spud_ts",
    "wellbore type": "wellbore_type",
    "elevation rkb-msl": "elevation_rkb_msl_m",
    "water depth msl": "water_depth_msl_m",
    "water depth": "water_depth_msl_m",
    "tight well": "tight_well",
    "hpht": "hpht",
    "temperature": "temperature_degc",
    "pressure": "pressure_psig",
    "date well complete": "date_well_complete",
}

FIELDS = list(dict.fromkeys(FIELD_MAP.values()))
def _safe_date_str(s):
    t = str(s or "").strip()
    if not t:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{1,2})", t)
    if not m:
        return s
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if d <= 0:
        d = 1
    return f"{y:04d}-{mo:02d}-{d:02d}"


class SummaryLeftTableParser:
    def extract(self, json_path: Path) -> Dict[str, Any]:
        """Extract table data from PaddleOCR JSON file"""
        
        # Load JSON with correct encoding
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Extract rec_texts and rec_boxes
        rec_texts = data.get('rec_texts', [])
        rec_boxes = data.get('rec_boxes', [])
        
        if not rec_texts or not rec_boxes:
            return {f: None for f in FIELDS}
        
        # Group texts by rows (similar Y coordinates within threshold)
        ROW_THRESHOLD = 15  # pixels - texts within this Y distance are on same row
        
        rows = []
        current_row = []
        current_y = None
        
        # Sort by Y first to group rows
        sorted_by_y = sorted(zip(rec_texts, rec_boxes), key=lambda x: x[1][1])
        
        for text, bbox in sorted_by_y:
            x1, y1, x2, y2 = bbox
            
            if current_y is None or abs(y1 - current_y) <= ROW_THRESHOLD:
                # Same row
                current_row.append((x1, y1, text))
                current_y = y1 if current_y is None else current_y
            else:
                # New row - sort current row by X (left to right) and save
                current_row.sort(key=lambda x: x[0])
                rows.append(current_row)
                current_row = [(x1, y1, text)]
                current_y = y1
        
        # Don't forget last row
        if current_row:
            current_row.sort(key=lambda x: x[0])
            rows.append(current_row)
        
        # Flatten rows into lines
        lines = []
        for row in rows:
            # Concatenate all text in the row with space
            row_text = " ".join(item[2] for item in row)
            row_y = row[0][1]
            lines.append((row_y, row_text))
    
        
        # Parse into fields
        payload = {f: None for f in FIELDS}
        pending = None
        last_date_field = None  # Track last field that got a date value
        
        
        for _, txt in lines:
            txt = clean(txt)
            if not txt:
                continue
            
            
            # Check if this is a standalone time (HH:MM) that should merge with previous date
            # MUST check this FIRST before any other parsing
            if re.fullmatch(r"\d{2}:\d{2}", txt) and last_date_field and payload[last_date_field]:
                # Merge time with the previous date field
                payload[last_date_field] = f"{payload[last_date_field]} {txt}"
                continue
            
            # Attach value to pending field
            if pending and (":" not in txt or re.match(r"^[\d-]", txt)):
                parsed = parse_val(txt)
                payload[pending] = parsed
                # Track if we just set a date value
                if isinstance(parsed, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", str(parsed)):
                    last_date_field = pending
                pending = None
                continue
            
            # Multi-key row: "Drilling contractor: Spud Date:"
            if txt.count(":") >= 2 and txt.endswith(":"):
                keys = re.findall(r"[^:]+:", txt)
                for i, k in enumerate(keys):
                    label = normalize_label(k)
                    field = FIELD_MAP.get(label)
                    if field:
                        payload[field] = None
                        pending = field if i == len(keys) - 1 else None
                        last_date_field = None  # Reset date tracking
                continue
            
            # Single "Key: Value" or "Key:"
            if ":" in txt or not pending:  # Check for colon OR if we're not waiting for a value
                label_txt, value_txt = split_label_value_row(txt)
                
                if label_txt and label_txt != txt:  # We got a label (not just the raw text)
                    label = normalize_label(label_txt)
                    field = FIELD_MAP.get(label)
                    
                    if field:
                        # Check if value is actually another key
                        if value_txt.endswith(":"):
                            payload[field] = None
                            next_label = normalize_label(value_txt)
                            next_field = FIELD_MAP.get(next_label)
                            if next_field:
                                payload[next_field] = None
                                pending = next_field
                            last_date_field = None  # Reset date tracking
                        else:
                            parsed = parse_val(value_txt) if value_txt else None
                            payload[field] = parsed
                            pending = field if not value_txt else None
                            # Track if we just set a date value
                            if isinstance(parsed, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", str(parsed)):
                                last_date_field = field
                            else:
                                last_date_field = None

        payload["date_well_complete"] = _safe_date_str(payload.get("date_well_complete"))

        sp = str(payload.get("spud_ts") or "").strip()
        if sp:
            d, *rest = sp.split(" ", 1)
            sd = _safe_date_str(d)
            payload["spud_ts"] = f"{sd} {rest[0]}" if rest and sd else sd

        rc = str(payload.get("report_creation_ts") or "").strip()
        if rc:
            d, *rest = rc.split(" ", 1)
            rd = _safe_date_str(d)
            payload["report_creation_ts"] = f"{rd} {rest[0]}" if rest and rd else rd
        return payload
