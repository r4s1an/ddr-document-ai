# # Initialize PaddleOCR instance
# from paddleocr import PaddleOCR
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)

# # Run OCR inference on a sample image 
# result = ocr.predict(
#     input=r"testing\page_001\crops\009_table.png")

# # Visualize the results and save the JSON results
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

import json
import re
from pathlib import Path
from typing import Any, Dict
from pprint import pprint

# Field mapping
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


def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("\n", " ")).strip()


def normalize_label(s: str) -> str:
    return re.sub(r"\([^)]*\)", "", clean(s).rstrip(":")).strip().lower()


def parse_val(v: str):
    v = clean(v)
    if not v:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}(\s+\d{2}:\d{2})?$", v):
        return v
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return float(v) if "." in v else int(v)
    m = re.search(r"-?\d+(\.\d+)?", v)
    return (float(m[0]) if "." in m[0] else int(m[0])) if m else v


def extract_from_paddleocr_json(json_path: str) -> Dict[str, Any]:
    """Extract table data from PaddleOCR JSON file"""
    
    # Load JSON with correct encoding
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    # Extract rec_texts and rec_boxes
    rec_texts = data.get('rec_texts', [])
    rec_boxes = data.get('rec_boxes', [])
    
    if not rec_texts or not rec_boxes:
        print("Error: No rec_texts or rec_boxes found in JSON")
        return {f: None for f in FIELDS}
    
    print("="*60)
    print("RAW OCR OUTPUT FROM JSON:")
    print("="*60)
    for i, (text, bbox) in enumerate(zip(rec_texts, rec_boxes)):
        x1, y1, x2, y2 = bbox
        print(f"{i:2d}. x={x1:3d} y={y1:3d} | {text}")
    
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
    
    print("\n" + "="*60)
    print("RECONSTRUCTED ROWS (left-to-right within each row):")
    print("="*60)
    for y, text in lines:
        print(f"y={y:3d} | {text}")
    
    # Parse into fields
    payload = {f: None for f in FIELDS}
    pending = None
    last_date_field = None  # Track last field that got a date value
    
    print("\n" + "="*60)
    print("PARSING INTO FIELDS:")
    print("="*60)
    
    for _, txt in lines:
        txt = clean(txt)
        if not txt:
            continue
        
        print(f"\nProcessing: '{txt}'")
        
        # Check if this is a standalone time (HH:MM) that should merge with previous date
        # MUST check this FIRST before any other parsing
        if re.fullmatch(r"\d{2}:\d{2}", txt) and last_date_field and payload[last_date_field]:
            # Merge time with the previous date field
            payload[last_date_field] = f"{payload[last_date_field]} {txt}"
            print(f"  ✓ Merged time with '{last_date_field}': {payload[last_date_field]}")
            continue
        
        # Attach value to pending field
        if pending and (":" not in txt or re.match(r"^[\d-]", txt)):
            parsed = parse_val(txt)
            payload[pending] = parsed
            # Track if we just set a date value
            if isinstance(parsed, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", str(parsed)):
                last_date_field = pending
            print(f"  ✓ Attached to '{pending}': {payload[pending]}")
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
            print(f"  ✓ Multi-key, pending: {pending}")
            continue
        
        # Single "Key: Value" or "Key:"
        if ":" in txt:
            parts = txt.split(":", 1)
            label = normalize_label(parts[0])
            field = FIELD_MAP.get(label)
            
            if field:
                val = parts[1].strip() if len(parts) > 1 else ""
                # Check if value is actually another key
                if val.endswith(":"):
                    payload[field] = None
                    next_label = normalize_label(val)
                    next_field = FIELD_MAP.get(next_label)
                    if next_field:
                        payload[next_field] = None
                        pending = next_field
                    last_date_field = None  # Reset date tracking
                    print(f"  ✓ '{field}' = None, next pending: {pending}")
                else:
                    parsed = parse_val(val) if val else None
                    payload[field] = parsed
                    pending = field if not val else None
                    # Track if we just set a date value
                    if isinstance(parsed, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", str(parsed)):
                        last_date_field = field
                    else:
                        last_date_field = None
                    print(f"  ✓ '{field}' = {payload[field]}, pending: {pending}")
            else:
                print(f"  ✗ Unknown field label: '{label}'")
    
    return payload


if __name__ == "__main__":
    json_path = "output/009_table_res.json"
    
    if not Path(json_path).exists():
        print(f"Error: JSON file not found at {json_path}")
        print("Please run PaddleOCR first to generate the JSON file.")
        exit(1)
    
    print(f"Reading JSON from: {json_path}\n")
    
    # Extract payload from JSON file
    payload = extract_from_paddleocr_json(json_path)
    
    print("\n" + "="*60)
    print("FINAL EXTRACTED PAYLOAD:")
    print("="*60)
    pprint(payload)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    filled = sum(1 for v in payload.values() if v is not None)
    print(f"Filled fields: {filled}/{len(FIELDS)}")
    print(f"Empty fields: {len(FIELDS) - filled}/{len(FIELDS)}")