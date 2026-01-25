import os
import torch
from pathlib import Path
import json
import logging
import paddle_ocr_playing.paddle_json_builder as paddle_json_builder

# 1. SETUP ENV FLAGS
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# 2. FORCE DISABLE MKLDNN VIA PADDLE API
paddle_json_builder.set_flags({'FLAGS_use_mkldnn': False})

from paddleocr import PaddleOCR, PPStructure

# -------- LOGGING SETUP --------
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddlex").setLevel(logging.WARNING)

# -------- CONFIG --------
CROPS_DIR = Path("out_test/page_001_crops")
OUT_DIR = Path("out_test/paddleocr_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- INITIALIZE ENGINES --------
print("Initializing Text OCR Engine...")
# Standard PaddleOCR for text
text_ocr = PaddleOCR(
    use_angle_cls=True,  # Replaces use_textline_orientation
    lang="en",
    enable_mkldnn=False,
    show_log=False
)

print("Initializing Table OCR Engine...")
# PPStructure for tables (Note: PPStructureV3 is usually accessed via PPStructure(version='PP-StructureV2') or similar)
# We use the generic PPStructure class which handles layout/table analysis
table_ocr = PPStructure(
    show_log=False,
    image_orientation=True,
    enable_mkldnn=False,
    layout=False,      # We only want table recognition, not full layout analysis
    table=True
)

# -------- HELPERS --------

def run_text_ocr(image_path: Path):
    """
    Runs text detection and recognition.
    PaddleOCR.ocr returns: [ [ [[x,y],..], (text, score) ], ... ]
    """
    img_str = str(image_path)
    
    # .ocr() returns a list. If input is a single image, it returns [result_list]
    # result_list contains: [box, (text, confidence)]
    try:
        result = text_ocr.ocr(img_str, cls=True)
    except Exception as e:
        print(f"PaddleOCR Runtime Error: {e}")
        return []

    texts = []

    # Check if result is empty or None
    if not result or result[0] is None:
        return []

    # The result for a single image is usually result[0]
    lines = result[0] 
    
    for line in lines:
        # line structure: [ [box_coords], (text, score) ]
        if len(line) >= 2:
            text_info = line[1] # (text, score)
            if isinstance(text_info, tuple) or isinstance(text_info, list):
                texts.append({
                    "text": text_info[0],
                    "confidence": float(text_info[1])
                })

    return texts

def run_table_ocr(image_path: Path):
    """
    Runs table structure recognition.
    PPStructure returns a list of dictionaries.
    """
    img_str = str(image_path)
    
    try:
        # PPStructure is callable directly. It returns a list.
        # DO NOT use next() here.
        result = table_ocr(img_str)
    except Exception as e:
        print(f"Table OCR Runtime Error: {e}")
        return []

    tables = []

    if not result:
        return []

    # result is a list of dicts. 
    # Example item: {'type': 'table', 'bbox': [..], 'img': array, 'res': {'html': '<table>...'}}
    # Or sometimes directly: {'html': '...'} depending on exact version/flags
    
    for item in result:
        html_content = None
        score = 0.0

        # Check nested 'res' structure (common in structure V2/V3)
        if 'res' in item and isinstance(item['res'], dict):
            html_content = item['res'].get('html')
            score = item['res'].get('score', 0.0)
        # Check direct keys
        elif 'html' in item:
            html_content = item['html']
            score = item.get('score', 0.0)

        if html_content:
            tables.append({
                "html": html_content,
                "confidence": float(score)
            })

    return tables

# -------- MAIN LOOP --------
def main():
    if not CROPS_DIR.exists():
        print(f"Error: Directory {CROPS_DIR} not found.")
        return

    print(f"Scanning {CROPS_DIR}...")
    
    for img_path in sorted(CROPS_DIR.glob("*.png")):
        name = img_path.name.lower()
        stem = img_path.stem

        print(f"Processing: {img_path.name}")

        try:
            if "table" in name:
                result = run_table_ocr(img_path)
                out_file = OUT_DIR / f"{stem}_table.json"
            else:
                result = run_text_ocr(img_path)
                out_file = OUT_DIR / f"{stem}_text.json"

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  Saved -> {out_file}")

        except Exception as e:
            print(f"  Error processing {name}: {e}")
            # import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()