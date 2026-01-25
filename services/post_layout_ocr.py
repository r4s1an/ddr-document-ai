from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from services.ocr_easy import EasyOCRService

def ocr_layout_json(page_dir: Path, gpu: bool = False) -> Dict[str, Any]:
    layout_path = page_dir / "layout.json"
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    items = data["items"]

    ocr = EasyOCRService(gpu=gpu)
    results = ocr.read_items(
        items,
        labels=["section_header", "plain_text", "wellbore_field", "period_field"]
    )

    for it in items:
        r = results.get(it["idx"])
        if r:
            it["ocr_text"] = r["text"]
            it["ocr_conf"] = r["conf"]

    out_path = page_dir / "layout_ocr.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data