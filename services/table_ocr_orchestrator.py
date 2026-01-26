from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from tables.paddle_runner import run_paddle_ocr

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "unknown_section"
    return s[:max_len]

def _get_items(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return doc["items"] if isinstance(doc, dict) and "items" in doc else doc

def _ocr_by_idx(layout_ocr: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    by = {}
    for it in _get_items(layout_ocr):
        if "idx" in it:
            by[int(it["idx"])] = it
    return by

def _header_text_for_table(table_item: Dict[str, Any], ocr_items_by_idx: Dict[int, Dict[str, Any]]) -> str:
    hdr_idx = table_item.get("linked_header_idx")
    if hdr_idx is None:
        return ""
    hdr = ocr_items_by_idx.get(int(hdr_idx), {})
    return (hdr.get("ocr_text") or "").strip()


def _sort_tables_left_mid_right(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(tables, key=lambda t: float(t.get("x1", 0.0)))

def _last_header_slug_on_page(layout_ocr: Dict[str, Any]) -> str | None:
    items = _get_items(layout_ocr)
    headers = [it for it in items if it.get("label") == "section_header"]

     
    headers = sorted(headers, key=lambda it: (float(it.get("y1", 0.0)), float(it.get("x1", 0.0))))

     
    for h in reversed(headers):
        txt = (h.get("ocr_text") or "").strip()
        if txt:
            return _slugify(txt)
    return None

def _header_text_for_idx(hdr_idx: int | None, ocr_items_by_idx: Dict[int, Dict[str, Any]]) -> str:
    if hdr_idx is None:
        return ""
    hdr = ocr_items_by_idx.get(int(hdr_idx), {})
    return (hdr.get("ocr_text") or "").strip()

def run_paddle_for_all_tables(
    processed_ddr_dir: str | Path,
    *,
    debug: bool = False,
    skip_if_exists: bool = True,
) -> Dict[str, Any]:
    """
    For every page_XXX/layout.json, find all label='table' items, route them into:
      page_XXX/section_tables/<section_slug>/<table_key>/
    and run Paddle OCR saving JSON there.

    Summary report special case: if section_slug == 'summary_report' and 3 tables exist on that page,
    store into summary_tables/{left|middle|right}/ to keep compatibility with your current pipeline.

    Returns a debug summary (counts + outputs).
    """
    processed_ddr_dir = Path(processed_ddr_dir)
    pages = sorted([p for p in processed_ddr_dir.iterdir() if p.is_dir() and p.name.startswith("page_")])

    debug_out: Dict[str, Any] = {
        "pages": [],
        "tables_total": 0,
        "tables_ocr_ran": 0,
        "tables_skipped": 0,
        "warnings": [],
    }
    prev_page_last_slug: str | None = None
    TOP_CONTINUATION_Y = 220
    for page_dir in pages:
        layout_path = page_dir / "layout.json"
        ocr_path = page_dir / "layout_ocr.json"
        crops_dir = page_dir / "crops"

        if not layout_path.exists() or not ocr_path.exists() or not crops_dir.exists():
            continue

        layout = _load_json(layout_path)
        layout_ocr = _load_json(ocr_path)

        layout_items = _get_items(layout)
        ocr_items_by_idx = _ocr_by_idx(layout_ocr)

         
        tables = [it for it in layout_items if it.get("label") == "table"]
        debug_out["tables_total"] += len(tables)

        by_section: Dict[str, List[Dict[str, Any]]] = {}
         
        tables_sorted = sorted(tables, key=lambda it: (float(it.get("y1", 0.0)), float(it.get("x1", 0.0))))
        first_table_idx = int(tables_sorted[0]["idx"]) if tables_sorted else None

        for t in tables:
            header_text = _header_text_for_table(t, ocr_items_by_idx)
            slug = _slugify(header_text)

             
            if slug == "unknown_section" and prev_page_last_slug:
                 
                hdr_text = _header_text_for_idx(t.get("linked_header_idx"), ocr_items_by_idx)

                is_first_table = (first_table_idx is not None and int(t.get("idx", -1)) == first_table_idx)

                 
                if is_first_table or not hdr_text:
                    slug = prev_page_last_slug

            by_section.setdefault(slug, []).append(t)

        page_debug = {"page": page_dir.name, "sections": {}}

        for section_slug, section_tables in by_section.items():
             
            if section_slug == "summary_report" and len(section_tables) == 3:
                 
                section_tables_sorted = _sort_tables_left_mid_right(section_tables)
                roles = ["left", "middle", "right"]
                base_dir = page_dir / "summary_tables"

                for role, t in zip(roles, section_tables_sorted):
                    crop_path = Path(t.get("crop_path", ""))
                    if not crop_path:
                         
                        debug_out["warnings"].append({"page": page_dir.name, "reason": "table missing crop_path"})
                        continue

                    out_dir = base_dir / role

                     
                    if skip_if_exists and any(out_dir.glob("*.json")):
                        debug_out["tables_skipped"] += 1
                        continue

                    run_paddle_ocr(crop_path, out_dir)
                    debug_out["tables_ocr_ran"] += 1

                page_debug["sections"][section_slug] = {
                    "tables": len(section_tables),
                    "mode": "summary_tables(left/middle/right)",
                }
                continue
             
            sec_base = page_dir / "section_tables" / section_slug
            sec_base.mkdir(parents=True, exist_ok=True)

            for t in section_tables:
                crop_path = Path(t.get("crop_path", ""))
                if not crop_path:
                    debug_out["warnings"].append({"page": page_dir.name, "reason": "table missing crop_path"})
                    continue
                 
                table_idx = t.get("idx")
                table_key = f"table_{int(table_idx):03d}" if table_idx is not None else crop_path.stem
                out_dir = sec_base / table_key

                if skip_if_exists and any(out_dir.glob("*.json")):
                    debug_out["tables_skipped"] += 1
                    continue

                run_paddle_ocr(crop_path, out_dir)
                debug_out["tables_ocr_ran"] += 1

            page_debug["sections"][section_slug] = {
                "tables": len(section_tables),
                "mode": "section_tables/<slug>/table_xxx/",
            }

        debug_out["pages"].append(page_debug)

        last_slug = _last_header_slug_on_page(layout_ocr)
        if last_slug:
            prev_page_last_slug = last_slug

        if debug:
            print(f"[{page_dir.name}] sections: {list(by_section.keys())}")

    return debug_out
