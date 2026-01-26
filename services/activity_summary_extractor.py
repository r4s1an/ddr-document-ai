import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


 
TARGET_HEADERS = {
    "activities_24h_text": "summary of activities 24 hours",
    "planned_24h_text": "summary of planned activities 24 hours",
}

def _clean_plain_text(s: str | None) -> str | None:
    if not s:
        return None

    s = s.strip()

    if s.lower() == "none":
        return None

    return s

def _normalize_header(s: str) -> str:
    """
    Aggressive normalization to handle OCR spacing/noise like:
    'Summar y of activ ities (24 Hours)'
    """
    if not s:
        return ""
    s = s.lower()
     
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _similarity(a: str, b: str) -> float:
    """Deterministic similarity score [0..1] using stdlib."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class LayoutItem:
    idx: int
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    linked_header_idx: Optional[int]
    ocr_text: str

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _items_from_page(layout_path: Path, layout_ocr_path: Path) -> List[LayoutItem]:
    """
    Expects both JSONs to have a list of items in the same order OR each item having 'idx'.
    If your JSON differs, adjust the mapping here (single place).
    """
    layout = _load_json(layout_path)
    layout_ocr = _load_json(layout_ocr_path)

     
     
     
    layout_items = layout["items"] if isinstance(layout, dict) and "items" in layout else layout
    ocr_items = layout_ocr["items"] if isinstance(layout_ocr, dict) and "items" in layout_ocr else layout_ocr

    if not isinstance(layout_items, list) or not isinstance(ocr_items, list):
        raise ValueError(f"Unexpected JSON structure in {layout_path} / {layout_ocr_path}")

    if len(layout_items) != len(ocr_items):
         
        by_idx_ocr = {}
        for it in ocr_items:
            if isinstance(it, dict) and "idx" in it:
                by_idx_ocr[int(it["idx"])] = it
        merged: List[LayoutItem] = []
        for it in layout_items:
            idx = int(it.get("idx"))
            ocr_it = by_idx_ocr.get(idx, {})
            merged.append(
                LayoutItem(
                    idx=idx,
                    label=str(it.get("label", "")),
                    x1=float(it.get("x1", it.get("bbox", [0, 0, 0, 0])[0])),
                    y1=float(it.get("y1", it.get("bbox", [0, 0, 0, 0])[1])),
                    x2=float(it.get("x2", it.get("bbox", [0, 0, 0, 0])[2])),
                    y2=float(it.get("y2", it.get("bbox", [0, 0, 0, 0])[3])),
                    linked_header_idx=(int(it["linked_header_idx"]) if it.get("linked_header_idx") is not None else None),
                    ocr_text=str(ocr_it.get("ocr_text", "")),
                )
            )
        return merged

     
    items: List[LayoutItem] = []
    for i, (it, oit) in enumerate(zip(layout_items, ocr_items)):
        idx = int(it.get("idx", i))
        items.append(
            LayoutItem(
                idx=idx,
                label=str(it.get("label", "")),
                x1=float(it.get("x1", it.get("bbox", [0, 0, 0, 0])[0])),
                y1=float(it.get("y1", it.get("bbox", [0, 0, 0, 0])[1])),
                x2=float(it.get("x2", it.get("bbox", [0, 0, 0, 0])[2])),
                y2=float(it.get("y2", it.get("bbox", [0, 0, 0, 0])[3])),
                linked_header_idx=(int(it["linked_header_idx"]) if it.get("linked_header_idx") is not None else None),
                ocr_text=str(oit.get("ocr_text", "")),
            )
        )
    return items


def _match_target_for_header(header_text: str, threshold: float = 0.78) -> Tuple[Optional[str], float]:
    """
    Returns (target_field_name, score) or (None, best_score).
    target_field_name is one of TARGET_HEADERS keys.
    """
    norm = _normalize_header(header_text)

    best_key = None
    best_score = 0.0
    for key, canon in TARGET_HEADERS.items():
        canon_norm = _normalize_header(canon)
        score = _similarity(norm, canon_norm)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= threshold:
        return best_key, best_score
    return None, best_score


def extract_activity_summaries_from_processed_ddr(
    processed_ddr_dir: str | Path,
    header_match_threshold: float = 0.78,
    debug: bool = False,
) -> Dict[str, str]:
    """
    Walks processed_ddr/page_XXX folders and extracts:
      - activities_24h_text
      - planned_24h_text
    Uses:
      - section_header items' OCR text to identify target headers
      - plain_text items' linked_header_idx to route text
    """
    processed_ddr_dir = Path(processed_ddr_dir)

     
    collected: Dict[str, List[Tuple[Tuple[float, float], str]]] = {
        "activities_24h_text": [],
        "planned_24h_text": [],
    }

     
    matched_headers_global: Dict[int, Tuple[str, float, str]] = {}   

    page_dirs = sorted([p for p in processed_ddr_dir.iterdir() if p.is_dir() and p.name.startswith("page_")])

    for page_dir in page_dirs:
        layout_path = page_dir / "layout.json"
        layout_ocr_path = page_dir / "layout_ocr.json"
        if not layout_path.exists() or not layout_ocr_path.exists():
            continue

        items = _items_from_page(layout_path, layout_ocr_path)

         
        by_idx: Dict[int, LayoutItem] = {it.idx: it for it in items}

         
        header_target_for_idx: Dict[int, str] = {}
        for it in items:
            if it.label != "section_header":
                continue
            target_key, score = _match_target_for_header(it.ocr_text, threshold=header_match_threshold)
            if target_key:
                header_target_for_idx[it.idx] = target_key
                matched_headers_global[it.idx] = (target_key, score, it.ocr_text)

         
        for it in items:
            if it.label != "plain_text":
                continue
            if it.linked_header_idx is None:
                continue

            target_key = header_target_for_idx.get(it.linked_header_idx)
            if not target_key:
                continue

            text = _clean_plain_text(it.ocr_text)
            if text is None:
                continue

             
            order_key = (it.y1, it.x1)
            collected[target_key].append((order_key, text))

        if debug:
            print(f"[{page_dir.name}] matched headers: {header_target_for_idx}")

     
    result: Dict[str, str] = {}
    for key in ("activities_24h_text", "planned_24h_text"):
        parts = sorted(collected[key], key=lambda t: (t[0][0], t[0][1]))
        if not parts:
            result[key] = None
        else:
            joined = "\n\n".join(t for _, t in parts).strip()
            result[key] = joined if joined else None

    if debug:
        print("Matched headers (global):")
        for hidx, (tkey, score, raw) in sorted(matched_headers_global.items(), key=lambda x: x[0]):
            print(f"  header_idx={hidx} -> {tkey} score={score:.3f} raw={raw!r}")

    return result