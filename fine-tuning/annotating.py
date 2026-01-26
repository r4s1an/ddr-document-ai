from __future__ import annotations
from PIL import Image
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import statistics

import pdfplumber
from pdf2image import convert_from_path
import random


PDF_FOLDER = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000"
PDF_SAMPLE_SIZE = 200
PDF_SAMPLE_SEED = 42

PDF_PATH = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_19_ST2_1993_01_05.pdf"
POPPLER_BIN = r"C:\Users\Yoked\Desktop\DDR Processor\poppler-25.12.0\Library\bin"
DPI = 320

OUT_ROOT = Path("dataset")                
SPLIT = "train"                           
SAVE_DEBUG = True                         
DEBUG_DIR = OUT_ROOT / "debug"
 
CLASSES = [
    "table",
    "figure",
    "plain_text", 
    "section_header",
    "wellbore_field",
    "period_field", 
]

CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

PdfBBox = Tuple[float, float, float, float]      
PixBBox = Tuple[int, int, int, int]              

@dataclass
class Detection:
    class_name: str
    pdf_bbox: PdfBBox
    score: float = 1.0   

DetectorFn = Callable[[pdfplumber.page.Page], List[Detection]]

def pdf_bbox_to_pixel_bbox(pdf_bbox: PdfBBox, dpi: int) -> PixBBox:
    """Convert PDF points (72 dpi) bbox to pixel bbox at render dpi."""
    x0, top, x1, bottom = pdf_bbox
    s = dpi / 72.0
    return (int(x0 * s), int(top * s), int(x1 * s), int(bottom * s))

def clip_pixel_bbox(b: PixBBox, img_w: int, img_h: int) -> Optional[PixBBox]:
    """Clip bbox to image bounds; return None if invalid after clipping."""
    x0, y0, x1, y1 = b
    x0 = max(0, min(img_w - 1, x0))
    y0 = max(0, min(img_h - 1, y0))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)

def pixel_bbox_to_yolo(b: PixBBox, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert pixel xyxy bbox to YOLO normalized (xc, yc, w, h)."""
    x0, y0, x1, y1 = b
    bw = x1 - x0
    bh = y1 - y0
    xc = x0 + bw / 2.0
    yc = y0 + bh / 2.0
    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def _same_line_words(words: List[dict], anchor_top: float, y_tol: float) -> List[dict]:
    """Return words whose 'top' is within y_tol of anchor_top."""
    return [w for w in words if abs(w["top"] - anchor_top) <= y_tol]


def _expand_bbox(pdf_bbox: PdfBBox, page_w: float, page_h: float, padding: float) -> PdfBBox:
    """Expand bbox by padding (in PDF points), clipped to page bounds."""
    x0, top, x1, bottom = pdf_bbox
    x0 = max(0.0, x0 - padding)
    top = max(0.0, top - padding)
    x1 = min(page_w, x1 + padding)
    bottom = min(page_h, bottom + padding)
    return (x0, top, x1, bottom)

def find_line_segment_bbox(
    page: pdfplumber.page.Page,
    start_pattern: str,
    stop_pattern: Optional[str] = None,
    y_tol: float = 4.0,
    padding: float = 1,
    flags=re.IGNORECASE,
) -> Optional[PdfBBox]:
    words = extract_words_clean(page)
    
    start_idx = None
    start_re = re.compile(start_pattern, flags=flags)
    for i, w in enumerate(words):
        if start_re.search(w["text"]):
            start_idx = i
            break
    
    if start_idx is None:
        return None
    
    anchor_word = words[start_idx]
    line_words = _same_line_words(words, anchor_top=anchor_word["top"], y_tol=y_tol)
    line_words = sorted(line_words, key=lambda x: x["x0"])
    
    
    stop_x = page.width
    if stop_pattern:
        stop_re = re.compile(stop_pattern, flags=flags)
        for w in line_words:
            if w["x0"] > anchor_word["x0"] and stop_re.search(w["text"]):
                stop_x = w["x0"]
                break
    
    seg = [
        w for w in line_words
        if w["x0"] >= anchor_word["x0"] and w["x1"] <= stop_x
    ]
        
    if not seg:
        return None

    x0 = min(w["x0"] for w in seg)
    x1 = max(w["x1"] for w in seg)
    top = min(w["top"] for w in seg)
    bottom = max(w["bottom"] for w in seg)

    return _expand_bbox((x0, top, x1, bottom), page.width, page.height, padding=padding)

def group_words_into_lines(words: List[dict], y_tol: float = 3.0) -> List[List[dict]]:
    """
    Group word dicts into lines based on similar 'top' coordinate.
    Returns list of lines, where each line is a list of word dicts sorted by x0.
    """
    if not words:
        return []

     
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))

    lines: List[List[dict]] = []
    current: List[dict] = []
    current_top: Optional[float] = None

    for w in words_sorted:
        if current_top is None:
            current = [w]
            current_top = w["top"]
            continue

        if abs(w["top"] - current_top) <= y_tol:
            current.append(w)
        else:
             
            current = sorted(current, key=lambda ww: ww["x0"])
            lines.append(current)

             
            current = [w]
            current_top = w["top"]

     
    current = sorted(current, key=lambda ww: ww["x0"])
    lines.append(current)

    return lines

def clip_table_at_next_section_header(
    table_bbox: PdfBBox,
    header_bboxes: List[PdfBBox],
    min_sep: float = 15.0,    
    margin: float = 2.0,      
) -> PdfBBox:
    x0, top, x1, bottom = table_bbox

     
    candidates = [
        hb for hb in header_bboxes
        if (hb[1] > top + min_sep) and (hb[1] < bottom - min_sep)
    ]
    if not candidates:
        return table_bbox

    next_h = min(candidates, key=lambda hb: hb[1])   
    new_bottom = max(top + 5.0, next_h[1] - margin)
    return (x0, top, x1, new_bottom)


def bbox_area(b: PdfBBox) -> float:
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

def bbox_center(b: PdfBBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def split_pdfs(pdf_paths, train_frac=0.8, val_frac=0.1, seed=42):
    pdfs = list(pdf_paths)
    random.Random(seed).shuffle(pdfs)

    n = len(pdfs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": pdfs[:n_train],
        "val": pdfs[n_train:n_train + n_val],
        "test": pdfs[n_train + n_val:],
    }

def bbox_contains_point(b: PdfBBox, pt: Tuple[float, float]) -> bool:
    x0, top, x1, bottom = b
    x, y = pt
    return (x0 <= x <= x1) and (top <= y <= bottom)

def drop_container_boxes(
    bboxes: List[PdfBBox],
    min_children: int = 2,
    area_mult: float = 3.0,
) -> List[PdfBBox]:
    """
    Drop boxes that are much bigger than typical AND contain multiple other boxes.
    This kills the 'whole page is a table' super-box while keeping real tables.
    """
    if len(bboxes) < 2:
        return bboxes

    areas = [bbox_area(b) for b in bboxes]
    med = statistics.median(areas) if areas else 0.0
    if med <= 0:
        return bboxes

    kept: List[PdfBBox] = []
    centers = [bbox_center(b) for b in bboxes]

    for i, b in enumerate(bboxes):
        child_count = 0
        for j, c in enumerate(bboxes):
            if i == j:
                continue
            if bbox_contains_point(b, centers[j]):
                child_count += 1

        if child_count >= min_children and bbox_area(b) > area_mult * med:
             
            continue

        kept.append(b)

    return kept

def bbox_iou(a: PdfBBox, b: PdfBBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def dedupe_bboxes(bboxes: List[PdfBBox], iou_thr: float = 0.85) -> List[PdfBBox]:
    """Keep the largest boxes when bboxes overlap heavily."""
    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    kept: List[PdfBBox] = []
    for b in bboxes:
        if all(bbox_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept

def words_intersect_bbox(page: pdfplumber.page.Page, bbox: PdfBBox) -> List[dict]:
    """Words that intersect bbox (not necessarily fully contained)."""
    x0, top, x1, bottom = bbox
    words = extract_words_clean(page, tol=2.5, extra_attrs=None)
    hit = []
    for w in words:
        if (w["x1"] > x0 and w["x0"] < x1 and
            w["bottom"] > top and w["top"] < bottom):
            hit.append(w)
    return hit


def tighten_bbox_to_words(
    page: pdfplumber.page.Page,
    bbox: PdfBBox,
    padding: float = 1.0,
) -> PdfBBox:
    """
    Shrink bbox to the union of words inside it (then add small padding).
    This fixes 'bbox extends to page edge' cases caused by template lines.
    """
    hit = words_intersect_bbox(page, bbox)
    if not hit:
        return bbox   

    x0 = min(w["x0"] for w in hit)
    x1 = max(w["x1"] for w in hit)
    top = min(w["top"] for w in hit)
    bottom = max(w["bottom"] for w in hit)

    return _expand_bbox((x0, top, x1, bottom), page.width, page.height, padding=padding)


def is_runaway_table_bbox(page: pdfplumber.page.Page, bbox: PdfBBox) -> bool:
    """Heuristic: bbox is probably wrong if it clamps to page edges / too tall."""
    x0, top, x1, bottom = bbox
    h = bottom - top
     
    return (top <= 1.0) or (bottom >= page.height - 1.0) or (h >= 0.70 * page.height)


def build_columns(line_words: List[dict], col_gap: float = 12.0) -> List[Tuple[float, float]]:
    """Group multi-word headers into columns using small gap threshold."""
    ws = sorted(line_words, key=lambda w: w["x0"])
    if not ws:
        return []
    cols = []
    cur_x0, cur_x1 = ws[0]["x0"], ws[0]["x1"]
    for prev, w in zip(ws, ws[1:]):
        if (w["x0"] - prev["x1"]) >= col_gap:
            cols.append((cur_x0, cur_x1))
            cur_x0, cur_x1 = w["x0"], w["x1"]
        else:
            cur_x1 = max(cur_x1, w["x1"])
    cols.append((cur_x0, cur_x1))
    return cols

def collapse_doubled_token(t: str) -> str:
    original = t
    t = (t or "").strip()
    
    if len(t) >= 2 and len(t) % 2 == 0:
        if all(t[i] == t[i+1] for i in range(0, len(t), 2)):
            result = t[::2]
            if "elbore" in result.lower() or "ellbore" in original.lower():
                print(f"COLLAPSE DEBUG: '{original}' -> '{result}'")
                print(f"  Length: {len(original)}, pairs matched: {all(original[i] == original[i+1] for i in range(0, len(original), 2))}")
            return result
    return t

def extract_words_clean(page: pdfplumber.page.Page, tol: float = 2.5, extra_attrs=None) -> List[dict]:
    """
    Extract words and collapse doubled tokens.
    NOTE: We do NOT use dedupe_chars here because it can create inconsistent results.
    """
    extra_attrs = extra_attrs or ["size", "fontname"]
     
    words = page.extract_words(extra_attrs=extra_attrs)
    
    for w in words:
        w["text"] = collapse_doubled_token(w["text"])
    
    return words

def line_to_text(line_words: List[dict]) -> str:
    return " ".join(collapse_doubled_token(w["text"]) for w in line_words).strip()

def resolve_nested_detections(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    If a smaller box is largely contained within a larger box of the same or similar class, 
    keep only the larger one (the table).
    """
    if not detections:
        return []

     
    detections = sorted(detections, key=lambda d: bbox_area(d.pdf_bbox), reverse=True)
    keep = []

    for i, det_heavy in enumerate(detections):
        is_redundant = False
        for j, det_kept in enumerate(keep):
             
            iou = bbox_iou(det_heavy.pdf_bbox, det_kept.pdf_bbox)
            
             
            if iou > iou_threshold:
                is_redundant = True
                break
            
             
             
            h_box = det_heavy.pdf_bbox
            k_box = det_kept.pdf_bbox
            if (h_box[0] >= k_box[0] - 2 and h_box[1] >= k_box[1] - 2 and
                h_box[2] <= k_box[2] + 2 and h_box[3] <= k_box[3] + 2):
                 
                if det_kept.class_name == "table":
                    is_redundant = True
                    break

        if not is_redundant:
            keep.append(det_heavy)
            
    return keep

def get_random_pdf_sample(folder_path: str, sample_size: int = 150, seed: int = 42):
    all_pdfs = list(Path(folder_path).glob("*.pdf"))

    if len(all_pdfs) <= sample_size:
        print(f"⚠️ Only {len(all_pdfs)} PDFs found. Using all.")
        return sorted(all_pdfs)

    random.seed(seed)
    sampled_pdfs = random.sample(all_pdfs, sample_size)
    return sorted(sampled_pdfs)

def get_section_headers_with_titles(page: pdfplumber.page.Page, y_tol: float = 3.0, padding: float = 1.5):
    words = extract_words_clean(page, tol=2.5, extra_attrs=["size", "fontname"])
    lines = group_words_into_lines(words, y_tol=y_tol)

    out = []
    for line_words in lines:
        text = line_to_text(line_words).lower()
        matched = None
        for target in SECTION_TITLES_NORM:
            if target in text:
                matched = target
                break
        if not matched:
            continue

        x0 = min(w["x0"] for w in line_words)
        x1 = max(w["x1"] for w in line_words)
        top = min(w["top"] for w in line_words)
        bottom = max(w["bottom"] for w in line_words)
        bbox = _expand_bbox((x0, top, x1, bottom), page.width, page.height, padding=padding)
        out.append((matched, bbox))

    out.sort(key=lambda x: x[1][1])   
    return out

def detect_tables_pdfplumber(page: pdfplumber.page.Page, padding: float = 2.0) -> List[Detection]:
    p = page.dedupe_chars(tolerance=1)

     
    header_boxes = [d.pdf_bbox for d in detect_section_headers(page, y_tol=3.0, padding=1.5)]

    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 20,
        "intersection_tolerance": 3,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
    }

    bboxes: List[PdfBBox] = []

     
    for settings in (
        dict(table_settings, vertical_strategy="lines", horizontal_strategy="lines"),
    ):
        try:
            for t in p.find_tables(table_settings=settings):
                bboxes.append(tuple(t.bbox))   
        except Exception:
            pass

    if not bboxes:
        return []

    page_area = page.width * page.height
    fixed: List[PdfBBox] = []

    for b in bboxes:
         
        b = clip_table_at_next_section_header(b, header_boxes, min_sep=15.0, margin=2.0)

         
        if (bbox_area(b) / page_area) > 0.85:
            continue

         
        if is_runaway_table_bbox(page, b):
            b = tighten_bbox_to_words(page, b, padding=1.0)

         
        b = _expand_bbox(b, page.width, page.height, padding=padding)
        fixed.append(b)

    fixed = dedupe_bboxes(fixed, iou_thr=0.85)
    return [Detection(class_name="table", pdf_bbox=b) for b in fixed]


def detect_wellbore_field(page: pdfplumber.page.Page) -> List[Detection]:
    bbox = find_line_segment_bbox(
        page,
        start_pattern=r"wellbore",  
        stop_pattern=r"period",     
        y_tol=4.0,                  
        padding=1,
    )
    if bbox is None:
        return []
    return [Detection(class_name="wellbore_field", pdf_bbox=bbox)]

def detect_period_field(page: pdfplumber.page.Page) -> List[Detection]:
    """
    Detect bbox for 'Period: 1979-12-31 00:00 - 1980-01-01 00:00' (line segment).
    We go from 'Period:' to end of that same line.
    """
    bbox = find_line_segment_bbox(
        page,
        start_pattern=r"Period:?",
        stop_pattern=None,    
        y_tol=3.0,
        padding=1.5,
    )
    if bbox is None:
        return []
    return [Detection(class_name="period_field", pdf_bbox=bbox)]

SECTION_TITLES = [
    "Summary report",
    "Summary of activities (24 Hours)",
    "Summary of planned activities (24 Hours)",
    "Operations",
    "Drilling Fluid",
    "Pore Pressure",
    "Survey Station",
    "Stratigraphic Information",
    "Lithology Information",
    "Gas Reading Information",
    "Bit Record",
    "Equipment Failure Information",
    "Casing Liner Tubing",
]

 
SECTION_TITLES_NORM = [t.lower() for t in SECTION_TITLES]

def detect_section_headers(page: pdfplumber.page.Page, y_tol: float = 3.0, padding: float = 1.5) -> List[Detection]:
    """
    Detect section header bars by matching known DDR section titles at line level.
    Returns one bbox per matched line.
    """
    words = extract_words_clean(page, tol=2.5, extra_attrs=["size", "fontname"])

    lines = group_words_into_lines(words, y_tol=y_tol)

    detections: List[Detection] = []
    for line_words in lines:
        text = line_to_text(line_words)
        text_norm = text.lower()

         
        matched = None
        for target in SECTION_TITLES_NORM:
            if target in text_norm:
                matched = target
                break

        if not matched:
            continue

         
        x0 = min(w["x0"] for w in line_words)
        x1 = max(w["x1"] for w in line_words)
        top = min(w["top"] for w in line_words)
        bottom = max(w["bottom"] for w in line_words)

        bbox = _expand_bbox((x0, top, x1, bottom), page.width, page.height, padding=padding)
        detections.append(Detection(class_name="section_header", pdf_bbox=bbox))

    return detections

def detect_summary_plain_text(
    page: pdfplumber.page.Page,
    y_tol: float = 3.0,
    padding: float = 1.0,
) -> List[Detection]:
    detections: List[Detection] = []
    
     
    headers_with_titles = get_section_headers_with_titles(page, y_tol=y_tol, padding=padding)

    headers_with_titles.sort(key=lambda x: x[1][1])
    words = extract_words_clean(page, tol=2.5, extra_attrs=["size", "fontname"])
    
     
    for i, (title, header_bbox) in enumerate(headers_with_titles):
        t_lower = title.lower()
        
         
         
        is_activity_summary = "summary" in t_lower and ("activities" in t_lower or "planned" in t_lower)

        if not is_activity_summary:
            continue
                    
        header_bottom = header_bbox[3]
        next_header_top = headers_with_titles[i + 1][1][1] if i + 1 < len(headers_with_titles) else page.height
                
         
        content_words = []
        for w in words:
            if w["top"] > header_bottom and w["top"] < next_header_top:
                content_words.append(w)
            elif w["top"] <= header_bottom:
                 
                pass 

        if not content_words:
             
            continue
        
        x0 = min(w["x0"] for w in content_words)
        x1 = max(w["x1"] for w in content_words)
        top = min(w["top"] for w in content_words)
        bottom = max(w["bottom"] for w in content_words)
        
        bbox = _expand_bbox((x0, top, x1, bottom), page.width, page.height, padding=padding)
        detections.append(Detection(class_name="plain_text", pdf_bbox=bbox))
    
    return detections

def detect_one_row_tables(
    page: pdfplumber.page.Page,
    y_tol: float = 3.0,
    padding: float = 1.0,
    min_columns: int = 3,   
) -> List[Detection]:
    """
    Detect one-row tables by finding:
    1. A section header (e.g., "Survey Station")
    2. A header row with multiple column names
    3. A single data row below it
    """
    detections: List[Detection] = []
    
     
    headers_with_titles = get_section_headers_with_titles(page, y_tol=y_tol, padding=padding)
    
     
    target_sections = [
        "survey station",
        "lithology information",
        "gas reading information",
        "pore pressure", 
        "bit record",
        "equipment failure information",
        "casing liner tubing",  
    ]
    
     
    headers_with_titles.sort(key=lambda x: x[1][1])
    
     
    words = extract_words_clean(page, tol=2.5, extra_attrs=["size", "fontname"])
    lines = group_words_into_lines(words, y_tol=y_tol)
    
    for i, (title, header_bbox) in enumerate(headers_with_titles):
        if title not in target_sections:
            continue
        
        header_bottom = header_bbox[3]
        
         
        next_header_top = page.height
        if i + 1 < len(headers_with_titles):
            next_header_top = headers_with_titles[i + 1][1][1]
        
         
        content_lines = [
            line for line in lines
            if line[0]["top"] > header_bottom + 2
            and line[0]["top"] < next_header_top - 5
        ]
        
        if len(content_lines) < 2:
             
            continue
        
         
        header_line = content_lines[0]
        data_line = content_lines[1]
        
         
        if len(header_line) < min_columns:
            continue
        
         
        if len(data_line) < min_columns - 1:   
            continue
        
         
        all_words = header_line + data_line
        x0 = min(w["x0"] for w in all_words)
        x1 = max(w["x1"] for w in all_words)
        top = min(w["top"] for w in all_words)
        bottom = max(w["bottom"] for w in all_words)
        
         
        bbox = _expand_bbox(
            (x0, top, x1, bottom),
            page.width,
            page.height,
            padding=padding
        )
        
        detections.append(Detection(class_name="table", pdf_bbox=bbox))
    
    return detections
 

def ensure_out_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def render_all_pages(pdf_path: str, dpi: int, poppler_bin: Optional[str]) -> List["Image.Image"]:
    """Render all pages in the PDF to PIL images."""
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=poppler_bin,
    )

def write_yolo_labels(label_path: Path, detections_yolo: List[str]) -> None:
    """Write YOLO label lines to file (or empty file if none)."""
    label_path.write_text("\n".join(detections_yolo) + ("\n" if detections_yolo else ""), encoding="utf-8")


def draw_debug(image_pil, pixel_boxes: List[Tuple[str, PixBBox]], out_path: Path) -> None:
    """Optional: draw bbox overlays for quick QA."""
    import cv2
    import numpy as np

    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    for cls, (x0, y0, x1, y1) in pixel_boxes:
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(img, cls, (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img)


def build_dataset_from_pdf(
    pdf_path: str,
    out_root: Path,
    split: str,
    dpi: int,
    poppler_bin: Optional[str],
    detectors: List[DetectorFn],
) -> None:
    img_dir, lbl_dir = ensure_out_dirs(out_root, split)

    images = render_all_pages(pdf_path, dpi=dpi, poppler_bin=poppler_bin)

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        if len(images) != n_pages:
            raise RuntimeError(f"Rendered images ({len(images)}) != PDF pages ({n_pages}).")

        for i in range(n_pages):
            page_num = i + 1
            page = pdf.pages[i]
            pil_img = images[i]
            img_w, img_h = pil_img.size

             
            detections: List[Detection] = []
            for det_fn in detectors:
                detections.extend(det_fn(page))

            detections = resolve_nested_detections(detections, iou_threshold=0.6)

             
            yolo_lines: List[str] = []
            debug_boxes: List[Tuple[str, PixBBox]] = []

            for det in detections:
                if det.class_name not in CLASS_TO_ID:
                     
                    continue

                pix_bbox = pdf_bbox_to_pixel_bbox(det.pdf_bbox, dpi=dpi)
                pix_bbox = clip_pixel_bbox(pix_bbox, img_w, img_h)
                if pix_bbox is None:
                    continue

                xc, yc, bw, bh = pixel_bbox_to_yolo(pix_bbox, img_w, img_h)
                cls_id = CLASS_TO_ID[det.class_name]
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                if SAVE_DEBUG:
                    debug_boxes.append((det.class_name, pix_bbox))

             
            stem = f"{Path(pdf_path).stem}_p{page_num:03d}"
            img_path = img_dir / f"{stem}.png"
            lbl_path = lbl_dir / f"{stem}.txt"

             
            pil_img.save(img_path)
            write_yolo_labels(lbl_path, yolo_lines)

             
            if SAVE_DEBUG and debug_boxes:
                dbg_path = DEBUG_DIR / f"{stem}_debug.png"
                draw_debug(pil_img, debug_boxes, dbg_path)

            print(f"[{page_num:03d}/{n_pages:03d}] saved {img_path.name} + {lbl_path.name} | labels={len(yolo_lines)}")

 
def test_single_pdf(pdf_path: str, detectors: List[DetectorFn]):
    """Runs the full pipeline on just one specific PDF for debugging."""
    print(f"\n🧪 TEST MODE: Processing single file: {pdf_path}")
    
     
    build_dataset_from_pdf(
        pdf_path=pdf_path,
        out_root=OUT_ROOT,
        split="test_run",
        dpi=DPI,
        poppler_bin=POPPLER_BIN,
        detectors=detectors,
    )
    print(f"✅ Test complete. Check results in {OUT_ROOT}/images/test_run and {DEBUG_DIR}")

if __name__ == "__main__":
    detectors = [
        detect_wellbore_field,
        detect_period_field,
        detect_section_headers,
        detect_tables_pdfplumber,
        detect_one_row_tables,
        detect_summary_plain_text,
    ]
  
    RUN_SINGLE_TEST = False
     

    if RUN_SINGLE_TEST:
         
        test_single_pdf(PDF_PATH, detectors)
    else:
         
        pdf_files = get_random_pdf_sample(
            folder_path=PDF_FOLDER,
            sample_size=PDF_SAMPLE_SIZE,
            seed=PDF_SAMPLE_SEED,
        )

        print(f"📄 Processing {len(pdf_files)} PDFs (seed={PDF_SAMPLE_SEED})")

        splits = split_pdfs(
            pdf_files,
            train_frac=0.8,
            val_frac=0.1,
            seed=42,
        )

        for split, pdf_list in splits.items():
            print(f"\n📂 Building {split} split ({len(pdf_list)} PDFs)")
            for pdf_path in pdf_list:
                build_dataset_from_pdf(
                    pdf_path=str(pdf_path),
                    out_root=OUT_ROOT,
                    split=split,
                    dpi=DPI,
                    poppler_bin=POPPLER_BIN,
                    detectors=detectors,
                )

    print("\n✅ Processing finished.")