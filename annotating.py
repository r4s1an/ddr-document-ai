"""
Auto-annotation dataset builder for DDR PDFs (YOLO format)

Creates:
dataset/
  images/train/*.png
  labels/train/*.txt
(and optionally dataset/debug/*.png)

You can add more "detectors" by writing a function that returns
a list of (class_name, pdf_bbox) where pdf_bbox is (x0, top, x1, bottom) in PDF points.
"""

from __future__ import annotations
from PIL import Image
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import statistics

import pdfplumber
from pdf2image import convert_from_path

# ---------------------------
# CONFIG
# ---------------------------

PDF_PATH = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_19_A_1997_08_09.pdf"
POPPLER_BIN = r"C:\Users\Yoked\Desktop\DDR Processor\poppler-25.12.0\Library\bin"
DPI = 320

OUT_ROOT = Path("dataset")               # output dataset root
SPLIT = "train"                          # keep "train" for now; you can add val split later
SAVE_DEBUG = True                        # saves debug images with bboxes drawn
DEBUG_DIR = OUT_ROOT / "debug"

# Your final YOLO class list (order matters => class_id)
CLASSES = [
    "table",          # keep for later (not detected here)
    "figure",         # keep for later (not detected here)
    "text_block",     # keep for later (not detected here)
    "section_header", # custom (not detected here)
    "wellbore_field", # ✅ detected by our example detector
    "period_field",   # custom (easy to add)
]

CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# ---------------------------
# TYPES
# ---------------------------

PdfBBox = Tuple[float, float, float, float]     # (x0, top, x1, bottom) in PDF points
PixBBox = Tuple[int, int, int, int]             # (x0, y0, x1, y1) in pixels


@dataclass
class Detection:
    class_name: str
    pdf_bbox: PdfBBox
    score: float = 1.0  # optional, for later extensions


DetectorFn = Callable[[pdfplumber.page.Page], List[Detection]]

# ---------------------------
# COORDINATE HELPERS
# ---------------------------

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

# ---------------------------
# TEXT-LINE DETECTION HELPERS
# ---------------------------

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
        # DEBUG PRINT
    print(f"Original width: {x1-x0:.2f} | Padding: {padding} | New width: {(x1+padding)-(x0-padding):.2f}")
    return (x0, top, x1, bottom)


def find_line_segment_bbox(
    page: pdfplumber.page.Page,
    start_pattern: str,
    stop_pattern: Optional[str] = None,
    y_tol: float = 3.0,
    padding: float = 1,
    flags=re.IGNORECASE,
) -> Optional[PdfBBox]:
    """
    Find a bbox for a line segment that starts at a token matching start_pattern
    and optionally stops before token matching stop_pattern.

    Returns bbox in PDF points: (x0, top, x1, bottom) or None.
    """
    # dedupe helps with duplicated characters
    words = extract_words_clean(page, tol=2.5, extra_attrs=["size", "fontname"])

    # find start token
    start_idx = None
    start_re = re.compile(start_pattern, flags=flags)
    for i, w in enumerate(words):
        if start_re.fullmatch(w["text"]):
            start_idx = i
            break
    if start_idx is None:
        return None

    line_top = words[start_idx]["top"]
    line_words = _same_line_words(words, anchor_top=line_top, y_tol=y_tol)

    # establish stop x if needed
    stop_x = None
    if stop_pattern:
        stop_re = re.compile(stop_pattern, flags=flags)
        for w in line_words:
            if stop_re.fullmatch(w["text"]):
                stop_x = w["x0"]
                break

    start_x = words[start_idx]["x0"]

    seg = [
        w for w in line_words
        if w["x0"] >= start_x and (stop_x is None or w["x1"] <= stop_x)
    ]
    if not seg:
        return None

    x0 = min(w["x0"] for w in seg)
    x1 = max(w["x1"] for w in seg)
    top = min(w["top"] for w in seg)
    bottom = max(w["bottom"] for w in seg)

    bbox = (x0, top, x1, bottom)
    bbox = _expand_bbox(bbox, page.width, page.height, padding=padding)
    return bbox

def group_words_into_lines(words: List[dict], y_tol: float = 3.0) -> List[List[dict]]:
    """
    Group word dicts into lines based on similar 'top' coordinate.
    Returns list of lines, where each line is a list of word dicts sorted by x0.
    """
    if not words:
        return []

    # Sort by vertical then horizontal
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
            # finalize old line
            current = sorted(current, key=lambda ww: ww["x0"])
            lines.append(current)

            # start new line
            current = [w]
            current_top = w["top"]

    # finalize last line
    current = sorted(current, key=lambda ww: ww["x0"])
    lines.append(current)

    return lines

def clip_table_at_next_section_header(
    table_bbox: PdfBBox,
    header_bboxes: List[PdfBBox],
    min_sep: float = 15.0,   # header must be at least this far below table top
    margin: float = 2.0,     # leave small gap above header
) -> PdfBBox:
    x0, top, x1, bottom = table_bbox

    # headers whose TOP lies inside the table vertically (and is not just barely below top)
    candidates = [
        hb for hb in header_bboxes
        if (hb[1] > top + min_sep) and (hb[1] < bottom - min_sep)
    ]
    if not candidates:
        return table_bbox

    next_h = min(candidates, key=lambda hb: hb[1])  # nearest header below
    new_bottom = max(top + 5.0, next_h[1] - margin)
    return (x0, top, x1, new_bottom)


def bbox_area(b: PdfBBox) -> float:
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

def bbox_center(b: PdfBBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

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
            # likely a merged "super-table"
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
    padding: float = 2.0,
) -> PdfBBox:
    """
    Shrink bbox to the union of words inside it (then add small padding).
    This fixes 'bbox extends to page edge' cases caused by template lines.
    """
    hit = words_intersect_bbox(page, bbox)
    if not hit:
        return bbox  # nothing to tighten with

    x0 = min(w["x0"] for w in hit)
    x1 = max(w["x1"] for w in hit)
    top = min(w["top"] for w in hit)
    bottom = max(w["bottom"] for w in hit)

    return _expand_bbox((x0, top, x1, bottom), page.width, page.height, padding=padding)


def is_runaway_table_bbox(page: pdfplumber.page.Page, bbox: PdfBBox) -> bool:
    """Heuristic: bbox is probably wrong if it clamps to page edges / too tall."""
    x0, top, x1, bottom = bbox
    h = bottom - top
    # touches edges OR covers most of page height
    return (top <= 1.0) or (bottom >= page.height - 1.0) or (h >= 0.70 * page.height)

def filter_words_in_y(words: List[dict], y0: float, y1: float) -> List[dict]:
    return [w for w in words if (w["bottom"] >= y0 and w["top"] <= y1)]

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

def row_matches_columns(row_words: List[dict], cols: List[Tuple[float, float]], x_tol: float = 10.0, min_hits: int = 3) -> bool:
    """Row is 'table-like' if words land in several header columns."""
    if not cols or not row_words:
        return False
    hits = 0
    for (cx0, cx1) in cols:
        for w in row_words:
            # word starts near column start OR intersects the column range
            if abs(w["x0"] - cx0) <= x_tol or (w["x0"] < cx1 and w["x1"] > cx0):
                hits += 1
                break
    return hits >= min_hits

def is_data_row(text: str) -> bool:
    t = text.strip()
    return any(ch.isdigit() for ch in t) or (":" in t)

def collapse_doubled_token(t: str) -> str:
    t = (t or "").strip()
    # Collapse only if every char is doubled: TTii -> Ti
    if len(t) >= 4 and len(t) % 2 == 0:
        if all(t[i] == t[i+1] for i in range(0, len(t), 2)):
            return t[::2]
    return t

def extract_words_clean(page: pdfplumber.page.Page, tol: float = 2.5, extra_attrs=None) -> List[dict]:
    """
    1) dedupe chars (reduces double-drawn glyphs)
    2) collapse doubled tokens (TTiimmee -> Time)
    """
    extra_attrs = extra_attrs or ["size", "fontname"]
    p = page.dedupe_chars(tolerance=tol)
    words = p.extract_words(extra_attrs=extra_attrs)
    for w in words:
        w["text"] = collapse_doubled_token(w["text"])
    return words

def line_to_text(line_words: List[dict]) -> str:
    return " ".join(collapse_doubled_token(w["text"]) for w in line_words).strip()

# ---------------------------
# DETECTORS (ADD MORE LIKE THIS)
# ---------------------------
ONE_ROW_SECTION_HINTS = {
    "pore pressure": ["time", "depth", "reading"],
    "lithology information": ["start", "end", "depth", "description"],
    # add more if needed:
    # "survey station": ["depth", "inclination", "azimuth"],
    # "stratigraphic information": ["depth", "formation", "description"],
}

def get_section_headers_with_titles(page: pdfplumber.page.Page, y_tol: float = 3.0, padding: float = 2.5):
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

    out.sort(key=lambda x: x[1][1])  # sort by top
    return out


def detect_one_row_tables_by_section(
    page: pdfplumber.page.Page,
    y_tol: float = 3.0,
    col_gap: float = 12.0,
    x_tol: float = 10.0,
    padding: float = 2.0,
) -> List[Detection]:
    words_all = extract_words_clean(page, tol=2.5, extra_attrs=None)


    headers = get_section_headers_with_titles(page, y_tol=y_tol, padding=2.5)
    if not headers:
        return []

    dets: List[Detection] = []

    for idx, (title, hb) in enumerate(headers):
        if title not in ONE_ROW_SECTION_HINTS:
            continue

        y0 = hb[3] + 2.0
        y1 = (headers[idx + 1][1][1] - 2.0) if idx + 1 < len(headers) else page.height

        words = filter_words_in_y(words_all, y0, y1)
        lines = group_words_into_lines(words, y_tol=y_tol)

        hints = ONE_ROW_SECTION_HINTS[title]

        # find header row by keywords
        header_i = None
        for i, ln in enumerate(lines):
            txt = line_to_text(ln).lower()
            if sum(1 for h in hints if h in txt) >= max(2, len(hints) - 1):
                header_i = i
                break
        if header_i is None:
            continue

        # find next non-empty line as data row
        data_i = None
        for j in range(header_i + 1, min(header_i + 4, len(lines))):
            txt = line_to_text(lines[j]).strip()
            if txt:
                data_i = j
                break
        if data_i is None:
            continue

        header_ln = lines[header_i]
        data_ln = lines[data_i]

        cols = build_columns(header_ln, col_gap=col_gap)
        if not row_matches_columns(data_ln, cols, x_tol=x_tol, min_hits=3):
            continue
        if not is_data_row(line_to_text(data_ln)):
            continue

        bbox = _expand_bbox(
            (
                min(w["x0"] for w in header_ln + data_ln),
                min(w["top"] for w in header_ln),
                max(w["x1"] for w in header_ln + data_ln),
                max(w["bottom"] for w in data_ln),
            ),
            page.width, page.height, padding=padding
        )
        dets.append(Detection("table", bbox))

    return dets

def detect_tables_pdfplumber(page: pdfplumber.page.Page, padding: float = 2.0) -> List[Detection]:
    p = page.dedupe_chars(tolerance=1)

    # get section headers on this page (cheap, and avoids crossing sections)
    header_boxes = [d.pdf_bbox for d in detect_section_headers(page, y_tol=3.0, padding=2.5)]

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

    # run BOTH strategies always
    for settings in (
        dict(table_settings, vertical_strategy="lines", horizontal_strategy="lines"),
        dict(table_settings, vertical_strategy="text",  horizontal_strategy="text"),
    ):
        try:
            for t in p.find_tables(table_settings=settings):
                bboxes.append(tuple(t.bbox))  # type: ignore
        except Exception:
            pass

    if not bboxes:
        return []

    page_area = page.width * page.height
    fixed: List[PdfBBox] = []

    for b in bboxes:
        # 1) stop tables from crossing into a new section
        b = clip_table_at_next_section_header(b, header_boxes, min_sep=15.0, margin=2.0)

        # 2) drop “almost whole page” tables
        if (bbox_area(b) / page_area) > 0.85:
            continue

        # 3) your runaway fix (optional)
        if is_runaway_table_bbox(page, b):
            b = tighten_bbox_to_words(page, b, padding=2.0)

        # 4) final expand
        b = _expand_bbox(b, page.width, page.height, padding=padding)
        fixed.append(b)

    fixed = dedupe_bboxes(fixed, iou_thr=0.85)
    return [Detection(class_name="table", pdf_bbox=b) for b in fixed]


def detect_wellbore_field(page: pdfplumber.page.Page) -> List[Detection]:
    """
    Detect bbox for 'Wellbore: ...' part.
    We stop before 'Period:' if it exists on the same line.
    """
    bbox = find_line_segment_bbox(
        page,
        start_pattern=r"Wellbore:?",
        stop_pattern=r"Period:?",
        y_tol=3.0,
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
        stop_pattern=None,   # None => go to end of line
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

# normalized targets for robust matching
SECTION_TITLES_NORM = [t.lower() for t in SECTION_TITLES]


def detect_section_headers(page: pdfplumber.page.Page, y_tol: float = 3.0, padding: float = 2.5) -> List[Detection]:
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

        # Match any known title as substring (handles minor spacing differences)
        matched = None
        for target in SECTION_TITLES_NORM:
            if target in text_norm:
                matched = target
                break

        if not matched:
            continue

        # Union bbox over the entire line (covers the full header text/bar region)
        x0 = min(w["x0"] for w in line_words)
        x1 = max(w["x1"] for w in line_words)
        top = min(w["top"] for w in line_words)
        bottom = max(w["bottom"] for w in line_words)

        bbox = _expand_bbox((x0, top, x1, bottom), page.width, page.height, padding=padding)
        detections.append(Detection(class_name="section_header", pdf_bbox=bbox))

    return detections


# Example stub to show how you'd add another detector later:
# def detect_period_field(page: pdfplumber.page.Page) -> List[Detection]:
#     bbox = find_line_segment_bbox(page, start_pattern=r"Period:?", stop_pattern=None, y_tol=3.0, padding=2.5)
#     return [] if bbox is None else [Detection("period_field", bbox)]

# ---------------------------
# DATASET WRITER
# ---------------------------

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

    # Render all pages once
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

            # run all detectors on this page
            detections: List[Detection] = []
            for det_fn in detectors:
                detections.extend(det_fn(page))

            # convert to YOLO label lines
            yolo_lines: List[str] = []
            debug_boxes: List[Tuple[str, PixBBox]] = []

            for det in detections:
                if det.class_name not in CLASS_TO_ID:
                    # Skip unknown classes (or raise error)
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

            # file names
            stem = f"{Path(pdf_path).stem}_p{page_num:03d}"
            img_path = img_dir / f"{stem}.png"
            lbl_path = lbl_dir / f"{stem}.txt"

            # save image + labels
            pil_img.save(img_path)
            write_yolo_labels(lbl_path, yolo_lines)

            # debug overlay
            if SAVE_DEBUG and debug_boxes:
                dbg_path = DEBUG_DIR / f"{stem}_debug.png"
                draw_debug(pil_img, debug_boxes, dbg_path)

            print(f"[{page_num:03d}/{n_pages:03d}] saved {img_path.name} + {lbl_path.name} | labels={len(yolo_lines)}")


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    detectors = [
        detect_wellbore_field,
        detect_period_field,
        detect_section_headers,
        detect_tables_pdfplumber,
        detect_one_row_tables_by_section,
    ]

    build_dataset_from_pdf(
        pdf_path=PDF_PATH,
        out_root=OUT_ROOT,
        split=SPLIT,
        dpi=DPI,
        poppler_bin=POPPLER_BIN,
        detectors=detectors,
    )

    print("\n✅ Done.")
    print(f"Images: {OUT_ROOT / 'images' / SPLIT}")
    print(f"Labels: {OUT_ROOT / 'labels' / SPLIT}")
    if SAVE_DEBUG:
        print(f"Debug:  {DEBUG_DIR}")
