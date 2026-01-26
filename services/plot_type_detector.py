from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Dict, Tuple

import cv2

class PlotType(str, Enum):
    PRESSURE_VS_TIME = "pressure_vs_time"
    PRESSURE_VS_DEPTH = "pressure_vs_depth"


class PlotTypeDetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectionResult:
    plot_type: PlotType
    scores: Dict[str, int]
    ocr_text: str


# --- Keyword sets (explicit & extendable) ---
TIME_SIGNALS = [
    r"\bdate\b",
    r"\bpressure comparison\b",
    r"\boffset wells\b",
    r"\bwell[_\s-]*0?\d\b",
    r"\b20\d{2}[-/]\d{2}\b",   # 2019-07
    r"\b20\d{2}\b",            # 2018
    r"\bjan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec\b",
]

DEPTH_SIGNALS = [
    r"\bformation pressure profile\b",
    r"\btrue vertical depth\b",
    r"\bsub sea\b",
    r"\btvd\b",
    r"\bdepth\b",
    r"\bvirgin pressure\b",
    r"\bmin\b.*\bsor\b",
    r"\bmax\b.*\bsor\b",
    r"\bbase\b.*\bsor\b",
    r"\bpsi\b",
]


def _preprocess_for_ocr(img_bgr):
    """Deterministic preprocessing to stabilize OCR."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # upscale improves OCR on plots
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # denoise + binarize
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return thr
def _ocr_text_from_bgr(img_bgr) -> str:
    import easyocr

    proc = _preprocess_for_ocr(img_bgr)

    reader = easyocr.Reader(["en"], gpu=True)

    results = reader.readtext(
        proc,
        detail=0,
        paragraph=True
    )

    return " ".join(results).lower().strip()


def _ocr_text(image_path: Path) -> str:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"OpenCV failed to read: {image_path}")
    return _ocr_text_from_bgr(img)


def _score(text: str) -> Tuple[int, int, Dict[str, int]]:
    """
    Returns: (time_score, depth_score, detailed_scores)
    """
    scores: Dict[str, int] = {}

    def add_score(bucket: str, patterns, weight: int = 1) -> int:
        s = 0
        for p in patterns:
            if re.search(p, text, flags=re.IGNORECASE):
                s += weight
                scores[f"{bucket}:{p}"] = weight
        return s

    time_score = add_score("time", TIME_SIGNALS, weight=2)
    depth_score = add_score("depth", DEPTH_SIGNALS, weight=2)

    # Extra deterministic numeric heuristics:
    # Depth plots often contain many "26xx" style values; time plots often contain years.
    if re.search(r"\b26\d{2}\b", text):  # e.g., 2625, 2630 ...
        depth_score += 2
        scores["depth:depth_numbers_26xx"] = 2

    # If multiple year-like tokens exist, it strongly suggests a time axis.
    years = re.findall(r"\b20\d{2}\b", text)
    if len(set(years)) >= 2:
        time_score += 2
        scores["time:multiple_years"] = 2

    return time_score, depth_score, scores


def detect_plot_type(image_path: str | Path, *, debug: bool = False) -> DetectionResult:
    """
    Deterministic detector:
    - OCR whole image
    - Score against explicit signals
    - Choose using stable thresholds
    """
    image_path = Path(image_path)

    text = _ocr_text(image_path)
    time_score, depth_score, detailed = _score(text)

    margin = 3
    min_accept = 4

    if time_score >= depth_score + margin:
        chosen = PlotType.PRESSURE_VS_TIME
    elif depth_score >= time_score + margin:
        chosen = PlotType.PRESSURE_VS_DEPTH
    elif time_score >= min_accept and depth_score < min_accept:
        chosen = PlotType.PRESSURE_VS_TIME
    elif depth_score >= min_accept and time_score < min_accept:
        chosen = PlotType.PRESSURE_VS_DEPTH
    else:
        msg = (
            f"Ambiguous plot type for {image_path.name}. "
            f"time_score={time_score}, depth_score={depth_score}. "
            f"Top OCR text (first 250 chars): {text[:250]!r}"
        )
        raise PlotTypeDetectionError(msg)

    if debug:
        # Keep this deterministic & concise
        print(f"[plot-type] {image_path.name} -> {chosen.value} (time={time_score}, depth={depth_score})")

    return DetectionResult(
        plot_type=chosen,
        scores={"time_score": time_score, "depth_score": depth_score, **detailed},
        ocr_text=text,
    )

def detect_plot_type_bytes(image_bytes: bytes, *, debug: bool = False) -> DetectionResult:
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image bytes")

    text = _ocr_text_from_bgr(img)
    time_score, depth_score, detailed = _score(text)

    margin = 3
    min_accept = 4

    if time_score >= depth_score + margin:
        chosen = PlotType.PRESSURE_VS_TIME
    elif depth_score >= time_score + margin:
        chosen = PlotType.PRESSURE_VS_DEPTH
    elif time_score >= min_accept and depth_score < min_accept:
        chosen = PlotType.PRESSURE_VS_TIME
    elif depth_score >= min_accept and time_score < min_accept:
        chosen = PlotType.PRESSURE_VS_DEPTH
    else:
        raise PlotTypeDetectionError(
            f"Ambiguous plot type. time_score={time_score}, depth_score={depth_score}. "
            f"OCR (first 250): {text[:250]!r}"
        )

    if debug:
        print(f"[plot-type] <bytes> -> {chosen.value} (time={time_score}, depth={depth_score})")

    return DetectionResult(
        plot_type=chosen,
        scores={"time_score": time_score, "depth_score": depth_score, **detailed},
        ocr_text=text,
    )
