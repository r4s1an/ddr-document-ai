import os
from pathlib import Path
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from doclayout_yolo import YOLOv10
from typing import List, Tuple
import json
import tempfile

POPPLER_BIN = r"C:\Users\Yoked\Desktop\DDR Processor\poppler-25.12.0\Library\bin"
DPI = 320
DEVICE = "0"
CONF = 0.15
IMGSZ = 896
CROP_LABELS = {
    "table",
    "figure",
    "plain_text", 
    "section_header",
    "wellbore_field",
    "period_field", 
}
PRETRAINED_WEIGHTS = Path("models/doclayout_yolo_docstructbench_imgsz1024.pt")

Box = Tuple[int, int, int, int]   

def _area(b: Box) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def _intersection(a: Box, b: Box) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)

def _iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0

def _inside_ratio(inner: Box, outer: Box) -> float:
    inter = _intersection(inner, outer)
    ain = _area(inner)
    return (inter / ain) if ain > 0 else 0.0

def dedupe_by_label(
    dets: List[dict],
    *,
    iou_thresh: float = 0.85,
    inside_thresh: float = 0.90,
) -> List[dict]:
    """
    Deduplicate detections by label, preferring larger boxes when overlapping.
    
    Args:
        dets: list of {"i": int, "label": str, "conf": float, "box": (x1,y1,x2,y2)}
        iou_thresh: IoU threshold for considering boxes as duplicates
        inside_thresh: Threshold for "mostly inside" detection
    
    Returns:
        Filtered list with duplicates removed
    """
    out: List[dict] = []
    
     
    by_label: dict[str, List[dict]] = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d)

    for label, group in by_label.items():
         
        group = sorted(group, key=lambda d: (-_area(d["box"]), -d["conf"]))
        
        kept: List[dict] = []

        for d in group:
            drop = False
            
            for k in kept:
                 
                inside_ratio = _inside_ratio(d["box"], k["box"])
                if inside_ratio >= inside_thresh:
                    drop = True
                    break
                
                 
                 
                inside_ratio_rev = _inside_ratio(k["box"], d["box"])
                if inside_ratio_rev >= inside_thresh:
                     
                    kept.remove(k)
                    continue
                
                 
                iou = _iou(d["box"], k["box"])
                if iou >= iou_thresh:
                     
                    drop = True
                    break
            
            if not drop:
                kept.append(d)

        out.extend(kept)

     
    out = sorted(out, key=lambda d: (d["box"][1], d["box"][0]))
    return out

class ProcessDDRModel:
    def __init__(self, model_choice="pretrained", custom_weights=None):
        self.model_choice = model_choice
        self.custom_weights = custom_weights or PRETRAINED_WEIGHTS
        self.model = self._ensure_model()

    def _get_weights_path(self):
        if self.model_choice == "pretrained":
            from huggingface_hub import hf_hub_download
            PRETRAINED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            return hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                cache_dir=str(PRETRAINED_WEIGHTS.parent),
            )
        elif self.model_choice == "custom":
            if not Path(self.custom_weights).exists():
                raise FileNotFoundError(f"Custom weights not found: {self.custom_weights}")
            return str(self.custom_weights)
        else:
            raise ValueError("model_choice must be 'pretrained' or 'custom'")

    def _ensure_model(self):
        weights_path = self._get_weights_path()
        return YOLOv10(weights_path)

    def _render_all_pages(self, pdf_bytes: bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        try:
            images = convert_from_path(tmp_path, dpi=DPI, poppler_path=POPPLER_BIN)
        finally:
             
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        if not images:
            raise RuntimeError("PDF rendering failed.")
        return images

    def process_pdf(self, uploaded_file, out_dir: Path):
        images = self._render_all_pages(uploaded_file.getvalue())
        total_pages = len(images)
        total_crops = 0

        for idx, img_pil in enumerate(images, start=1):
            page_dir = out_dir / f"page_{idx:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            image_path = page_dir / f"page_{idx:03d}.png"
            annotated_path = page_dir / f"page_{idx:03d}_annotated.png"
            crops_dir = page_dir / "crops"
            crops_dir.mkdir(exist_ok=True)

            img_pil.save(image_path)
            results = self.model.predict(str(image_path), imgsz=IMGSZ, conf=CONF, device=DEVICE,
                                        agnostic_nms=True)
            
            annotated = results[0].plot(pil=False, line_width=4, font_size=16)
            cv2.imwrite(str(annotated_path), annotated)

            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            boxes = results[0].boxes
            names = results[0].names

             
            dets = []
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = names[cls_id]
                score = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                dets.append({"i": i, "label": label, "conf": score, "box": (x1, y1, x2, y2)})

             
            dets_before = len(dets)
            dets = dedupe_by_label(dets, iou_thresh=0.85, inside_thresh=0.90)
            dets_after = len(dets)

             
            for d in dets:
                label = d["label"]
                if label not in CROP_LABELS:
                    continue

                x1, y1, x2, y2 = d["box"]
                crop = img[y1:y2, x1:x2]

                 
                crop_path = crops_dir / f"{d['i']:03d}_{label.replace(' ', '_')}.png"
                cv2.imwrite(str(crop_path), crop)
                total_crops += 1

             
            detections_txt = page_dir / f"page_{idx:03d}_detections.txt"
            lines = [
                f"Image: {image_path}", 
                f"Detected regions (after deduplication): {len(dets)}",
                f"Original detections: {dets_before}",
                f"Removed duplicates: {dets_before - dets_after}",
                ""
            ]
            for d in dets:
                area = _area(d["box"])
                lines.append(
                    f"[{d['i']:03d}] {d['label']:18s} conf={d['conf']:.3f} "
                    f"bbox={d['box']} area={area:,}"
                )

            detections_txt.with_suffix(".json").write_text(
                json.dumps(dets, ensure_ascii=False, indent=2),
                encoding="utf-8")

        return total_pages, total_crops