import os
from pathlib import Path
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from doclayout_yolo import YOLOv10

POPPLER_BIN = r"C:\Users\Yoked\Desktop\DDR Processor\poppler-25.12.0\Library\bin"
DPI = 320
DEVICE = "0"
CONF = 0.2
IMGSZ = 896
CROP_LABELS = {
    "table", "figure", "title", "plain_text",
    'table_footnote', "text", "header", "section_title"
}
PRETRAINED_WEIGHTS = Path("models/doclayout_yolo_docstructbench_imgsz1024.pt")
CUSTOM_WEIGHTS = Path(r"C:\Users\Yoked\Desktop\DDR Processor\runs\detect\DDR_Phase1_Frozen\weights\last.pt")

class ProcessDDRModel:
    def __init__(self, model_choice="pretrained", custom_weights=None):
        self.model_choice = model_choice
        self.custom_weights = custom_weights or CUSTOM_WEIGHTS
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
        temp_pdf = Path("temp_uploaded.pdf")
        temp_pdf.write_bytes(pdf_bytes)
        images = convert_from_path(str(temp_pdf), dpi=DPI, poppler_path=POPPLER_BIN)
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

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = names[cls_id]
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                if label in CROP_LABELS:
                    crop = img[y1:y2, x1:x2]
                    crop_path = crops_dir / f"{i:03d}_{label.replace(' ', '_')}.png"
                    cv2.imwrite(str(crop_path), crop)
                    total_crops += 1
            
            # save detections.txt for this page
            detections_txt = page_dir / f"page_{idx:03d}_detections.txt"
            lines = [f"Image: {image_path}", f"Detected regions: {len(boxes)}", ""]
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = names[cls_id]
                score = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                lines.append(f"[{i:03d}] {label:18s} conf={score:.3f} bbox=({x1},{y1},{x2},{y2})")

            detections_txt.write_text("\n".join(lines), encoding="utf-8")

        return total_pages, total_crops
