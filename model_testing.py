import os
from pathlib import Path
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from doclayout_yolo import YOLOv10

PDF_PATH = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_19_A_1980_01_01.pdf"
POPPLER_BIN = r"C:\Users\Yoked\Desktop\DDR Processor\poppler-25.12.0\Library\bin"
DPI = 320
DEVICE = "0"
CONF = 0.2
IMGSZ = 896
CROP_LABELS = {
    "table", 'abandon'}
# "figure", "title", "plain_text", 'table_footnote', "text", "header", "section_title"
OUT_DIR = Path("testing")

# -----------------------------
# MODEL OPTIONS
# -----------------------------
MODEL_CHOICE = "custom"  # "pretrained" or "custom"

PRETRAINED_WEIGHTS = Path("models/doclayout_yolo_docstructbench_imgsz1024.pt")
CUSTOM_WEIGHTS = Path(r"models\custom\best.pt")
# -----------------------------

def get_weights_path():
    if MODEL_CHOICE == "pretrained":
        from huggingface_hub import hf_hub_download
        PRETRAINED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            cache_dir=str(PRETRAINED_WEIGHTS.parent),
        )
    elif MODEL_CHOICE == "custom":
        if not CUSTOM_WEIGHTS.exists():
            raise FileNotFoundError(f"Custom weights not found: {CUSTOM_WEIGHTS}")
        return str(CUSTOM_WEIGHTS)
    else:
        raise ValueError("MODEL_CHOICE must be 'pretrained' or 'custom'")

def get_pdf_page_count(pdf_path, poppler_bin):
    info = pdfinfo_from_path(pdf_path, poppler_path=poppler_bin)
    return int(info["Pages"])

def render_all_pages(pdf_path: str, dpi: int, poppler_bin: str | None):
    print(f"Rendering all pages from {pdf_path}...")
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_bin)
    if not images:
        raise RuntimeError("PDF rendering failed. Check Poppler path and PDF file.")
    return images

def ensure_model():
    weights_path = get_weights_path()
    model = YOLOv10(weights_path)
    return model

def run_detection(model, image_pil, page_idx, page_dir):
    page_dir.mkdir(parents=True, exist_ok=True)
    image_path = page_dir / f"page_{page_idx:03d}.png"
    annotated_path = page_dir / f"page_{page_idx:03d}_annotated.png"
    crops_dir = page_dir / "crops"
    detections_txt = page_dir / f"page_{page_idx:03d}_detections.txt"

    image_pil.save(image_path)
    results = model.predict(str(image_path), imgsz=IMGSZ, conf=CONF, device=DEVICE,
                            agnostic_nms=True)
    
    annotated = results[0].plot(pil=False, line_width=4, font_size=16)
    cv2.imwrite(str(annotated_path), annotated)

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    boxes = results[0].boxes
    names = results[0].names
    lines = [f"Image: {image_path}", f"Detected regions: {len(boxes)}", ""]
    saved = []

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        label = names[cls_id]
        score = float(boxes.conf[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        lines.append(f"[{i:03d}] {label:18s} conf={score:.3f} bbox=({x1},{y1},{x2},{y2})")

        if label in CROP_LABELS:
            crops_dir.mkdir(exist_ok=True)
            crop = img[y1:y2, x1:x2]
            crop_path = crops_dir / f"{i:03d}_{label.replace(' ', '_')}.png"
            cv2.imwrite(str(crop_path), crop)
            saved.append((label, score, (x1, y1, x2, y2)))

    detections_txt.write_text("\n".join(lines), encoding="utf-8")
    return len(boxes), len(saved)

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"1) Loading model ({MODEL_CHOICE})...")
    model = ensure_model()

    print("2) Rendering PDF...")
    images = render_all_pages(PDF_PATH, DPI, POPPLER_BIN)
    total_pages = len(images)
    print(f"✅ Found {total_pages} pages.")

    print("3) Running detection on all pages...")   
    for idx, img_pil in enumerate(images, start=1):
        print(f"   --- Processing Page {idx}/{total_pages} ---")
        page_folder = OUT_DIR / f"page_{idx:03d}"
        num_det, num_saved = run_detection(model, img_pil, idx, page_folder)
        print(f"   ✅ Detected {num_det} regions, saved {num_saved} crops.")

    print(f"\n✅ All pages processed. Results in: {OUT_DIR}")

if __name__ == "__main__":
    main()
