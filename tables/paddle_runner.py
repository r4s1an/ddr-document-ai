import os
from pathlib import Path
import streamlit as st

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

@st.cache_resource(show_spinner=False)
def get_paddle_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

def run_paddle_ocr(table_image: str | Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = get_paddle_ocr()
    result = ocr.predict(input=str(table_image))

    for res in result:
        res.save_to_json(out_dir)

    json_files = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        raise RuntimeError(f"No JSON saved in {out_dir}")

    return json_files[0]