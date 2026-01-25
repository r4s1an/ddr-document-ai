import streamlit as st
from pathlib import Path
from services.utils import AppServices
from actions.truncate_db import TruncateDDRAction
from model_code.process_ddr_model import ProcessDDRModel
from actions.ingest_ddr import IngestDDRAction
import shutil

st.set_page_config(page_title="DDR & Drilling Analytics Engine", layout="wide")
st.title("DDR & Drilling Analytics Engine")
st.markdown("##### Intelligent extraction and analysis for reports and graphical data.")

services = AppServices()
engine = services.get_engine()

# -----------------------------
# Model Settings UI
# -----------------------------
st.sidebar.subheader("Model Settings")
custom_weights_path = st.sidebar.text_input(
    "Custom model weights path:",
    value="models\\custom\\best.pt"
)
# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Insert a file (PDF/DOCX for DDRs, PNG/JPG for Graphs)",
    type=["pdf", "docx", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    file_bytes = uploaded_file.getvalue()
    file_hash = services.sha256_bytes(file_bytes)
    st.success(f"File '{uploaded_file.name}' loaded into memory!")

    if st.button("Run DDR Model", key="run_ddr_model_btn"):
        status_box = st.empty()
        try:
            out_dir = Path("processed_ddr")
            out_dir.mkdir(exist_ok=True)

            process_model = ProcessDDRModel(
                model_choice="custom",
                custom_weights=custom_weights_path
            )

            status_box.info("Running DDR structure detection using YOLO model...")
            total_pages, total_crops = process_model.process_pdf(uploaded_file, out_dir)
            status_box.success(f"YOLO finished: {total_pages} pages, {total_crops} crops.")
            st.success(f"Results saved in: {out_dir.resolve()}")

            status_box.info("Running layout + OCR + metadata extraction + DB insert...")
            ingest = IngestDDRAction(engine, ocr_gpu=True)
            result = ingest.execute(
                filename=uploaded_file.name,
                file_hash=file_hash,
                out_dir=out_dir
            )

            status_box.success(f"Done. ddr_documents.id = {result.document_id}")
            st.write("Wellbore:", result.wellbore_name)
            st.write("Period:", result.period_start, "→", result.period_end)
            st.write("Metadata found on:", result.used_page)

        except Exception as e:
            st.error("Failed during DDR processing.")
            st.exception(e)

# -----------------------------
# Database Maintenance
# -----------------------------
st.divider()
st.subheader("Processed files maintenance")

confirm_delete_processed = st.checkbox(
    "I understand this will permanently delete ALL files in processed_ddr."
)

if st.button("DELETE processed_ddr contents", key="delete_processed_btn"):
    if not confirm_delete_processed:
        st.warning("Tick the confirmation checkbox first.")
    else:
        processed_dir = Path("processed_ddr")
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
            processed_dir.mkdir(exist_ok=True)
            st.success("processed_ddr directory cleared.")
        else:
            st.info("processed_ddr directory does not exist.")
            
st.divider()
st.subheader("Database maintenance")
confirm_truncate = st.checkbox("I understand this will permanently delete ALL rows in ddr_documents.")

if st.button("TRUNCATE ddr_documents", key="truncate_ddr_btn"):
    if not confirm_truncate:
        st.warning("Tick the confirmation checkbox first.")
    else:
        TruncateDDRAction(engine).execute()
        st.success("ddr_documents truncated (IDs reset)")
