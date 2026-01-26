import streamlit as st
from pathlib import Path
import shutil
from actions.ingest_engineering_image import IngestEngineeringImageAction
from services.utils import AppServices
from actions.truncate_db import TruncateDDRAction, TruncatePressureTimePlotsAction, TruncatePressureProfilePlotsAction
from model_code.process_ddr_model import ProcessDDRModel
from actions.ingest_ddr import IngestDDRAction

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

st.set_page_config(page_title="DDR & Drilling Analytics Engine", layout="wide")
st.title("DDR & Drilling Analytics Engine")
st.markdown("##### Intelligent extraction and analysis for reports and graphical data.")

services = AppServices()
engine = services.get_engine()

# ------------------------------------------------------------------
# Processing mode selector
# ------------------------------------------------------------------

processing_mode = st.radio(
    "Select processing mode",
    options=[
        "DDR PDF Report",
        "Engineering Image",
    ],
)

# ------------------------------------------------------------------
# Sidebar - model settings (DDR only)
# ------------------------------------------------------------------

st.sidebar.subheader("Model Settings (DDR reports)")
custom_weights_path = st.sidebar.text_input(
    "Custom model weights path:",
    value="models\\custom\\best.pt"
)

# ------------------------------------------------------------------
# DDR PDF PIPELINE
# ------------------------------------------------------------------

if processing_mode == "DDR PDF Report":
    st.subheader("Upload DDR PDF Report")

    uploaded_pdf = st.file_uploader(
        "Upload a DDR PDF file",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        key="ddr_pdf_uploader",
    )

    if uploaded_pdf:
        file_bytes = uploaded_pdf.getvalue()
        file_hash = services.sha256_bytes(file_bytes)
        st.success(f"File '{uploaded_pdf.name}' loaded into memory.")

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
                total_pages, total_crops = process_model.process_pdf(
                    uploaded_pdf,
                    out_dir
                )
                status_box.success(
                    f"YOLO finished: {total_pages} pages, {total_crops} crops."
                )

                status_box.info("Running layout + OCR + metadata extraction + DB insert...")
                ingest = IngestDDRAction(engine, ocr_gpu=True)
                result = ingest.execute(
                    filename=uploaded_pdf.name,
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

# ------------------------------------------------------------------
# ENGINEERING IMAGE PIPELINE (ENTRY POINT ONLY)
# ------------------------------------------------------------------

if processing_mode == "Engineering Image":
    st.subheader("Upload Engineering Image")

    uploaded_image = st.file_uploader(
        "Upload an image (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="image_uploader",
    )

    if uploaded_image:
        st.success(f"Image '{uploaded_image.name}' loaded into memory.")

        st.info(
            "Supported image workflows:\n"
            "- Pressure vs Time (Offset Wells)\n"
            "- Pressure vs Depth (Pressure Profile)"
        )

        if st.button("Process Image", key="process_image_btn"):
            try:
                image_bytes = uploaded_image.getvalue()
                image_name = uploaded_image.name

                # Placeholder: backend image pipeline entry point
                action = IngestEngineeringImageAction(engine, debug=True)
                mime_type = uploaded_image.type  # "image/png" or "image/jpeg"
                source_key = services.sha256_bytes(image_bytes)  # deterministic re-ingest key

                result = action.execute(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    source_key=source_key,
                )

                st.success(f"Done. Plot type = {result.plot_type}")
                st.success(f"Inserted plot_id = {result.plot_id}")

            except Exception as e:
                st.error("Failed during image processing.")
                st.exception(e)

# ------------------------------------------------------------------
# Maintenance - processed files
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# Maintenance - database
# ------------------------------------------------------------------

st.divider()
st.subheader("Database maintenance")

confirm_truncate = st.checkbox(
    "I understand this will permanently delete ALL rows in ddr_documents."
)

if st.button("TRUNCATE ddr_documents", key="truncate_ddr_btn"):
    if not confirm_truncate:
        st.warning("Tick the confirmation checkbox first.")
    else:
        TruncateDDRAction(engine).execute()
        st.success("ddr_documents truncated (IDs reset)")

st.divider()
st.subheader("Image database maintenance")

confirm_truncate_images = st.checkbox(
    "I understand this will permanently delete ALL image-based plot data."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("TRUNCATE pressure vs time plots"):
        if not confirm_truncate_images:
            st.warning("Tick the confirmation checkbox first.")
        else:
            TruncatePressureTimePlotsAction(engine).execute()
            st.success("Pressure vs Time plot tables truncated.")

with col2:
    if st.button("TRUNCATE pressure profile plots"):
        if not confirm_truncate_images:
            st.warning("Tick the confirmation checkbox first.")
        else:
            TruncatePressureProfilePlotsAction(engine).execute()
            st.success("Pressure Profile plot tables truncated.")
