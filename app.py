import streamlit as st
from pathlib import Path
import json
from services.tag_chat_orchestrator import run_tag_chat_turn
from actions.ingest_engineering_image import IngestEngineeringImageAction
from services.utils import AppServices, clear_dir_contents
from actions.truncate_db import TruncateDDRAction, TruncatePressureTimePlotsAction, TruncatePressureProfilePlotsAction
from model_code.process_ddr_model import ProcessDDRModel
from actions.ingest_ddr import IngestDDRAction
from services.ddr_analytics_fetch import list_ddr_documents, fetch_ddr_payload
from AI.gemini_ddr_analytics import run_ddr_analytics
from reports.ddr_analytics_pdf import build_pdf_bytes
from services.ddr_trends_fetch import fetch_daily_ops_metrics
import pandas as pd

GEMINI_MODEL = "gemini-2.5-flash"

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
        "DDR Analytics",   # NEW
    ],
)

# ------------------------------------------------------------------
# Sidebar - model settings (DDR only)
# ------------------------------------------------------------------
custom_weights_path = None
keep_processed_files = False

if processing_mode == "DDR PDF Report":
    st.sidebar.subheader("Model Settings (DDR reports)")

    custom_weights_path = st.sidebar.text_input(
        "Custom model weights path:",
        value="models\\custom\\best.pt"
    )

    keep_processed_files = st.sidebar.checkbox(
        "Keep processed_ddr files (debug)",
        value=False,
        help="If OFF, processed_ddr will be cleared automatically after a successful Process DDR."
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

        if st.button("Process DDR", key="run_ddr_model_btn"):
            status_box = st.empty()
            out_dir = Path("processed_ddr")
            out_dir.mkdir(exist_ok=True)

            if not keep_processed_files:
                clear_dir_contents(out_dir)

            try:
                process_model = ProcessDDRModel(
                    model_choice="custom",
                    custom_weights=custom_weights_path
                )

                status_box.info("Running DDR structure detection using YOLO model...")
                total_pages, total_crops = process_model.process_pdf(uploaded_pdf, out_dir)
                status_box.success(f"YOLO finished: {total_pages} pages, {total_crops} crops.")

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

                # ✅ Auto cleanup AFTER success
                if not keep_processed_files:
                    clear_dir_contents(out_dir)
                    st.info("processed_ddr cleared (auto-clean).")

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
# DDR ANALYTICS (SQL -> Gemini -> PDF)  [UI ONLY]
# ------------------------------------------------------------------

if processing_mode == "DDR Analytics":
    st.subheader("DDR Analytics (PDF only)")
    st.markdown(
        "- Select an already ingested DDR using `document_id`\n")

    document_id = 0
    docs_df = list_ddr_documents(engine, limit=200)

    if docs_df.empty:
        st.warning("No DDR documents found in ddr_documents.")
        st.stop()
    else:
        docs_df["label"] = docs_df.apply(
            lambda r: f"#{r['id']} | {r.get('wellbore_name','')} | {r.get('period_start','')} → {r.get('period_end','')} | {r.get('source_filename','')}",
            axis=1
        )

        selected_label = st.selectbox("Select document", docs_df["label"].tolist())
        document_id = int(selected_label.split("|")[0].replace("#", "").strip())
    st.text_input("Gemini model", value=GEMINI_MODEL, disabled=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("Run Analytics & Generate PDF", key="run_ddr_analytics_btn")
    with col2:
        st.caption("This will: fetch SQL rows → call Gemini → build reportlab PDF → provide download.")

    if run_btn:
        with st.spinner("Fetching structured data from SQL..."):
            payload = fetch_ddr_payload(engine, document_id)

        st.success("SQL payload fetched.")

        with st.spinner("Running Gemini analytics..."):
            analytics = run_ddr_analytics(payload, model_name=GEMINI_MODEL)

        st.success("Gemini analytics complete.")

        pdf_bytes = build_pdf_bytes(payload, analytics)

        st.download_button(
            "Download Analytics PDF",
            data=pdf_bytes,
            file_name=f"ddr_analytics_{document_id}.pdf",
            mime="application/pdf",
        )

    st.divider()
    st.subheader("TAG Chatbot (SQL-grounded, read-only)")

    # Chat state
    if "tag_messages" not in st.session_state:
        st.session_state.tag_messages = []  # list of dicts: {"role": "user"/"assistant", "content": dict}

    # Context selectors (document_id + wellbore_name + date)
    # We already have document_id from dropdown above.
    # We'll also allow optional override filters.
    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        chat_document_id = st.number_input(
            "document_id (optional override)",
            value=int(document_id),
            min_value=0,
            step=1,
            help="If 0, router will infer from question or use selected document."
        )

    with colB:
        chat_wellbore_name = st.text_input(
            "wellbore_name (optional)",
            value="",
            help="Optional: used when question references wellbore."
        )

    with colC:
        chat_day = st.date_input(
            "day (optional)",
            value=None,
            help="Optional: used when question references a day. If empty, router may infer or ask."
        )

    # Show chat history
    for msg in st.session_state.tag_messages:
        role = msg["role"]
        content = msg["content"]  # dict (STRICT JSON response or user payload)
        with st.chat_message(role):
            if role == "user":
                st.markdown(content["text"])
            else:
                # assistant content is strict JSON
                st.markdown(content.get("answer", ""))

                with st.expander("SQL used"):
                    st.json(content.get("sql_used", []))

                with st.expander("Tables preview"):
                    st.json(content.get("tables_preview", {}))

                limits = content.get("assumptions_or_limits", [])
                if limits:
                    with st.expander("Assumptions / limits"):
                        st.write(limits)

                followups = content.get("followups", [])
                if followups:
                    st.markdown("**Follow-ups:**")
                    for f in followups:
                        st.caption(f"- {f}")

    # User input
    user_text = st.chat_input("Ask about DDR operations, downtime, failures, gas readings...")

    if user_text:
        # Append user message
        st.session_state.tag_messages.append({
            "role": "user",
            "content": {"text": user_text}
        })

        # Resolve overrides (keep UI-only)
        doc_id_override = int(chat_document_id) if chat_document_id else None
        wellbore_override = chat_wellbore_name.strip() or None
        day_override = chat_day.isoformat() if chat_day else None  # pass as ISO date string

        with st.spinner("Running TAG: routing → SQL → Gemini..."):
            try:
                # Single orchestration call (backend module will do router + safe SQL + Gemini)
                response_json = run_tag_chat_turn(
                    engine=engine,
                    question=user_text,
                    document_id=doc_id_override,
                    wellbore_name=wellbore_override,
                    day=day_override,
                    model_name=GEMINI_MODEL,
                )

            except Exception as e:
                response_json = {
                    "answer": f"Error: {type(e).__name__}: {str(e)}",
                    "sql_used": [],
                    "tables_preview": {},
                    "assumptions_or_limits": ["Backend exception occurred. Check logs/traceback."],
                    "followups": []
                }

        # Append assistant message (strict JSON)
        st.session_state.tag_messages.append({
            "role": "assistant",
            "content": response_json
        })

        st.rerun()

# ------------------------------------------------------------------
# DDR Trends (Operations)
# ------------------------------------------------------------------
if processing_mode == "DDR Analytics":
    wellbore = st.text_input("Filter wellbore_name (optional)", value="")

    df = fetch_daily_ops_metrics(engine, wellbore_name=wellbore.strip() or None)

    if df.empty:
        st.warning("No data for this filter.")
    else:
        # Ensure day is datetime for charts
        df["day"] = pd.to_datetime(df["day"])

        st.markdown("### Reliability / friction")
        st.line_chart(
            df.set_index("day")[["fail_ops", "fail_hours", "total_ops"]]
        )

        st.markdown("### NPT proxy (repair + interruption hours)")
        df["npt_hours"] = (df["repair_hours"].fillna(0) + df["interruption_hours"].fillna(0))
        st.line_chart(df.set_index("day")[["npt_hours"]])

        st.dataframe(df)
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
