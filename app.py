import streamlit as st
from services.utils import AppServices
from actions.process_ddr import ProcessDDRAction
from actions.truncate_db import TruncateDDRAction
from actions.extract_ddr import ExtractDDRAction


st.title("DDR & Drilling Analytics Engine")
st.markdown("##### Intelligent extraction and analysis for reports and graphical data.")

services = AppServices()
engine = services.get_engine()


def run_step(status_box, start_msg, ok_msg, fn):
    status_box.info(start_msg)
    out = fn()
    status_box.success(ok_msg)
    return out


uploaded_file = st.file_uploader(
    "Insert a file (PDF/DOCX for DDRs, PNG/JPG for Graphs)",
    type=["pdf", "docx", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    file_bytes = uploaded_file.getvalue()
    file_hash = services.sha256_bytes(file_bytes)

    st.success(f"File '{uploaded_file.name}' loaded into memory!")

    if file_extension in ["pdf", "docx"]:

        if st.button("Save DDR to Database", key="save_ddr_btn"):
            status_box = st.empty()

            def ui_log(level: str, msg: str):
                if level == "info":
                    status_box.info(msg)
                elif level == "success":
                    status_box.success(msg)
                elif level == "warning":
                    status_box.warning(msg)
                else:
                    status_box.write(msg)

            try:
                action = ProcessDDRAction(engine)
                result = action.execute(
                    filename=uploaded_file.name,
                    file_hash=file_hash,
                    file_bytes=file_bytes,
                    debug=False,
                    log=ui_log,
                )

                if result.status == "created":
                    st.success(f"All operations completed successfully (document_id={result.document_id})")
                else:
                    st.info(f"Nothing to do (already in DB). document_id={result.document_id}")

            except Exception as e:
                st.error("Failed during DDR processing. No data was saved.")
                st.exception(e)

    if uploaded_file and st.button("Run DDR Text Extraction", key="extract_ddr_btn"):
        ExtractDDRAction()
        st.write("Next step: extract sections and tables from the stored PDF...")


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
