import streamlit as st
from services.utils import AppServices
from actions.save_ddr import SaveDDRAction, SaveDDRSummaryAction
from actions.truncate_db import TruncateDDRAction
from actions.extract_ddr import ExtractDDRAction

st.title("DDR & Drilling Analytics Engine")
st.markdown("##### Intelligent extraction and analysis for reports and graphical data.")

services = AppServices()
engine = services.get_engine()

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

        if st.button("Save DDR to Database"):

            save_doc = SaveDDRAction(engine)
            save_summary = SaveDDRSummaryAction(engine)

            status_box = st.empty()

            try:
                status_box.info("Saving DDR document...")

                with engine.begin() as conn:

                    result = save_doc.execute_with_conn(
                        conn=conn,
                        filename=uploaded_file.name,
                        file_hash=file_hash,
                        file_bytes=file_bytes,
                    )

                    status_box.success("DDR document saved.")

                    status_box.info("Extracting and saving summary report...")

                    save_summary.execute(
                        conn=conn,
                        document_id=result["document_id"],
                        file_bytes=file_bytes,
                    )

                    status_box.success("Summary report saved.")

                st.success(
                    f"All operations completed successfully "
                    f"(document_id={result['document_id']})"
                )

            except Exception as e:
                st.error("Failed during DDR processing. No data was saved.")
                st.exception(e)


        if st.button("Run DDR Text Extraction"):
            action = ExtractDDRAction()
            st.write("Next step: extract sections and tables from the stored PDF...")

st.divider()
st.subheader("Database maintenance")

confirm_truncate = st.checkbox(
    "I understand this will permanently delete ALL rows in ddr_documents."
)

if st.button("TRUNCATE ddr_documents"):
    if not confirm_truncate:
        st.warning("Tick the confirmation checkbox first.")
    else:
        TruncateDDRAction(engine).execute()
        st.success("ddr_documents truncated (IDs reset)")
