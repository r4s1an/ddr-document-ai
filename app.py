import streamlit as st
import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from services.utils import AppServices

st.title("DDR & Drilling Analytics Engine")
st.markdown("##### Intelligent extraction and analysis for reports and graphical data.")

services = AppServices()

uploaded_file = st.file_uploader(
    "Insert a file (PDF/DOCX for DDRs, PNG/JPG for Graphs)",
    type=["pdf", "docx", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    # Read file bytes directly (no saving locally)
    file_bytes = uploaded_file.getvalue()
    file_hash = services.sha256_bytes(file_bytes)

    st.success(f"File '{uploaded_file.name}' loaded into memory!")
    st.write({"sha256": file_hash, "size_bytes": len(file_bytes)})

    if file_extension in ["pdf", "docx"]:
        st.info("Detected Type: **Document (DDR)**")

        if st.button("Save DDR to Database", key="save_ddr"):
            engine = services.get_engine()

            # Insert, or if duplicate hash exists, fetch existing id
            insert_sql = text("""
                INSERT INTO ddr_documents (source_filename, file_sha256)
                VALUES (:name, :hash)
                RETURNING id;
            """)

            select_sql = text("""
                SELECT id FROM ddr_documents WHERE file_sha256 = :hash;
            """)

            try:
                with engine.begin() as conn:
                    doc_id = conn.execute(insert_sql, {
                        "name": uploaded_file.name,
                        "hash": file_hash,
                    }).scalar_one()

                st.success(f"Saved to DB ✅ document_id = {doc_id}")

            except IntegrityError:
                # Hash already exists -> reuse existing record
                with engine.begin() as conn:
                    doc_id = conn.execute(select_sql, {"hash": file_hash}).scalar_one()
                st.warning(f"Duplicate document (same SHA-256). Using existing document_id = {doc_id}")

            except Exception as e:
                st.error(f"DB error: {e}")

        # Keep your next step button for later parsing
        if st.button("Run DDR Text Extraction", key="extract_ddr"):
            st.write("Next step: extract sections and tables from the stored PDF...")

    else:
        st.info("Detected Type: **Graphical Data** (we'll handle later)")

st.divider()
st.subheader("Database maintenance")

confirm_truncate = st.checkbox(
    "I understand this will permanently delete ALL rows in ddr_documents.",
    key="confirm_truncate"
)

if st.button("TRUNCATE ddr_documents", key="truncate_ddr"):
    if not confirm_truncate:
        st.warning("Tick the confirmation checkbox first.")
    else:
        try:
            engine = services.get_engine()
            with engine.begin() as conn:
                conn.execute(text("TRUNCATE TABLE ddr_documents RESTART IDENTITY;"))
            st.success("ddr_documents truncated ✅ (IDs reset)")
        except Exception as e:
            st.error(f"DB error: {e}")