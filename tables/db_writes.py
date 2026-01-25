from __future__ import annotations

from sqlalchemy import text


SQL_UPSERT_DDR_DOCUMENT = text("""
INSERT INTO ddr_documents (
    source_filename,
    file_sha256,
    uploaded_at,
    wellbore_name,
    period_start,
    period_end
)
VALUES (:filename, :hash, NOW(), :wellbore_name, :period_start, :period_end)
ON CONFLICT (file_sha256)
DO UPDATE SET
    wellbore_name = COALESCE(EXCLUDED.wellbore_name, ddr_documents.wellbore_name),
    period_start  = COALESCE(EXCLUDED.period_start,  ddr_documents.period_start),
    period_end    = COALESCE(EXCLUDED.period_end,    ddr_documents.period_end)
RETURNING id;
""")


SQL_UPDATE_REPORT_NUMBER = text("""
UPDATE ddr_documents
SET report_number = :rn
WHERE id = :id
""")


SQL_UPSERT_ACTIVITY_SUMMARY = text("""
INSERT INTO ddr_activity_summary (
    document_id,
    activities_24h_text,
    planned_24h_text
)
VALUES (
    :document_id,
    :activities_24h_text,
    :planned_24h_text
)
ON CONFLICT (document_id)
DO UPDATE SET
    activities_24h_text = EXCLUDED.activities_24h_text,
    planned_24h_text    = EXCLUDED.planned_24h_text;
""")


def upsert_ddr_document(
    conn,
    *,
    filename: str,
    file_hash: str,
    wellbore_name: str | None,
    period_start: str | None,
    period_end: str | None,
) -> int:
    row = conn.execute(
        SQL_UPSERT_DDR_DOCUMENT,
        {
            "filename": filename,
            "hash": file_hash,
            "wellbore_name": wellbore_name,
            "period_start": period_start,
            "period_end": period_end,
        },
    ).fetchone()
    return int(row[0])


def update_report_number(conn, *, document_id: int, report_number: int) -> None:
    conn.execute(SQL_UPDATE_REPORT_NUMBER, {"rn": report_number, "id": document_id})


def upsert_ddr_activity_summary(
    conn,
    *,
    document_id: int,
    activities_24h_text: str,
    planned_24h_text: str,
) -> None:
    conn.execute(
        SQL_UPSERT_ACTIVITY_SUMMARY,
        {
            "document_id": document_id,
            "activities_24h_text": activities_24h_text,
            "planned_24h_text": planned_24h_text,
        },
    )
