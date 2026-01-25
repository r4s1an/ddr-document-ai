from __future__ import annotations

from sqlalchemy import text
from typing import Any, Dict, List


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

SQL_DELETE_DRILLING_FLUID = text("""
DELETE FROM ddr_drilling_fluid
WHERE document_id = :document_id
""")

SQL_INSERT_DRILLING_FLUID = text("""
INSERT INTO ddr_drilling_fluid (
    document_id,
    sample_time,
    sample_point,
    sample_depth_mmd,
    fluid_type,
    fluid_density_g_cm3,
    funnel_visc_s,
    plastic_visc_mpas,
    yield_point_pa,
    test_temp_hpht_degc
)
VALUES (
    :document_id,
    :sample_time,
    :sample_point,
    :sample_depth_mmd,
    :fluid_type,
    :fluid_density_g_cm3,
    :funnel_visc_s,
    :plastic_visc_mpas,
    :yield_point_pa,
    :test_temp_hpht_degc
)
""")

SQL_DELETE_OPERATIONS = text("""
DELETE FROM ddr_operations
WHERE document_id = :document_id
""")

SQL_INSERT_OPERATIONS = text("""
INSERT INTO ddr_operations (
    document_id,
    start_time,
    end_time,
    end_depth_mmd,
    main_activity,
    sub_activity,
    state,
    remark
)
VALUES (
    :document_id,
    :start_time,
    :end_time,
    :end_depth_mmd,
    :main_activity,
    :sub_activity,
    :state,
    :remark
)
""")

SQL_DELETE_SURVEY_STATION = text("""
DELETE FROM ddr_survey_station
WHERE document_id = :document_id
""")

SQL_INSERT_SURVEY_STATION = text("""
INSERT INTO ddr_survey_station (
    document_id,
    depth_mmd,
    depth_mtvd,
    inclination_dega,
    azimuth_dega,
    comment
)
VALUES (
    :document_id,
    :depth_mmd,
    :depth_mtvd,
    :inclination_dega,
    :azimuth_dega,
    :comment
)
""")

SQL_DELETE_STRATIGRAPHIC_INFORMATION = text("""
DELETE FROM ddr_stratigraphic_information
WHERE document_id = :document_id
""")

SQL_INSERT_STRATIGRAPHIC_INFORMATION = text("""
INSERT INTO ddr_stratigraphic_information (
    document_id,
    depth_top_formation_mmd,
    depth_top_formation_mtvd,
    description
)
VALUES (
    :document_id,
    :depth_top_formation_mmd,
    :depth_top_formation_mtvd,
    :description
)
""")

SQL_DELETE_LITHOLOGY_INFORMATION = text("""
DELETE FROM ddr_lithology_information
WHERE document_id = :document_id
""")

SQL_INSERT_LITHOLOGY_INFORMATION = text("""
INSERT INTO ddr_lithology_information (
    document_id,
    start_depth_mmd,
    end_depth_mmd,
    start_depth_mtvd,
    end_depth_mtvd,
    shows_description,
    lithology_description
)
VALUES (
    :document_id,
    :start_depth_mmd,
    :end_depth_mmd,
    :start_depth_mtvd,
    :end_depth_mtvd,
    :shows_description,
    :lithology_description
)
""")

SQL_DELETE_GAS_READING_INFORMATION = text("""
DELETE FROM ddr_gas_reading_information
WHERE document_id = :document_id
""")

SQL_INSERT_GAS_READING_INFORMATION = text("""
INSERT INTO ddr_gas_reading_information (
    document_id,
    time,
    class,
    depth_to_top_mmd,
    depth_to_bottom_md,
    depth_to_top_mtvd,
    depth_to_bottom_tvd,
    highest_gas_percent,
    lowest_gas_percent,
    c1_ppm,
    c2_ppm,
    c3_ppm,
    ic4_ppm,
    ic5_ppm
)
VALUES (
    :document_id,
    :time,
    :class,
    :depth_to_top_mmd,
    :depth_to_bottom_md,
    :depth_to_top_mtvd,
    :depth_to_bottom_tvd,
    :highest_gas_percent,
    :lowest_gas_percent,
    :c1_ppm,
    :c2_ppm,
    :c3_ppm,
    :ic4_ppm,
    :ic5_ppm
)
""")

def replace_ddr_drilling_fluid_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    # idempotent: delete then insert
    conn.execute(SQL_DELETE_DRILLING_FLUID, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_DRILLING_FLUID, rows)

def replace_ddr_operations_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_OPERATIONS, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_OPERATIONS, rows)

def replace_ddr_survey_station_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_SURVEY_STATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_SURVEY_STATION, rows)

def replace_ddr_stratigraphic_information_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_STRATIGRAPHIC_INFORMATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_STRATIGRAPHIC_INFORMATION, rows)
def replace_ddr_lithology_information_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_LITHOLOGY_INFORMATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_LITHOLOGY_INFORMATION, rows)

def replace_ddr_gas_reading_information_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_GAS_READING_INFORMATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_GAS_READING_INFORMATION, rows)
