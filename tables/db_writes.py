 

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

SQL_UPSERT_DDR_ACTIVITY_SUMMARY = text("""
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

SQL_UPSERT_DDR_SUMMARY_REPORT = text("""
INSERT INTO ddr_summary_report (
    document_id,
    status,
    report_creation_ts,
    report_number,
    days_ahead_behind,
    operator,
    rig_name,
    drilling_contractor,
    spud_ts,
    wellbore_type,
    elevation_rkb_msl_m,
    water_depth_msl_m,
    tight_well,
    hpht,
    temperature_degc,
    pressure_psig,
    date_well_complete,
    dist_drilled_m,
    penetration_rate_mph,
    hole_dia_in,
    pressure_test_type,
    formation_strength_g_cm3,
    dia_last_casing,
    depth_kickoff_mmd,
    depth_kickoff_mtvd,
    depth_mmd,
    depth_mtvd,
    plug_back_depth_mmd,
    depth_formation_strength_mmd,
    depth_formation_strength_mtvd,
    depth_last_casing_mmd,
    depth_last_casing_mtvd,
    extracted_at
)
VALUES (
    :document_id,
    :status,
    :report_creation_ts,
    :report_number,
    :days_ahead_behind,
    :operator,
    :rig_name,
    :drilling_contractor,
    :spud_ts,
    :wellbore_type,
    :elevation_rkb_msl_m,
    :water_depth_msl_m,
    :tight_well,
    :hpht,
    :temperature_degc,
    :pressure_psig,
    :date_well_complete,
    :dist_drilled_m,
    :penetration_rate_mph,
    :hole_dia_in,
    :pressure_test_type,
    :formation_strength_g_cm3,
    :dia_last_casing,
    :depth_kickoff_mmd,
    :depth_kickoff_mtvd,
    :depth_mmd,
    :depth_mtvd,
    :plug_back_depth_mmd,
    :depth_formation_strength_mmd,
    :depth_formation_strength_mtvd,
    :depth_last_casing_mmd,
    :depth_last_casing_mtvd,
    NOW()
)
ON CONFLICT (document_id)
DO UPDATE SET
    status = EXCLUDED.status,
    report_creation_ts = EXCLUDED.report_creation_ts,
    report_number = EXCLUDED.report_number,
    days_ahead_behind = EXCLUDED.days_ahead_behind,
    operator = EXCLUDED.operator,
    rig_name = EXCLUDED.rig_name,
    drilling_contractor = EXCLUDED.drilling_contractor,
    spud_ts = EXCLUDED.spud_ts,
    wellbore_type = EXCLUDED.wellbore_type,
    elevation_rkb_msl_m = EXCLUDED.elevation_rkb_msl_m,
    water_depth_msl_m = EXCLUDED.water_depth_msl_m,
    tight_well = EXCLUDED.tight_well,
    hpht = EXCLUDED.hpht,
    temperature_degc = EXCLUDED.temperature_degc,
    pressure_psig = EXCLUDED.pressure_psig,
    date_well_complete = EXCLUDED.date_well_complete,
    dist_drilled_m = EXCLUDED.dist_drilled_m,
    penetration_rate_mph = EXCLUDED.penetration_rate_mph,
    hole_dia_in = EXCLUDED.hole_dia_in,
    pressure_test_type = EXCLUDED.pressure_test_type,
    formation_strength_g_cm3 = EXCLUDED.formation_strength_g_cm3,
    dia_last_casing = EXCLUDED.dia_last_casing,
    depth_kickoff_mmd = EXCLUDED.depth_kickoff_mmd,
    depth_kickoff_mtvd = EXCLUDED.depth_kickoff_mtvd,
    depth_mmd = EXCLUDED.depth_mmd,
    depth_mtvd = EXCLUDED.depth_mtvd,
    plug_back_depth_mmd = EXCLUDED.plug_back_depth_mmd,
    depth_formation_strength_mmd = EXCLUDED.depth_formation_strength_mmd,
    depth_formation_strength_mtvd = EXCLUDED.depth_formation_strength_mtvd,
    depth_last_casing_mmd = EXCLUDED.depth_last_casing_mmd,
    depth_last_casing_mtvd = EXCLUDED.depth_last_casing_mtvd,
    extracted_at = NOW();
""")


SQL_UPDATE_DDR_REPORT_NUMBER = text("""
UPDATE ddr_documents
SET report_number = :rn
WHERE id = :id
""")
 

SQL_DELETE_DDR_DRILLING_FLUID = text("""
DELETE FROM ddr_drilling_fluid
WHERE document_id = :document_id
""")

SQL_DELETE_DDR_OPERATIONS = text("""
DELETE FROM ddr_operations
WHERE document_id = :document_id
""")

SQL_DELETE_DDR_SURVEY_STATION = text("""
DELETE FROM ddr_survey_station
WHERE document_id = :document_id
""")

SQL_DELETE_DDR_STRATIGRAPHIC_INFORMATION = text("""
DELETE FROM ddr_stratigraphic_information
WHERE document_id = :document_id
""")

SQL_DELETE_DDR_LITHOLOGY_INFORMATION = text("""
DELETE FROM ddr_lithology_information
WHERE document_id = :document_id
""")

SQL_DELETE_DDR_GAS_READING_INFORMATION = text("""
DELETE FROM ddr_gas_reading_information
WHERE document_id = :document_id
""")


SQL_INSERT_DDR_DRILLING_FLUID = text("""
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

SQL_INSERT_DDR_OPERATIONS = text("""
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

SQL_INSERT_DDR_SURVEY_STATION = text("""
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

SQL_INSERT_DDR_STRATIGRAPHIC_INFORMATION = text("""
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

SQL_INSERT_DDR_LITHOLOGY_INFORMATION = text("""
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

SQL_INSERT_DDR_GAS_READING_INFORMATION = text("""
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
    conn.execute(SQL_UPDATE_DDR_REPORT_NUMBER, {"rn": report_number, "id": document_id})


def upsert_ddr_activity_summary(
    conn,
    *,
    document_id: int,
    activities_24h_text: str,
    planned_24h_text: str,
) -> None:
    conn.execute(
        SQL_UPSERT_DDR_ACTIVITY_SUMMARY,
        {
            "document_id": document_id,
            "activities_24h_text": activities_24h_text,
            "planned_24h_text": planned_24h_text,
        },
    )

def upsert_ddr_summary_report(conn, *, payload: Dict[str, Any]) -> None:
    conn.execute(SQL_UPSERT_DDR_SUMMARY_REPORT, payload)

def insert_ddr_drilling_fluid_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_DDR_DRILLING_FLUID, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_DDR_DRILLING_FLUID, rows)


def insert_ddr_operations_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_DDR_OPERATIONS, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_DDR_OPERATIONS, rows)

def insert_ddr_survey_station_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_DDR_SURVEY_STATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_DDR_SURVEY_STATION, rows)

def insert_ddr_stratigraphic_information_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_DDR_STRATIGRAPHIC_INFORMATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_DDR_STRATIGRAPHIC_INFORMATION, rows)

def insert_ddr_lithology_information_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_DDR_LITHOLOGY_INFORMATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_DDR_LITHOLOGY_INFORMATION, rows)

def insert_ddr_gas_reading_information_rows(
    conn,
    *,
    document_id: int,
    rows: List[Dict[str, Any]],
) -> None:
    conn.execute(SQL_DELETE_DDR_GAS_READING_INFORMATION, {"document_id": document_id})
    if rows:
        conn.execute(SQL_INSERT_DDR_GAS_READING_INFORMATION, rows)