# services/ddr_analytics_fetch.py
from io import BytesIO
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import pandas as pd
import decimal

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    if isinstance(obj, decimal.Decimal):
        return float(obj)

    return obj


def list_ddr_documents(engine, limit: int = 200) -> pd.DataFrame:
    """
    For dropdown selection in UI.
    """
    q = """
    SELECT id, source_filename, wellbore_name, period_start, period_end
    FROM ddr_documents
    ORDER BY id DESC
    LIMIT %(limit)s
    """
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"limit": limit})


def _read_one(engine, sql: str, params: dict) -> dict:
    """
    Reads a single-row table (0 or 1 row).
    Returns {} if empty.
    """
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return {}
    return df.to_dict(orient="records")[0]


def _read_many(engine, sql: str, params: dict) -> list[dict]:
    """
    Reads multi-row table.
    Returns [] if empty.
    """
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return []
    df = df.fillna("")  # make LLM payload clean
    return df.to_dict(orient="records")


def fetch_ddr_payload(engine, document_id: int) -> dict:
    """
    Fetch EVERYTHING needed for analytics from SQL, by document_id.
    Output is JSON-serializable dict.
    """

    params = {"id": document_id}

    payload = {
        # Single-row tables
        "document": _read_one(engine, "SELECT * FROM ddr_documents WHERE id = %(id)s", params),
        "activity_summary": _read_one(engine, "SELECT * FROM ddr_activity_summary WHERE document_id = %(id)s", params),
        "summary_report": _read_one(engine, "SELECT * FROM ddr_summary_report WHERE document_id = %(id)s", params),

        # Multi-row tables (ordered where it matters)
        "operations": _read_many(
            engine,
            """
            SELECT start_time, end_time, end_depth_mmd, main_activity, sub_activity, state, remark
            FROM ddr_operations
            WHERE document_id = %(id)s
            ORDER BY start_time NULLS LAST
            """,
            params,
        ),

        "drilling_fluid": _read_many(
            engine,
            """
            SELECT sample_time, sample_point, sample_depth_mmd, fluid_type,
                   fluid_density_g_cm3, funnel_visc_s, plastic_visc_mpas, yield_point_pa, test_temp_hpht_degc
            FROM ddr_drilling_fluid
            WHERE document_id = %(id)s
            ORDER BY sample_time NULLS LAST
            """,
            params,
        ),

        "survey_station": _read_many(
            engine,
            """
            SELECT depth_mmd, depth_mtvd, inclination_dega, azimuth_dega, comment
            FROM ddr_survey_station
            WHERE document_id = %(id)s
            ORDER BY depth_mmd NULLS LAST
            """,
            params,
        ),

        "stratigraphic_information": _read_many(
            engine,
            """
            SELECT depth_top_formation_mmd, depth_top_formation_mtvd, description
            FROM ddr_stratigraphic_information
            WHERE document_id = %(id)s
            ORDER BY depth_top_formation_mmd NULLS LAST
            """,
            params,
        ),

        "lithology_information": _read_many(
            engine,
            """
            SELECT start_depth_mmd, end_depth_mmd, start_depth_mtvd, end_depth_mtvd,
                   shows_description, lithology_description
            FROM ddr_lithology_information
            WHERE document_id = %(id)s
            ORDER BY start_depth_mmd NULLS LAST
            """,
            params,
        ),

        "gas_reading_information": _read_many(
            engine,
            """
            SELECT time, class,
                   depth_to_top_mmd, depth_to_bottom_md,
                   depth_to_top_mtvd, depth_to_bottom_tvd,
                   highest_gas_percent, lowest_gas_percent,
                   c1_ppm, c2_ppm, c3_ppm, ic4_ppm, ic5_ppm
            FROM ddr_gas_reading_information
            WHERE document_id = %(id)s
            ORDER BY time NULLS LAST
            """,
            params,
        ),
    }

    return _json_safe(payload)