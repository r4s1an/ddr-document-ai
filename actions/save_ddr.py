from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from datetime import date, time
import pdfplumber
import re
import io
from services.pdf_utils import map_table_by_order, find_y_of_text, drop_empty_rows

class SaveDDRAction:
    def __init__(self, engine):
        self.engine = engine

    def execute_with_conn(self, conn, filename: str, file_hash: str, file_bytes: bytes) -> dict:
        metadata = self._extract_metadata(file_bytes)

        insert_sql = text("""
            INSERT INTO ddr_documents (
                source_filename,
                file_sha256,
                wellbore_name,
                period_start_date,
                period_start_time,
                period_end_date,
                period_end_time,
                report_number
            )
            VALUES (
                :filename,
                :hash,
                :wellbore_name,
                :period_start_date,
                :period_start_time,
                :period_end_date,
                :period_end_time,
                :report_number
            )
            RETURNING id;
        """)

        select_sql = text("""
            SELECT id FROM ddr_documents
            WHERE file_sha256 = :hash;
        """)

        try:
            doc_id = conn.execute(
                insert_sql,
                {
                    "filename": filename,
                    "hash": file_hash,
                    **metadata,
                },
            ).scalar_one()

            return {
                "status": "created",
                "document_id": doc_id,
                "metadata": metadata,
            }
        
        except IntegrityError:
            doc_id = conn.execute(
                select_sql, {"hash": file_hash}
            ).scalar_one()

            return {
                "status": "duplicate",
                "document_id": doc_id,
            }

    def _extract_metadata(self, file_bytes: bytes) -> dict:
        metadata = {
            "wellbore_name": None,
            "period_start_date": None,
            "period_start_time": None,
            "period_end_date": None,
            "period_end_time": None,
            "report_number": None,
        }

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if not pdf.pages:
                    return metadata

                page = pdf.pages[0].dedupe_chars(tolerance=1)
                text = page.extract_text() or ""
        except Exception:
            return metadata

        well_match = re.search(
            r'Wellbore\s*:\s*([0-9/]+(?:-[A-Za-z0-9]+)*(?:\s+[A-Z]{1,3})?)\b',
            text,
        )

        if well_match:
            metadata["wellbore_name"] = well_match.group(1).strip()


        # Period (date + optional time)
        period_match = re.search(
            r'Period:\s*'
            r'(\d{4}-\d{2}-\d{2})'
            r'(?:\s+(\d{2}:\d{2}))?\s*'
            r'[-–—]\s*'
            r'(\d{4}-\d{2}-\d{2})'
            r'(?:\s+(\d{2}:\d{2}))?',
            text,
            re.IGNORECASE,
        )

        if period_match:
            metadata["period_start_date"] = date.fromisoformat(
                period_match.group(1)
            )
            metadata["period_end_date"] = date.fromisoformat(
                period_match.group(3)
            )

            if period_match.group(2):
                metadata["period_start_time"] = time.fromisoformat(
                    period_match.group(2)
                )

            if period_match.group(4):
                metadata["period_end_time"] = time.fromisoformat(
                    period_match.group(4)
                )

        # Report number
        report_match = re.search(
            r'Report\s*(?:number|#)\s*[:=]\s*(\d+)',
            text,
            re.IGNORECASE,
        )
        if report_match:
            metadata["report_number"] = int(report_match.group(1))

        return metadata
    
class SaveDDRSummaryAction:

    TABLE_1_FIELDS = [
        "status",
        "report_creation_ts",
        "report_number",
        "days_ahead_behind",
        "operator",
        "rig_name",
        "drilling_contractor",
        "spud_ts",
        "wellbore_type",
        "elevation_rkb_msl_m",
        "water_depth_msl_m",
        "tight_well",
        "hpht",
        "temperature_degc",
        "pressure_psig",
        "date_well_complete",
    ]

    TABLE_2_FIELDS = [
        "dist_drilled_m",
        "penetration_rate_mph",
        "hole_dia_in",
        "pressure_test_type",
        "formation_strength_g_cm3",
        "dia_last_casing",
    ]

    TABLE_3_FIELDS = [
        "depth_kickoff_mmd",
        "depth_kickoff_mtvd",
        "depth_mmd",
        "depth_mtvd",
        "plug_back_depth_mmd",
        "depth_formation_strength_mmd",
        "depth_formation_strength_mtvd",
        "depth_last_casing_mmd",
        "depth_last_casing_mtvd",
    ]

    def __init__(self, engine):
        self.engine = engine

    def execute(self, conn, document_id: int, file_bytes: bytes):

        summary = self._extract_summary(file_bytes)

        insert_sql = text("""
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
                depth_last_casing_mtvd
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
                :depth_last_casing_mtvd
            )
            ON CONFLICT (document_id)
            DO UPDATE SET
                extracted_at = NOW();
        """)

        conn.execute(insert_sql, {"document_id": document_id, **summary})

    def _extract_summary(self, file_bytes: bytes) -> dict:
        data = {}

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if "summary report" not in text.lower():
                    continue

                page = page.dedupe_chars(tolerance=1)

                y_start = find_y_of_text(page, "Summary report") or 0
                y_end = find_y_of_text(page, "Summary of activities") or page.height

                tables = []
                for t in page.find_tables():
                    _, top, _, bottom = t.bbox
                    if top >= y_start and bottom <= y_end:
                        tables.append(t.extract())

                if len(tables) < 3:
                    continue
                
                t1 = drop_empty_rows(tables[0])
                t2 = drop_empty_rows(tables[1])
                t3 = drop_empty_rows(tables[2])

                data.update(map_table_by_order(t1, self.TABLE_1_FIELDS))
                data.update(map_table_by_order(t2, self.TABLE_2_FIELDS))
                data.update(map_table_by_order(t3, self.TABLE_3_FIELDS))
                break

        return data