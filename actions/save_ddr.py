from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from datetime import date, time
import pdfplumber, re ,io
from typing import Optional, List, Tuple, Dict, Any
from services.pdf_utils import (map_table_by_order, find_y_of_text,
                                 drop_empty_rows, _strip_leading_header,
                                 _strip_trailing_headers, extract_text_between, 
                                    find_next_header_line_y, _normalize_nullable_text,
                                    split_main_sub_activity)    
from typing import Optional, Tuple

class SaveDDROperationsAction:
    """
    Extracts Operations table rows and saves into ddr_operations.
    Strategy:
      - Find 'Operations' header y on the first relevant page
      - Pick the first detected table below it
      - Normalize rows + split main/sub activity
      - DELETE existing ops rows for document_id
      - Bulk INSERT normalized rows
    """

    def __init__(self, engine):
        self.engine = engine

    def execute(self, conn, document_id: int, file_bytes: bytes, debug: bool = False) -> int:
        rows = self._extract_operations_rows(file_bytes=file_bytes, debug=debug)

        # Always clear old rows for this doc, then insert what we found
        conn.execute(
            text("DELETE FROM ddr_operations WHERE document_id = :document_id"),
            {"document_id": document_id},
        )

        if not rows:
            # nothing to insert
            return 0

        insert_sql = text("""
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

        # Add document_id per row and bulk insert
        payload = [{"document_id": document_id, **r} for r in rows]
        conn.execute(insert_sql, payload)
        return len(payload)

    # --------------------------
    # Extraction + normalization
    # --------------------------
    def _extract_operations_rows(self, file_bytes: bytes, debug: bool = False) -> List[Dict[str, Any]]:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").lower()
                if "operations" not in page_text and "summary" not in page_text:
                    # cheap skip; ops is usually near summary pages
                    continue

                page = page.dedupe_chars(tolerance=1)

                raw_table = self._extract_operations_table_first_below(page, debug=debug)
                if raw_table:
                    if debug:
                        print(f"[Ops] Found operations table on page {pno} with {len(raw_table)} rows (incl header)")
                    return self._normalize_operations_table(raw_table)

        if debug:
            print("[Ops] No operations table found in PDF.")
        return []

    def _extract_operations_table_first_below(self, page, debug: bool = False):
        OPERATIONS_HDR = "Operations"
        y_ops = find_y_of_text(page, OPERATIONS_HDR)
        if y_ops is None:
            return None

        tables = page.find_tables() or []
        candidates: List[Tuple[float, object]] = []

        for i, t in enumerate(tables):
            x0, top, x1, bottom = t.bbox
            if debug:
                print(f"[Ops] table #{i}: bbox=({x0:.1f},{top:.1f},{x1:.1f},{bottom:.1f})")
            if top >= y_ops - 2:  # small tolerance
                candidates.append((top, t))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        _, table_obj = candidates[0]

        if debug:
            x0, top, x1, bottom = table_obj.bbox
            print(f"[Ops] Selected table bbox=({x0:.1f},{top:.1f},{x1:.1f},{bottom:.1f})")

        return table_obj.extract()

    def _normalize_operations_table(self, raw_table) -> List[Dict[str, Any]]:
        if not raw_table or len(raw_table) < 2:
            return []

        rows = raw_table[1:]  # skip header row
        out: List[Dict[str, Any]] = []

        for r in rows:
            if not r or len(r) < 6:
                continue

            start_time = self._clean_time(r[0])
            end_time = self._clean_time(r[1])
            end_depth_mmd = self._clean_numeric(r[2])

            main_activity, sub_activity = split_main_sub_activity(r[3])

            state = self._clean_text_keep_spaces(r[4])
            remark = self._clean_remark(r[5])

            out.append({
                "start_time": start_time,           # string "HH:MM" is OK; Postgres TIME will parse
                "end_time": end_time,
                "end_depth_mmd": end_depth_mmd,     # numeric as float/Decimal/str
                "main_activity": main_activity,
                "sub_activity": sub_activity,
                "state": state,
                "remark": remark,
            })

        return out

    # --------------------------
    # Cleaners (simple + safe)
    # --------------------------
    def _clean_time(self, s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        t = s.strip()
        # allow "00:00" etc.
        return t if re.fullmatch(r"\d{2}:\d{2}", t) else t

    def _clean_numeric(self, s: Optional[str]) -> Optional[str]:
        """
        Keep numeric as string (safe for SQLAlchemy -> Postgres NUMERIC).
        Handles commas used as decimal separators.
        """
        if not s:
            return None
        t = s.strip()
        t = t.replace(",", ".")
        # remove any stray spaces
        t = re.sub(r"\s+", "", t)
        return t or None

    def _clean_text_keep_spaces(self, s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        t = re.sub(r"[ \t]+", " ", s.replace("\n", " ")).strip()
        return t or None

    def _clean_remark(self, s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        # For remark, newlines are sentence breaks: convert to single spaces
        t = s.replace("\n", " ")
        t = re.sub(r"[ \t]+", " ", t).strip()
        return t or None
    
class SaveDDRActivitySummaryAction:
    """
    Extracts text under:
      - 'Summary of activities (24 Hours)'
      - 'Summary of planned activities (24 Hours)'
    and upserts into ddr_activity_summary.
    """
    
    def __init__(self, engine):
        self.engine = engine

    def extract_two_summaries_from_page(self, page, debug: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Same logic you already have, but packaged here for direct use in the action.
        """
        ACTIVITIES_HDR = "Summary of activities (24 Hours)"
        PLANNED_HDR    = "Summary of planned activities (24 Hours)"

        y_act = find_y_of_text(page, ACTIVITIES_HDR)
        y_plan = find_y_of_text(page, PLANNED_HDR)

        activities = None
        planned = None

        if y_act is not None:
            if y_plan is not None and y_plan > y_act:
                y_end = y_plan
                if debug:
                    print(f"Activities stop at planned header y={y_end:.2f}")
            else:
                nxt = find_next_header_line_y(page, y_after=y_act)
                y_end = nxt[0] if nxt else float(page.height)
                if debug:
                    print(f"Activities next header: '{nxt[1] if nxt else 'None'}' at y={y_end:.2f}")

            activities = extract_text_between(page, y_act, y_end)
            activities = _strip_leading_header(activities, ACTIVITIES_HDR)

            activities = _strip_trailing_headers(activities, [PLANNED_HDR])

            activities = _normalize_nullable_text(activities)

        if y_plan is not None:
            nxt = find_next_header_line_y(page, y_after=y_plan)
            y_end = nxt[0] if nxt else float(page.height)
            if debug:
                print(f"Planned next header: '{nxt[1] if nxt else 'None'}' at y={y_end:.2f}")

            planned = extract_text_between(page, y_plan, y_end)
            planned = _strip_leading_header(planned, PLANNED_HDR)
            planned = _normalize_nullable_text(planned)

        return activities, planned

    def execute(self, conn, document_id: int, file_bytes: bytes, debug: bool = False) -> None:
        activities, planned = self._extract_activity_summary(file_bytes, debug=debug)

        insert_sql = text("""
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

        conn.execute(insert_sql, {
            "document_id": document_id,
            "activities_24h_text": activities,
            "planned_24h_text": planned,
        })

    def _extract_activity_summary(self, file_bytes: bytes, debug: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Scan pages likely containing the summary section and return (activities, planned).
        Stops at the first page where we find at least one of them.
        """
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                # quick filter to skip irrelevant pages
                page_text = (page.extract_text() or "").lower()
                if "summary" not in page_text:
                    continue

                # match how you do table extraction: dedupe to reduce split/overprint issues
                page = page.dedupe_chars(tolerance=1)

                activities, planned = self.extract_two_summaries_from_page(page, debug=debug)

                # If either one is found, accept and stop
                if activities or planned:
                    return activities, planned

        return None, None

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

        existing_id = conn.execute(select_sql, {"hash": file_hash}).scalar()
        if existing_id is not None:
            return {
                "status": "duplicate",
                "document_id": existing_id,
                "metadata": metadata,
            }

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