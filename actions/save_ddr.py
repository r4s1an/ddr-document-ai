from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from datetime import date, datetime, time
import pdfplumber, re ,io
from typing import Optional, List, Tuple, Dict, Any
from services.pdf_utils import (map_table_by_order, find_y_of_text,
                                 drop_empty_rows, _strip_leading_header,
                                 _strip_trailing_headers, extract_text_between, 
                                    find_next_header_line_y, _normalize_nullable_text,
                                    split_main_sub_activity)    
from typing import Optional, Tuple
import pandas as pd
from decimal import Decimal, InvalidOperation

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
                period_start,
                period_end,
                report_number
            )
            VALUES (
                :filename,
                :hash,
                :wellbore_name,
                :period_start,
                :period_end,
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
            "period_start": None,
            "period_end": None,
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
            start_date = date.fromisoformat(period_match.group(1))
            end_date = date.fromisoformat(period_match.group(3))

            start_time = (
                time.fromisoformat(period_match.group(2))
                if period_match.group(2)
                else time(0, 0)
            )

            end_time = (
                time.fromisoformat(period_match.group(4))
                if period_match.group(4)
                else time(0, 0)
            )

            metadata["period_start"] = datetime.combine(start_date, start_time)
            metadata["period_end"] = datetime.combine(end_date, end_time)
    
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

class SaveDDRDrillingFluidAction:
    """
    Extracts 'Drilling Fluid' table, handles page splits, transposes data,
    and maps dynamic headers to fixed SQL columns.
    """

    def __init__(self, engine):
        self.engine = engine

    def execute(self, conn, document_id: int, file_bytes: bytes, debug: bool = False) -> int:
        # 1. Extract and Transpose Data
        df = self._extract_drilling_fluid_df(file_bytes)

        # 2. Clear old data
        conn.execute(
            text("DELETE FROM ddr_drilling_fluid WHERE document_id = :document_id"),
            {"document_id": document_id},
        )

        if df.empty:
            if debug:
                print("[Drilling Fluid] No table found or table is empty.")
            return 0

        # 3. Normalize rows for SQL
        rows_to_insert = []
        for _, row in df.iterrows():
            clean_row = self._normalize_row(row, document_id)
            # Only add if we have at least a sample time or depth (avoid completely empty rows)
            if clean_row['sample_time'] or clean_row['sample_depth_mmd']:
                rows_to_insert.append(clean_row)

        if not rows_to_insert:
            return 0

        # 4. Bulk Insert
        insert_sql = text("""
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

        conn.execute(insert_sql, rows_to_insert)
        return len(rows_to_insert)

    # --------------------------
    # Extraction Logic (Your robust function)
    # --------------------------
    def _extract_drilling_fluid_df(self, file_bytes: bytes) -> pd.DataFrame:
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
        }

        full_table_rows = []
        expected_col_count = 0
        start_page_index = -1
        found = False

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            # STEP 1: Find the Start
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables(table_settings)
                for table in tables:
                    if table and table[0] and "Sample Time" in str(table[0][0]):
                        full_table_rows.extend(table)
                        expected_col_count = len(table[0])
                        start_page_index = i
                        found = True
                        break
                if found:
                    break

            if not found:
                return pd.DataFrame()

            # STEP 2: Check Next Page
            next_page_idx = start_page_index + 1
            if next_page_idx < len(pdf.pages):
                next_page = pdf.pages[next_page_idx]
                next_page_tables = next_page.extract_tables(table_settings)

                if next_page_tables:
                    continuation_table = next_page_tables[0]
                    if len(continuation_table[0]) == expected_col_count:
                        # Append (skip header if repeated)
                        if "Sample Time" in str(continuation_table[0][0]):
                            full_table_rows.extend(continuation_table[1:])
                        else:
                            full_table_rows.extend(continuation_table)

        # STEP 3: Clean Empty Rows
        cleaned_rows = []
        for row in full_table_rows:
            param_name = row[0]
            if param_name and str(param_name).strip():
                cleaned_rows.append(row)

        # STEP 4: Transpose
        df = pd.DataFrame(cleaned_rows)
        df = df.fillna("")
        if not df.empty:
            df = df.set_index(0)
            df_transposed = df.T
            df_transposed.reset_index(drop=True, inplace=True)
            return df_transposed
        
        return pd.DataFrame()

    # --------------------------
    # Data Normalization & Mapping
    # --------------------------
    def _normalize_row(self, row: pd.Series, document_id: int) -> Dict[str, Any]:
        """
        Maps a pandas Series (one column of the original PDF table) to SQL columns.
        Uses fuzzy matching for keys to handle 'Funnel Visc (s)' vs 'Funnel Visc ()'.
        """
        return {
            "document_id": document_id,
            "sample_time": self._clean_time(self._get_val_fuzzy(row, "Sample Time")),
            "sample_point": self._clean_text(self._get_val_fuzzy(row, "Sample Point")),
            "sample_depth_mmd": self._clean_numeric(self._get_val_fuzzy(row, "Sample Depth")),
            "fluid_type": self._clean_text(self._get_val_fuzzy(row, "Fluid Type")),
            
            # Numeric fields with changing units
            "fluid_density_g_cm3": self._clean_numeric(self._get_val_fuzzy(row, "Fluid Density")),
            "funnel_visc_s": self._clean_numeric(self._get_val_fuzzy(row, "Funnel Visc")),
            "plastic_visc_mpas": self._clean_numeric(self._get_val_fuzzy(row, "Plastic visc")),
            "yield_point_pa": self._clean_numeric(self._get_val_fuzzy(row, "Yield point")),
            "test_temp_hpht_degc": self._clean_numeric(self._get_val_fuzzy(row, "Test Temp HPHT")),
        }

    def _get_val_fuzzy(self, row: pd.Series, keyword: str) -> Optional[str]:
        """
        Finds value in the row where the column name contains the keyword.
        Example: keyword="Funnel Visc" matches column "Funnel Visc (s)" or "Funnel Visc ()"
        """
        # Iterate over the index (column names of the transposed df)
        for col_name in row.index:
            if keyword.lower() in str(col_name).lower():
                return row[col_name]
        return None

    # --------------------------
    # Cleaners
    # --------------------------
    def _clean_text(self, s: Any) -> Optional[str]:
        if not s:
            return None
        t = str(s).strip()
        return t if t else None

    def _clean_time(self, s: Any) -> Optional[str]:
        if not s:
            return None
        t = str(s).strip()
        # Basic check for HH:MM format
        if re.match(r"^\d{1,2}:\d{2}", t):
            return t
        return None

    def _clean_numeric(self, s: Any) -> Optional[float]:
        if not s:
            return None
        t = str(s).strip()
            
        # Replace comma with dot
        t = t.replace(",", ".")
        # Remove non-numeric chars except dot and minus (e.g. "1.36+" -> "1.36")
        # NOTE: Be careful with negative signs
        t = re.sub(r"[^\d\.-]", "", t)
        
        try:
            return float(t)
        except ValueError:
            return None
        
class SaveDDRPorePressureAction:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, conn, document_id: int, file_bytes: bytes, debug: bool = False) -> int:
        # 1. Extract Data using the provided logic
        extraction_result = self._extract_pore_pressure_table(file_bytes)

        # 2. Clear old data for this document
        conn.execute(
            text("DELETE FROM ddr_pore_pressure WHERE document_id = :document_id"),
            {"document_id": document_id},
        )

        if not extraction_result or not extraction_result.get("rows"):
            if debug:
                print("[Pore Pressure] No table found or table is empty.")
            return 0

        # 3. Normalize rows for SQL
        rows_to_insert = []
        for row_dict in extraction_result["rows"]:
            clean_row = self._map_row_to_sql(row_dict, document_id)
            
            # Only add if we have at least one meaningful value
            if any(x for k, x in clean_row.items() if k != "document_id" and x is not None):
                rows_to_insert.append(clean_row)

        if not rows_to_insert:
            return 0

        # 4. Bulk Insert
        insert_sql = text("""
            INSERT INTO ddr_pore_pressure (
                document_id,
                time,
                depth_mmd,
                depth_tvd,
                equ_mud_weight_g_cm3,
                reading
            )
            VALUES (
                :document_id,
                :time,
                :depth_mmd,
                :depth_tvd,
                :equ_mud_weight_g_cm3,
                :reading
            )
        """)

        conn.execute(insert_sql, rows_to_insert)
        return len(rows_to_insert)

    # ---------------------------------------------------------
    #  LOGIC FROM YOUR SNIPPET (Adapted to Class/Bytes)
    # ---------------------------------------------------------
    def _to_decimal_or_none(self, v: Optional[str]) -> Optional[Decimal]:
        if v is None:
            return None

        s = str(v).strip()
        if s == "":
            return None

        # common sentinel junk values in DDR PDFs
        if s in ("-999.99", "-999", "999.99"):
            return None

        # handle comma decimals like "1,02"
        s = s.replace(",", ".")

        try:
            return Decimal(s)
        except (InvalidOperation, ValueError):
            return None
    
    def _undouble(self, s: str) -> str:
        """Fix tokens like 'TTiimmee' -> 'Time' (every char duplicated)."""
        if not s:
            return ""
        s = str(s).strip()
        if len(s) >= 4 and len(s) % 2 == 0:
            a, b = s[::2], s[1::2]
            if a == b:
                return a
        return s

    def _collapse_repeats(self, s: str) -> str:
        """
        Collapse repeated letters inside words:
        DDeepptthh -> Depth, WWeeiigghhtt -> Weight, TTVVDD -> TVD, gg//ccmm33 -> g/cm3
        """
        if not s:
            return ""
        s = str(s)

        # normalize whitespace/newlines early
        s = re.sub(r"\s+", " ", s).strip()

        # collapse repeated alphabetic characters: "DD"->"D", "ee"->"e"
        s = re.sub(r"([A-Za-z])\1+", r"\1", s)

        # collapse repeated slashes
        s = re.sub(r"/{2,}", "/", s)

        # fix some common unit artifacts
        s = s.replace("cm33", "cm3")  # e.g., cm33 -> cm3
        s = re.sub(r"\(\(+", "(", s)
        s = re.sub(r"\)\)+", ")", s)
        s = re.sub(r"\s+\)", ")", s)
        s = re.sub(r"\(\s+", "(", s)

        return s

    def _clean_cell(self, s: str) -> str:
        # Apply BOTH cleaners as requested
        s = self._undouble(s)
        s = self._collapse_repeats(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _norm(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    def _extract_pore_pressure_table(self, file_bytes: bytes):
        """
        Exact logic from your extract_pore_pressure_table function, 
        adapted to read bytes instead of a file path.
        """
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):

                text_content = page.extract_text() or ""
                if "Pore Pressure" not in text_content:
                    continue

                # Using default extract_tables() as in your snippet
                tables = page.extract_tables()
                if not tables:
                    continue

                # pick the table whose header contains "time" and "reading"
                for t in tables:
                    if not t or not t[0]:
                        continue

                    header_raw = t[0]
                    header = [self._clean_cell(c) for c in header_raw]
                    header_n = [self._norm(h) for h in header]

                    if any("time" == h or "time" in h for h in header_n) and any("reading" in h for h in header_n):
                        # clean all rows
                        cleaned = [[self._clean_cell(c) for c in row] for row in t]

                        # build structured rows
                        hdr = cleaned[0]
                        rows = []
                        for row in cleaned[1:]:
                            if not any(row):
                                continue
                            rows.append({
                                hdr[i] if i < len(hdr) else f"col_{i}": row[i] if i < len(row) else ""
                                for i in range(max(len(hdr), len(row)))
                            })

                        return {
                            "page": pageno,
                            "header": hdr,
                            "rows": rows,
                            "raw_table": cleaned
                        }

        return None

    # ---------------------------------------------------------
    #  SQL MAPPING HELPERS
    # ---------------------------------------------------------

    def _map_row_to_sql(self, row_dict: Dict[str, str], document_id: int) -> Dict[str, Any]:
        return {
            "document_id": document_id,

            # time stays as text (or None)
            "time": self._to_text_or_none(self._find_value(row_dict, "Time")),

            # ✅ numeric fields -> Decimal/None (never "")
            "depth_mmd": self._to_decimal_or_none(self._find_value(row_dict, "Depth mMD")),
            "depth_tvd": self._to_decimal_or_none(self._find_value(row_dict, "Depth TVD")),
            "equ_mud_weight_g_cm3": self._to_decimal_or_none(self._find_value(row_dict, "Equ Mud Weight")),

            # reading stays as text (or None)
            "reading": self._to_text_or_none(self._find_value(row_dict, "Reading")),
        }
    
    def _to_text_or_none(self, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s != "" else None

    def _find_value(self, row: Dict[str, str], keyword: str) -> Optional[str]:
        """Case-insensitive partial key lookup."""
        keyword_lower = keyword.lower()
        for k, v in row.items():
            if keyword_lower in k.lower():
                return v
        return None

class SaveDDRSurveyStation:
    """
    Extracts the Survey Station table and inserts into ddr_survey_station.
    Expected DB schema:

        ddr_survey_station(
            document_id BIGINT FK,
            depth_mmd NUMERIC,
            depth_mtvd NUMERIC,
            inclination_dega NUMERIC,
            azimuth_dega NUMERIC,
            comment TEXT
        )
    """

    def __init__(self, engine):
        self.engine = engine

    # --------------------------
    # Public API
    # --------------------------
    def execute_with_conn(
        self,
        conn,
        document_id: int,
        file_bytes: bytes,
        debug: bool = False
    ) -> Dict[str, Any]:
        rows = self._extract_survey_station_rows(file_bytes=file_bytes, debug=debug)

        if not rows:
            return {"status": "no_data", "inserted": 0}

        insert_sql = text("""
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
            );
        """)

        inserted = 0
        for r in rows:
            payload = {
                "document_id": document_id,
                "depth_mmd": r.get("depth_mmd"),
                "depth_mtvd": r.get("depth_mtvd"),
                "inclination_dega": r.get("inclination_dega"),
                "azimuth_dega": r.get("azimuth_dega"),
                "comment": r.get("comment"),
            }
            conn.execute(insert_sql, payload)
            inserted += 1

        return {"status": "ok", "inserted": inserted}

    # --------------------------
    # Extraction
    # --------------------------
    def _extract_survey_station_rows(
        self,
        file_bytes: bytes,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").lower()
                if "survey station" not in page_text:
                    continue

                if debug:
                    print(f"[SurveyStation] PAGE {page_no}: header found")

                page = page.dedupe_chars(tolerance=1)

                # locate "Survey Station" header via words
                words = page.extract_words(use_text_flow=True)
                # Find the word "Survey" (robust enough for this PDF)
                header_word = next(
                    (w for w in words if (w.get("text") or "").strip().lower() == "survey"),
                    None
                )
                if not header_word:
                    if debug:
                        print("[SurveyStation] Could not locate header word bbox")
                    continue

                header_bottom = header_word["bottom"]

                cropped = page.crop((0, header_bottom + 5, page.width, page.height))

                tables = cropped.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 2,
                })

                if not tables:
                    if debug:
                        print("[SurveyStation] No tables found under header")
                    continue

                table = tables[0]
                if not table or not table[0]:
                    if debug:
                        print("[SurveyStation] Extracted empty table structure")
                    continue

                # Build canonical headers so:
                # "Inclination (dega)" and "Inclination ()" => "inclination"
                # "Azimuth (dega)" and "Azimuth ()" => "azimuth"
                raw_headers = [self._safe_str(h) for h in table[0]]
                headers = [self._canon_header(h) for h in raw_headers]

                if debug:
                    print("[SurveyStation] Raw headers:", raw_headers)
                    print("[SurveyStation] Canon headers:", headers)

                # Row parse
                out: List[Dict[str, Any]] = []
                for row in table[1:]:
                    if not row or not any(cell for cell in row if self._safe_str(cell).strip()):
                        continue

                    row_map = {}
                    for i, key in enumerate(headers):
                        if not key:
                            continue
                        if i >= len(row):
                            continue
                        row_map[key] = self._safe_str(row[i]).strip()

                    # Map into DB schema (NUMERIC => Decimal or None)
                    out.append({
                        "depth_mmd": self._to_decimal(row_map.get("depth_mmd")),
                        "depth_mtvd": self._to_decimal(row_map.get("depth_mtvd")),
                        "inclination_dega": self._to_decimal(row_map.get("inclination")),
                        "azimuth_dega": self._to_decimal(row_map.get("azimuth")),
                        "comment": row_map.get("comment") or None,
                    })

                return out  # only one Survey Station per PDF

        return []

    # --------------------------
    # Helpers
    # --------------------------
    def _safe_str(self, v: Any) -> str:
        return "" if v is None else str(v)

    def _canon_header(self, h: str) -> str:
        """
        Canonicalize headers so unit variations do not break mapping:
        - Removes anything in parentheses
        - Lowercases
        - Normalizes whitespace and punctuation
        - Maps to known keys we expect in Survey Station
        """
        h = (h or "").strip()

        # Remove unit text "(dega)" or "()" etc
        h = re.sub(r"\s*\(.*?\)\s*", "", h)

        # Normalize spaces
        h = re.sub(r"\s+", " ", h).strip().lower()

        # Handle typical column names from this PDF
        # (after parentheses removal)
        if h in ("depth mmd", "depth mmd "):
            return "depth_mmd"
        if h in ("depth mtvd", "depth mtvd "):
            return "depth_mtvd"
        if h == "inclination":
            return "inclination"
        if h == "azimuth":
            return "azimuth"
        if h == "comment":
            return "comment"

        # Fallback: make it a safe-ish key (won't crash, but may be ignored)
        return h.replace(" ", "_")

    def _to_decimal(self, s: Optional[str]) -> Optional[Decimal]:
        """
        Convert string to Decimal safely:
        - Returns None for '', None, '-999.99' (sentinel), etc.
        - Accepts comma decimals like '1,48' -> '1.48'
        """
        if s is None:
            return None
        s = s.strip()
        if not s:
            return None

        # Treat common sentinels as NULL
        if s in ("-999.99", "-999", "999.99"):
            return None

        # Convert comma decimal to dot
        s = s.replace(",", ".")

        try:
            return Decimal(s)
        except (InvalidOperation, ValueError):
            return None