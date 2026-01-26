from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
from tables.summary.orchestrator import SummaryReportParser
from services.layout_graph import build_and_save_layout
from services.post_layout_ocr import ocr_layout_json
from services.ddr_metadata import parse_wellbore, parse_period
from tables.utils import coerce_summary_payload
from services.activity_summary_extractor import extract_activity_summaries_from_processed_ddr
from tables.db_writes import (
    upsert_ddr_document, insert_ddr_operations_rows,
    upsert_ddr_activity_summary, insert_ddr_survey_station_rows,
    update_report_number, insert_ddr_drilling_fluid_rows,
    insert_ddr_stratigraphic_information_rows,
    insert_ddr_lithology_information_rows,
    insert_ddr_gas_reading_information_rows,
    upsert_ddr_summary_report
)
from services.table_ocr_orchestrator import run_paddle_for_all_tables
from tables.drilling_fluid_parser import parse_drilling_fluid_rows
from tables.operations_parser import parse_operations_rows
from tables.survey_station_parser import parse_survey_station_rows
from tables.stratigraphic_information_parser import parse_stratigraphic_information_rows
from tables.lithology_information_parser import parse_lithology_information_rows
from tables.gas_reading_information_parser import parse_gas_reading_information_rows

@dataclass
class IngestResult:
    document_id: int
    wellbore_name: Optional[str]
    period_start: Optional[str]
    period_end: Optional[str]
    used_page: Optional[str]
    debug: Dict[str, Any]

class IngestDDRAction:
    def __init__(self, engine, ocr_gpu: bool = True):
        self.engine = engine
        self.ocr_gpu = ocr_gpu

    def _find_metadata_from_pages(self, out_dir: Path) -> Tuple[Optional[str], Optional[Tuple], Optional[str], Dict[str, Any]]:
        """
        Look for wellbore/period in OCR text.
        Strategy: first try page_001, then fallback to all pages.
        """
        debug = {"pages_checked": []}

        page_dirs = sorted(out_dir.glob("page_*"))

        def scan_page(pd: Path):
            data = json.loads((pd / "layout_ocr.json").read_text(encoding="utf-8"))
            items = data.get("items") if isinstance(data, dict) else data
            if not isinstance(items, list):
                debug.setdefault("errors", []).append({
                    "step": "metadata_extraction",
                    "page": pd.name,
                    "reason": "layout_ocr items not a list"
                })
                return None, None

            # DEBUG: record counts per page
            debug.setdefault("metadata_page_counts", {})[pd.name] = {
                "num_items": len(items),
                "nonempty_ocr_text": sum(1 for it in items if (it.get("ocr_text") or "").strip()),
                "labels": {
                    "wellbore_field": sum(1 for it in items if it.get("label") == "wellbore_field"),
                    "period_field": sum(1 for it in items if it.get("label") == "period_field"),
                    "section_header": sum(1 for it in items if it.get("label") == "section_header"),
                    "plain_text": sum(1 for it in items if it.get("label") == "plain_text"),
                }
            }
            # collect OCR text from wellbore_field + period_field first
            wb_texts = [it.get("ocr_text","") for it in items if it.get("label") == "wellbore_field"]
            per_texts = [it.get("ocr_text","") for it in items if it.get("label") == "period_field"]
            all_texts = [it.get("ocr_text","") for it in items if (it.get("ocr_text") or "").strip()]

            well = None
            period = None

            for t in wb_texts + all_texts:
                well = well or parse_wellbore(t)
                if well:
                    break

            for t in per_texts + all_texts:
                period = period or parse_period(t)
                if period:
                    break

            return well, period

        # Prefer page_001
        first = out_dir / "page_001"
        if first.exists():
            debug["pages_checked"].append(first.name)
            well, period = scan_page(first)
            if well or period:
                return well, period, first.name, debug

        # Fallback scan all pages
        well = None
        period = None
        used = None
        for pd in page_dirs:
            debug["pages_checked"].append(pd.name)
            w, p = scan_page(pd)
            well = well or w
            period = period or p
            if (w or p) and used is None:
                used = pd.name
            if well and period:
                break

        return well, period, used, debug
    
    def _upsert_activity_summary(self, conn, out_dir: Path, document_id: int, debug: Dict[str, Any]) -> None:
        try:
            summaries = extract_activity_summaries_from_processed_ddr(
                processed_ddr_dir=out_dir,
                header_match_threshold=0.78,
                debug=debug.get("debug_activity", False),
            )

            activities = summaries.get("activities_24h_text") or ""
            planned = summaries.get("planned_24h_text") or ""

            upsert_ddr_activity_summary(
                conn,
                document_id=document_id,
                activities_24h_text=activities,
                planned_24h_text=planned,
            )

            debug["activity_summary_inserted"] = True
            debug["activity_summary_lengths"] = {
                "activities_24h": len(activities),
                "planned_24h": len(planned)
            }
        except Exception as e:
            debug.setdefault("errors", []).append({
                "step": "activity_summary_extraction",
                "error": str(e),
                "error_type": type(e).__name__
            })
            debug["activity_summary_inserted"] = False


    def _write_summary_report(self, conn, out_dir: Path, document_id: int, debug: Dict[str, Any]) -> None:
        page_dirs = sorted(out_dir.glob("page_*"))
        found_any = False

        for pd in page_dirs:
            layout_path = pd / "layout.json"
            ocr_path = pd / "layout_ocr.json"
            if not layout_path.exists() or not ocr_path.exists():
                continue

            try:
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
                ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            except Exception as e:
                debug.setdefault("errors", []).append({
                    "step": "summary_report_parsing",
                    "page": pd.name,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                continue

            summary_header_idxs = {
                it["idx"]
                for it in ocr.get("items", [])
                if it.get("label") == "section_header"
                and (it.get("ocr_text") or "").strip().lower() == "summary report"
            }

            if not summary_header_idxs:
                continue

            summary_tables = [
                it for it in layout.get("items", [])
                if it.get("label") == "table" and it.get("linked_header_idx") in summary_header_idxs
            ]

            # Warning: Expected structure not found (non-fatal)
            if len(summary_tables) != 3:
                debug.setdefault("warnings", []).append({
                    "step": "summary_report_parsing",
                    "page": pd.name,
                    "reason": f"expected 3 tables, got {len(summary_tables)}",
                })
                continue

            try:
                parser = SummaryReportParser()
                payload = parser.extract(document_id=document_id, table_items=summary_tables)
                payload = coerce_summary_payload(payload)

                upsert_ddr_summary_report(conn, payload=payload)

                debug["summary_report_inserted"] = True
                debug["summary_report_payload"] = payload
                found_any = True
                break
            except Exception as e:
                debug.setdefault("errors", []).append({
                    "step": "summary_report_insertion",
                    "page": pd.name,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                continue

        if not found_any:
            debug["summary_report_inserted"] = False
            debug.setdefault("warnings", []).append({
                "step": "summary_report_parsing",
                "reason": "no valid summary report found in any page"
            })
            
    def execute(self, filename: str, file_hash: str, out_dir: Path) -> IngestResult:
        # ===================================================================
        # PHASE 1: Pre-processing (Layout Detection & OCR)
        # ===================================================================
        
        # Step 1.1: Build layout for all pages
        page_dirs = sorted(out_dir.glob("page_*"))
        for pd in page_dirs:
            build_and_save_layout(pd)
            ocr_layout_json(pd, gpu=self.ocr_gpu)

        # Step 1.2: Parse wellbore and period metadata
        well, period, used_page, debug = self._find_metadata_from_pages(out_dir)
        period_start = period[0].isoformat() if period else None
        period_end   = period[1].isoformat() if period else None

        # Step 1.3: Run PaddleOCR for all tables (after all crops + layouts exist)
        ocr_debug = run_paddle_for_all_tables(out_dir, debug=False, skip_if_exists=True)
        debug["table_ocr"] = ocr_debug

        # ===================================================================
        # PHASE 2: Database Operations (Single Transaction)
        # ===================================================================
        
        with self.engine.begin() as conn:
            # Step 2.1: Insert/Update Document Metadata
            doc_id = upsert_ddr_document(
                conn,
                filename=filename,
                file_hash=file_hash,
                wellbore_name=well,
                period_start=period_start,
                period_end=period_end,
            )
            debug["doc_id"] = doc_id

            # Step 2.2: Parse and Insert All Table Data
            
            # Drilling Fluid (can be split across pages)
            try:
                df_rows = parse_drilling_fluid_rows(out_dir, document_id=doc_id)
                insert_ddr_drilling_fluid_rows(conn, document_id=doc_id, rows=df_rows)
                debug["drilling_fluid"] = {
                    "rows_parsed": len(df_rows),
                    "rows_inserted": len(df_rows),
                    "success": True
                }
            except Exception as e:
                debug["drilling_fluid"] = {
                    "rows_parsed": 0,
                    "rows_inserted": 0,
                    "success": False
                }
                debug.setdefault("errors", []).append({
                    "step": "drilling_fluid_parsing",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            # Operations
            try:
                ops_rows = parse_operations_rows(out_dir, document_id=doc_id)
                insert_ddr_operations_rows(conn, document_id=doc_id, rows=ops_rows)
                debug["operations"] = {
                    "rows_parsed": len(ops_rows),
                    "rows_inserted": len(ops_rows),
                    "success": True
                }
            except Exception as e:
                debug["operations"] = {
                    "rows_parsed": 0,
                    "rows_inserted": 0,
                    "success": False
                }
                debug.setdefault("errors", []).append({
                    "step": "operations_parsing",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            # Survey Station
            try:
                ss_rows = parse_survey_station_rows(out_dir, document_id=doc_id)
                insert_ddr_survey_station_rows(conn, document_id=doc_id, rows=ss_rows)
                debug["survey_station"] = {
                    "rows_parsed": len(ss_rows),
                    "rows_inserted": len(ss_rows),
                    "success": True
                }
            except Exception as e:
                debug["survey_station"] = {
                    "rows_parsed": 0,
                    "rows_inserted": 0,
                    "success": False
                }
                debug.setdefault("errors", []).append({
                    "step": "survey_station_parsing",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            # Stratigraphic Information
            try:
                si_rows = parse_stratigraphic_information_rows(out_dir, document_id=doc_id)
                insert_ddr_stratigraphic_information_rows(conn, document_id=doc_id, rows=si_rows)
                debug["stratigraphic_information"] = {
                    "rows_parsed": len(si_rows),
                    "rows_inserted": len(si_rows),
                    "success": True
                }
            except Exception as e:
                debug["stratigraphic_information"] = {
                    "rows_parsed": 0,
                    "rows_inserted": 0,
                    "success": False
                }
                debug.setdefault("errors", []).append({
                    "step": "stratigraphic_information_parsing",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            # Lithology Information
            try:
                li_rows = parse_lithology_information_rows(out_dir, document_id=doc_id)
                insert_ddr_lithology_information_rows(conn, document_id=doc_id, rows=li_rows)
                debug["lithology_information"] = {
                    "rows_parsed": len(li_rows),
                    "rows_inserted": len(li_rows),
                    "success": True
                }
            except Exception as e:
                debug["lithology_information"] = {
                    "rows_parsed": 0,
                    "rows_inserted": 0,
                    "success": False
                }
                debug.setdefault("errors", []).append({
                    "step": "lithology_information_parsing",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            # Gas Reading Information
            try:
                gr_rows = parse_gas_reading_information_rows(out_dir, document_id=doc_id)
                insert_ddr_gas_reading_information_rows(conn, document_id=doc_id, rows=gr_rows)
                debug["gas_reading_information"] = {
                    "rows_parsed": len(gr_rows),
                    "rows_inserted": len(gr_rows),
                    "success": True
                }
            except Exception as e:
                debug["gas_reading_information"] = {
                    "rows_parsed": 0,
                    "rows_inserted": 0,
                    "success": False
                }
                debug.setdefault("errors", []).append({
                    "step": "gas_reading_information_parsing",
                    "error": str(e),
                    "error_type": type(e).__name__
                })

            # Step 2.3: Parse and Insert Summary Report (structured tables)
            self._write_summary_report(conn, out_dir, doc_id, debug)

            # Step 2.4: Extract and Insert Activity Summaries (plain text)
            self._upsert_activity_summary(conn, out_dir, doc_id, debug)

            # Step 2.5: Final Updates (report_number from summary if available)
            report_number = debug.get("summary_report_payload", {}).get("report_number")
            if report_number is not None:
                update_report_number(conn, document_id=doc_id, report_number=report_number)

        # ===================================================================
        # PHASE 3: Return Results
        # ===================================================================
        
        return IngestResult(
            doc_id,
            well,
            period_start,
            period_end,
            used_page,
            debug, 
        )
