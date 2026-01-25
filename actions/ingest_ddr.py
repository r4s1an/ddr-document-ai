from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import streamlit as st
from tables.summary.orchestrator import SummaryReportParser
from sqlalchemy import text
from services.layout_graph import build_and_save_layout
from services.post_layout_ocr import ocr_layout_json
from services.ddr_metadata import parse_wellbore, parse_period
from tables.utils import coerce_summary_payload
from services.activity_summary_extractor import extract_activity_summaries_from_processed_ddr
from tables.db_writes import (
    upsert_ddr_document,
    upsert_ddr_activity_summary,
    update_report_number,
)
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
            # collect OCR text from wellbore_field + period_field first
            wb_texts = [it.get("ocr_text","") for it in data["items"] if it["label"] == "wellbore_field"]
            per_texts = [it.get("ocr_text","") for it in data["items"] if it["label"] == "period_field"]
            # also allow fallback: search in all text on the page (headers/plain_text)
            all_texts = [it.get("ocr_text","") for it in data["items"] if it.get("ocr_text")]

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
        summaries = extract_activity_summaries_from_processed_ddr(
            processed_ddr_dir=out_dir,
            header_match_threshold=0.78,
            debug=debug.get("debug_activity", False),
        )
        debug["activity_summary"] = {
            "activities_24h": {
                "is_null": summaries.get("activities_24h_text") is None,
                "length": len(summaries["activities_24h_text"]) if summaries.get("activities_24h_text") else 0,
                "preview": (
                    summaries["activities_24h_text"][:200]
                    if summaries.get("activities_24h_text")
                    else None
                ),
            },
            "planned_24h": {
                "is_null": summaries.get("planned_24h_text") is None,
                "length": len(summaries["planned_24h_text"]) if summaries.get("planned_24h_text") else 0,
                "preview": (
                    summaries["planned_24h_text"][:200]
                    if summaries.get("planned_24h_text")
                    else None
                ),
            },
        }
        upsert_ddr_activity_summary(
            conn,
            document_id=document_id,
            activities_24h_text=summaries.get("activities_24h_text", ""),
            planned_24h_text=summaries.get("planned_24h_text", ""),
        )

        debug["activity_summary_inserted"] = True
        debug["activity_summary_lengths"] = {
            "activities_24h": len(summaries.get("activities_24h_text", "") or ""),
            "planned_24h": len(summaries.get("planned_24h_text", "") or ""),
        }
    def _upsert_summary_report(self, conn, out_dir: Path, document_id: int, debug: Dict[str, Any]) -> None:
        """
        Find Summary report tables (3 tables), run PaddleOCR, parse left/middle/right,
        and UPSERT into ddr_summary_report.
        """
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

        page_dirs = sorted(out_dir.glob("page_*"))
        found_any = False

        for pd in page_dirs:
            layout_path = pd / "layout.json"
            ocr_path = pd / "layout_ocr.json"
            if not layout_path.exists() or not ocr_path.exists():
                continue

            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))

            # build idx -> ocr_item map
            ocr_by_idx = {it["idx"]: it for it in ocr.get("items", [])}

            # find section headers whose OCR text is "summary report"
            summary_header_idxs = set()
            for it in ocr.get("items", []):
                if it.get("label") == "section_header":
                    txt = (it.get("ocr_text") or "").strip().lower()
                    if txt == "summary report":
                        summary_header_idxs.add(it["idx"])

            if not summary_header_idxs:
                continue

            # collect tables linked to that header
            summary_tables = []
            for it in layout.get("items", []):
                if it.get("label") != "table":
                    continue
                hdr_idx = it.get("linked_header_idx")
                if hdr_idx in summary_header_idxs:
                    summary_tables.append(it)

            if not summary_tables:
                continue

            # We expect exactly 3; if not, log and skip this page
            if len(summary_tables) != 3:
                debug.setdefault("summary_warnings", []).append({
                    "page": pd.name,
                    "reason": f"expected 3 tables, got {len(summary_tables)}",
                })
                continue

            parser = SummaryReportParser()
            payload = parser.extract(document_id=document_id, table_items=summary_tables)
            payload = coerce_summary_payload(payload)
            conn.execute(insert_sql, payload)

            debug["summary_report_inserted"] = True
            debug["summary_report_payload"] = payload
            found_any = True
            break

        if not found_any:
            debug["summary_report_inserted"] = False
            
    def execute(self, filename: str, file_hash: str, out_dir: Path) -> IngestResult:
        # 1) build layout + OCR for ALL pages
        page_dirs = sorted(out_dir.glob("page_*"))
        for pd in page_dirs:
            build_and_save_layout(pd)
            ocr_layout_json(pd, gpu=self.ocr_gpu)

        # 2) parse wellbore/period
        well, period, used_page, debug = self._find_metadata_from_pages(out_dir)
        period_start = period[0].isoformat() if period else None
        period_end   = period[1].isoformat() if period else None

        # 3) insert document row first (without report_number)
        with self.engine.begin() as conn:
            # 3) Insert/Upsert document row
            doc_id = upsert_ddr_document(
                conn,
                filename=filename,
                file_hash=file_hash,
                wellbore_name=well,
                period_start=period_start,
                period_end=period_end,
            )
            debug["doc_id"] = doc_id

            # 4) Summary report (tables)
            self._upsert_summary_report(conn, out_dir, doc_id, debug)

            # 5) Activity summaries (plain_text under headers)
            self._upsert_activity_summary(conn, out_dir, doc_id, debug)

            # 6) Update report_number if extracted
            report_number = debug.get("summary_report_payload", {}).get("report_number")
            if report_number is not None:
                update_report_number(conn, document_id=doc_id, report_number=report_number)

                # 4) Return once, after DB work is done
                return IngestResult(
                    doc_id,
                    well,
                    period_start,
                    period_end,
                    used_page,
                    debug,  # already includes reused + summary status
                )
