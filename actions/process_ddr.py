# actions/process_ddr.py
from dataclasses import dataclass
from typing import Callable, Optional

# ✅ 1. ADD IMPORT HERE (Assuming you saved the new class in actions.save_ddr)
from actions.save_ddr import (
    SaveDDRAction,
    SaveDDRSummaryAction,
    SaveDDRActivitySummaryAction,
    SaveDDROperationsAction, 
    SaveDDRDrillingFluidAction,
    SaveDDRPorePressureAction,
    SaveDDRSurveyStation
)

@dataclass
class ProcessDDRResult:
    status: str                 
    document_id: int
    inserted_operations: int = 0
    inserted_fluids: int = 0
    inserted_pore_pressure: int = 0
    inserted_survey_station: int = 0

class ProcessDDRAction:
    """
    Orchestrates the full DDR processing pipeline inside a single DB transaction.
    """

    def __init__(self, engine):
        self.engine = engine
        self.save_doc = SaveDDRAction(engine)
        self.save_summary = SaveDDRSummaryAction(engine)
        self.save_activity = SaveDDRActivitySummaryAction(engine)
        self.save_ops = SaveDDROperationsAction(engine)
        self.save_fluids = SaveDDRDrillingFluidAction(engine)
        self.save_pore_pressure = SaveDDRPorePressureAction(engine)
        self.save_survey_station = SaveDDRSurveyStation(engine)

    def execute(
        self,
        filename: str,
        file_hash: str,
        file_bytes: bytes,
        debug: bool = False,
        log: Optional[Callable[[str, str], None]] = None,
    ) -> ProcessDDRResult:

        def _log(level: str, msg: str):
            if log:
                log(level, msg)

        with self.engine.begin() as conn:
            _log("info", "Saving DDR document...")
            doc_result = self.save_doc.execute_with_conn(
                conn=conn,
                filename=filename,
                file_hash=file_hash,
                file_bytes=file_bytes,
            )

            doc_id = doc_result["document_id"]

            if doc_result["status"] == "duplicate":
                _log("warning", f"This file already exists (document_id={doc_id}). Skipping extraction.")
                return ProcessDDRResult(status="duplicate", document_id=doc_id)

            _log("success", "DDR document saved.")

            _log("info", "Extracting and saving summary report...")
            self.save_summary.execute(conn=conn, document_id=doc_id, file_bytes=file_bytes)
            _log("success", "Summary report saved.")

            _log("info", "Extracting and saving activity summaries (24 Hours)...")
            self.save_activity.execute(
                conn=conn,
                document_id=doc_id,
                file_bytes=file_bytes,
                debug=debug,
            )
            _log("success", "Activity summaries saved.")

            _log("info", "Extracting and saving Operations rows...")
            inserted_ops = self.save_ops.execute(
                conn=conn,
                document_id=doc_id,
                file_bytes=file_bytes,
                debug=debug,
            )
            _log("success", f"Operations saved ({inserted_ops} rows).")

            _log("info", "Extracting and saving Drilling Fluid data...")
            inserted_fluids = self.save_fluids.execute(
                conn=conn,
                document_id=doc_id,
                file_bytes=file_bytes,
                debug=debug,
            )
            _log("success", f"Drilling Fluid saved ({inserted_fluids} rows).")

            _log("info", "Extracting and saving Pore Pressure data...")
            inserted_pp = self.save_pore_pressure.execute(
                conn=conn,
                document_id=doc_id,
                file_bytes=file_bytes,
                debug=debug,
            )
            _log("success", f"Pore Pressure saved ({inserted_pp} rows).")

            _log("info", "Extracting and saving Survey Station data...")
            inserted_ss = self.save_survey_station.execute_with_conn(
                conn=conn,
                document_id=doc_id,
                file_bytes=file_bytes,
                debug=debug,
            )["inserted"] 

            _log("success", f"Survey Station saved ({inserted_ss} rows).")

            return ProcessDDRResult(
                status="created", 
                document_id=doc_id, 
                inserted_operations=inserted_ops,
                inserted_fluids=inserted_fluids,
                inserted_pore_pressure=inserted_pp,
                inserted_survey_station=inserted_ss
            )