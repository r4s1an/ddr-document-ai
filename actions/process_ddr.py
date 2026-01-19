# actions/process_ddr.py
from dataclasses import dataclass
from typing import Callable, Optional

from actions.save_ddr import (
    SaveDDRAction,
    SaveDDRSummaryAction,
    SaveDDRActivitySummaryAction,
    SaveDDROperationsAction,
)

@dataclass
class ProcessDDRResult:
    status: str                 
    document_id: int
    inserted_operations: int = 0

class ProcessDDRAction:
    """
    Orchestrates the full DDR processing pipeline inside a single DB transaction.
    You can pass a logger callback to update Streamlit UI (status_box.info/success/warning).
    """

    def __init__(self, engine):
        self.engine = engine
        self.save_doc = SaveDDRAction(engine)
        self.save_summary = SaveDDRSummaryAction(engine)
        self.save_activity = SaveDDRActivitySummaryAction(engine)
        self.save_ops = SaveDDROperationsAction(engine)

    def execute(
        self,
        filename: str,
        file_hash: str,
        file_bytes: bytes,
        debug: bool = False,
        log: Optional[Callable[[str, str], None]] = None,  # log(level, message)
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
            inserted = self.save_ops.execute(
                conn=conn,
                document_id=doc_id,
                file_bytes=file_bytes,
                debug=debug,
            )
            _log("success", f"Operations saved ({inserted} rows).")

            return ProcessDDRResult(status="created", document_id=doc_id, inserted_operations=inserted)
