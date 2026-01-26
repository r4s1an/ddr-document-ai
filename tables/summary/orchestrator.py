from pathlib import Path

from tables.paddle_runner import run_paddle_ocr
from tables.summary.left_table import SummaryLeftTableParser
from tables.summary.middle_table import SummaryMiddleTableParser
from tables.summary.right_table import SummaryRightTableParser

class SummaryReportParser:
    def extract(self, document_id: int, table_items: list[dict]) -> dict:
        tables = sorted(table_items, key=lambda t: t["x1"])
        if len(tables) != 3:
            raise ValueError(f"Summary report must have exactly 3 tables, got {len(tables)}")

        left, middle, right = tables

        left_crop = Path(left["crop_path"])
        mid_crop  = Path(middle["crop_path"])
        right_crop= Path(right["crop_path"])

        base_dir = left_crop.parent.parent / "summary_tables"

        left_json  = run_paddle_ocr(left_crop,  base_dir / "left")
        mid_json   = run_paddle_ocr(mid_crop,   base_dir / "middle")
        right_json = run_paddle_ocr(right_crop, base_dir / "right")

         
        assert isinstance(left_json, (str, Path)), f"left_json wrong type: {type(left_json)}"
        assert isinstance(mid_json, (str, Path)), f"mid_json wrong type: {type(mid_json)}"
        assert isinstance(right_json, (str, Path)), f"right_json wrong type: {type(right_json)}"

        payload = {}
        payload.update(SummaryLeftTableParser().extract(left_json))
        payload.update(SummaryMiddleTableParser().extract(mid_json))
        payload.update(SummaryRightTableParser().extract(right_json))

        payload["document_id"] = document_id
        return payload
