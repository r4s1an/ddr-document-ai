from pathlib import Path
from tables.summary.left_table import SummaryLeftTableParser
from tables.utils import coerce_summary_payload

if __name__ == "__main__":
    parser = SummaryLeftTableParser()

    # replace this with a REAL PaddleOCR JSON file
    json_path = Path(r"C:\Users\Yoked\Desktop\DDR Processor\processed_ddr\page_001\summary_tables\left\001_table_res.json")
    payload = parser.extract(json_path)

    print("\n=== RAW PAYLOAD FROM SummaryLeftTableParser ===")
    for k, v in payload.items():
        print(f"{k:25s} -> {v!r} ({type(v)})")

    payload = coerce_summary_payload(payload)
    print("\n=== changed PAYLOAD FROM SummaryLeftTableParser ===")
    for k, v in payload.items():
        print(f"{k:25s} -> {v!r} ({type(v)})")
