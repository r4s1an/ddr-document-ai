import pdfplumber, re
from typing import Optional, List, Tuple
from services.pdf_utils import split_main_sub_activity
from services.pdf_utils import find_y_of_text

OPERATIONS_HDR = "Operations"
from typing import Tuple, Optional


def extract_operations_table_first_below(
    page,
    debug: bool = True,
    dedupe: bool = True,
):
    """
    Find 'Operations' header y, then pick the first detected table whose bbox top is below that y.
    Returns the extracted table (list of rows) or None.
    """
    if dedupe:
        page = page.dedupe_chars(tolerance=1)

    y_ops = find_y_of_text(page, OPERATIONS_HDR)
    if y_ops is None:
        if debug:
            print("Operations header not found.")
        return None

    if debug:
        print(f"Operations y_start={y_ops:.2f}")

    # Find all tables and choose the first one below the Operations header
    tables = page.find_tables() or []
    if debug:
        print(f"find_tables() found {len(tables)} tables total on page")

    # Candidates: tables with top below Operations header
    candidates: List[Tuple[float, object]] = []
    for i, t in enumerate(tables):
        x0, top, x1, bottom = t.bbox
        if debug:
            print(f"  table #{i}: bbox=({x0:.1f},{top:.1f},{x1:.1f},{bottom:.1f})")
        if top >= y_ops - 2:  # small tolerance
            candidates.append((top, t))

    if not candidates:
        if debug:
            print("No table found below Operations header.")
        return None

    # pick nearest table below ops header
    candidates.sort(key=lambda x: x[0])
    top, table_obj = candidates[0]

    if debug:
        x0, top, x1, bottom = table_obj.bbox
        print(f"Selected table bbox=({x0:.1f},{top:.1f},{x1:.1f},{bottom:.1f})")

    return table_obj.extract()

def normalize_operations_table(raw_table):
    """
    Converts extracted Operations table into structured rows with
    separate main_activity and sub_activity columns.
    """
    if not raw_table or len(raw_table) < 2:
        return []

    header = raw_table[0]  # keep if needed
    rows = raw_table[1:]

    normalized_rows = []

    for r in rows:
        if len(r) < 6:
            continue  # defensive

        start_time = r[0]
        end_time = r[1]
        end_depth_mmd = r[2]

        main_activity, sub_activity = split_main_sub_activity(r[3])

        state = r[4]
        remark = " ".join(r[5].split()) if r[5] else None  # remove \n

        normalized_rows.append({
            "start_time": start_time,
            "end_time": end_time,
            "end_depth_mmd": end_depth_mmd,
            "main_activity": main_activity,
            "sub_activity": sub_activity,
            "state": state,
            "remark": remark,
        })

    return normalized_rows

# ---- quick runner ----
def test(pdf_path: str, page_number: int = 1):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        tbl = extract_operations_table_first_below(page, debug=True)
        normalized = normalize_operations_table(tbl)

        print("\n=== OPERATIONS TABLE (FIRST BELOW) ===")
        if not normalized:
            print("None")
            return

        for r in normalized:
            print(r)


if __name__ == "__main__":
    PDF_PATH = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_14_2008_06_14.pdf"
    test(PDF_PATH, page_number=1)