import pdfplumber
import re
import sys
import io

# Force UTF-8 output for console (Æ, Ø, etc.)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# Your PDF path
pdf_path = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_10_2009_05_24.pdf"
# We'll only process the first page
metadata = {
    "wellbore": None,
    "period_start_date": None,
    "period_start_time": None,
    "period_end_date": None,
    "period_end_time": None,
    "report_number": None
}

with pdfplumber.open(pdf_path) as pdf:
    if len(pdf.pages) == 0:
        print("PDF has no pages!")
        sys.exit()

    # Only first page
    page = pdf.pages[0]
    print("Processing first page only...")

    # Fix duplicated bold characters
    deduped_page = page.dedupe_chars(tolerance=1)

    # Get clean text
    text = deduped_page.extract_text() or ""

    # ── Wellbore name ───────────────────────────────────────────────
    well_match = re.search(r'(Wellbore\s*:\s*)?([\d/]+[A-Za-z\s-]*\d+[A-Za-z]?)', text, re.IGNORECASE)
    if well_match and not metadata["wellbore"]:
        metadata["wellbore"] = well_match.group(2).strip()
        print(f"Found Wellbore: {metadata['wellbore']}")

    # ── Period with times (required format) ─────────────────────────
    period_match = re.search(
        r'Period:\s*'
        r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s*'
        r'[-–—]\s*'
        r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})',
        text,
        re.IGNORECASE
    )

    if period_match:
        metadata["period_start_date"] = period_match.group(1)
        metadata["period_start_time"] = period_match.group(2)
        metadata["period_end_date"]   = period_match.group(3)
        metadata["period_end_time"]   = period_match.group(4)

        print(f"Period start: {metadata['period_start_date']} {metadata['period_start_time']}")
        print(f"Period end:   {metadata['period_end_date']} {metadata['period_end_time']}")
    else:
        print("Period not found or format is different")

    # ── Report number ───────────────────────────────────────────────
    report_match = re.search(
        r'Report\s*(?:number|#)\s*[:=]\s*(\d+)',
        text,
        re.IGNORECASE
    )
    if report_match:
        metadata["report_number"] = report_match.group(1)
        print(f"Report number: {metadata['report_number']}")

# ── Final clean summary ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("EXTRACTED METADATA (first page only)")
print("═" * 60)
print(f"Wellbore          : {metadata['wellbore'] or 'Not found'}")
print(f"Period start      : {metadata['period_start_date'] or 'Not found'} "
      f"{metadata['period_start_time'] or ''}".strip())
print(f"Period end        : {metadata['period_end_date'] or 'Not found'} "
      f"{metadata['period_end_time'] or ''}".strip())
print(f"Report number     : {metadata['report_number'] or 'Not found'}")
print("═" * 60)