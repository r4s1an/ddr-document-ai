import pdfplumber
import re

PDF_PATH = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_14_2008_06_14.pdf"
def undouble(s: str) -> str:
    """Fix tokens like 'TTiimmee' -> 'Time' (every char duplicated)."""
    if not s:
        return ""
    s = str(s).strip()
    if len(s) >= 4 and len(s) % 2 == 0:
        a, b = s[::2], s[1::2]
        if a == b:
            return a
    return s

def clean_cell(s: str) -> str:
    # ✅ CHANGE THIS FUNCTION: apply BOTH cleaners
    s = undouble(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def extract_pore_pressure_table(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):

            text = page.extract_text() or ""
            if "Pore Pressure" not in text:
                continue

            tables = page.extract_tables()
            if not tables:
                continue

            # pick the table whose header contains "time" and "reading"
            for t in tables:
                if not t or not t[0]:
                    continue

                header_raw = t[0]
                header = [clean_cell(c) for c in header_raw]
                header_n = [norm(h) for h in header]

                if any("time" == h or "time" in h for h in header_n) and any("reading" in h for h in header_n):
                    # clean all rows
                    cleaned = [[clean_cell(c) for c in row] for row in t]

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

result = extract_pore_pressure_table(PDF_PATH)

if not result:
    print("❌ Pore Pressure table not found.")
else:
    print(f"✅ Pore Pressure table found on page {result['page']}")
    print("Header:", result["header"])
    print("First row:", result["rows"][0] if result["rows"] else "(no data rows)")