import pdfplumber
from typing import List, Dict

def extract_survey_station_table(
    pdf_path: str,
    debug: bool = False
) -> List[Dict[str, str]]:
    """
    Extracts the 'Survey Station' table that appears directly below
    the 'Survey Station' section header.
    """

    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").lower()

            # 1️⃣ Detect section header
            if "Stratigraphic Information" not in text:
                continue

            if debug:
                print(f"[PAGE {page_no}] 'Stratigraphic Infor mation' found")

            # 2️⃣ Deduplicate characters (fixes TTiiMMee issues)
            page = page.dedupe_chars(tolerance=1)

            # 3️⃣ Locate the header position
            words = page.extract_words(use_text_flow=True)

            header_word = next(
                (w for w in words if w["text"].lower() == "Stratigraphic"),
                None
            )

            if not header_word:
                continue

            header_bottom = header_word["bottom"]

            # 4️⃣ Crop area below header
            cropped = page.crop(
                (
                    0,
                    header_bottom + 5,
                    page.width,
                    page.height
                )
            )

            # 5️⃣ Extract tables from cropped area
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
                    print("❌ No tables found under Survey Station")
                continue

            table = tables[0]

            # 6️⃣ Normalize header
            headers = [h.strip() if h else "" for h in table[0]]

            # Expected columns sanity check
            if len(headers) < 4:
                if debug:
                    print("⚠️ Table detected but columns look wrong:", headers)
                continue

            # 7️⃣ Convert rows to dicts
            for row in table[1:]:
                if not any(row):
                    continue

                row_dict = {
                    headers[i]: (row[i] or "").strip()
                    for i in range(len(headers))
                }
                results.append(row_dict)

            break  # only one Survey Station section per PDF

    return results
pdf_path = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_14_2008_06_14.pdf"

rows = extract_survey_station_table(pdf_path, debug=True)

for r in rows:
    print(r)
