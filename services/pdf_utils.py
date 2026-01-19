import io
import pdfplumber
from services.pdf_utils import extract_text_between, drop_empty_rows, find_y_of_text

def extract_activity_summaries(pdf_path) -> dict:
    result = {
        "summary_activities_24h": None,
        "summary_planned_activities_24h": None,
    }

    def next_section_top_after(page, y_after):
        """
        Heuristic: next line after y_after whose font looks like a header
        (bold and/or larger than median body font size).
        """
        words = page.extract_words(
            use_text_flow=True,
            extra_attrs=["top", "x0", "size", "fontname"],
        ) or []
        if not words:
            return page.height

        # Body font estimate (median-ish): just sort sizes and take middle
        sizes = sorted([w.get("size", 0) or 0 for w in words if (w.get("size", 0) or 0) > 0])
        body_size = sizes[len(sizes) // 2] if sizes else 0

        # Group words into lines using rounded "top"
        lines = {}
        for w in words:
            if w["top"] <= y_after + 1:
                continue
            key = round(w["top"], 1)
            lines.setdefault(key, []).append(w)

        for top in sorted(lines.keys()):
            lw = sorted(lines[top], key=lambda x: x["x0"])
            line_text = " ".join(w["text"] for w in lw).strip()
            if not line_text:
                continue

            # Basic header style signals
            max_size = max((w.get("size", 0) or 0) for w in lw)
            any_bold = any("bold" in (w.get("fontname") or "").lower() for w in lw)

            # Skip obvious body-like lines (very long sentences)
            if len(line_text) > 120:
                continue

            # Decide "header-like"
            if any_bold or (body_size and max_size >= body_size + 1.0):
                return float(top)

        return page.height

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page = page.dedupe_chars(tolerance=1)
            text = page.extract_text() or ""

            if "summary report" not in text.lower():
                continue

            # headers (mind the spacing)
            y_activities = find_y_of_text(page, "Summar y of activities (24 Hours)")
            y_planned   = find_y_of_text(page, "Summar y of planned activities (24 Hours)")

            # IMPORTANT: don't use `if not y_...`
            if y_activities is None or y_planned is None:
                continue

            y_next_section = next_section_top_after(page, y_planned)

            result["summary_activities_24h"] = extract_text_between(page, y_activities, y_planned)
            result["summary_planned_activities_24h"] = extract_text_between(page, y_planned, y_next_section)
            break

    return result
print(extract_activity_summaries(r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_14_2008_06_14.pdf"))