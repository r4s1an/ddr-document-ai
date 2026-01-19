import pdfplumber

def normalize_for_match(s):
    return ''.join(s.split())

pdf_path = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_14_2008_06_14.pdf"

result = {
    "summary_activities_24h": None,
    "summary_planned_activities_24h": None,
}

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text() or ""
        if "summary report" not in text.lower():
            continue

        lines = text.splitlines()

        activities_header = "Summary of activities (24 Hours)"
        planned_header = "Summary of planned activities (24 Hours)"
        activities_header_norm = normalize_for_match(activities_header)
        planned_header_norm = normalize_for_match(planned_header)

        activities_idx = None
        planned_idx = None
        for i, line in enumerate(lines):
            line_norm = normalize_for_match(line)
            if line_norm == activities_header_norm:
                activities_idx = i
            elif line_norm == planned_header_norm:
                planned_idx = i

        if activities_idx is None or planned_idx is None:
            continue

        # Extract activities
        activities_lines = lines[activities_idx + 1 : planned_idx]
        result["summary_activities_24h"] = ' '.join(' '.join(activities_lines).split())

        # Extract planned
        next_idx = len(lines)
        for j in range(planned_idx + 1, len(lines)):
            clean_line = ' '.join(lines[j].split())
            if clean_line.istitle() and len(clean_line.split()) > 1:
                next_idx = j
                break

        planned_lines = lines[planned_idx + 1 : next_idx]
        result["summary_planned_activities_24h"] = ' '.join(' '.join(planned_lines).split())

        break

print(result)