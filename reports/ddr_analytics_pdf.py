from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

def _wrap_lines(text: str, max_chars: int) -> list[str]:
    """
    Wrap text to max_chars per line, respecting word boundaries.
    """
    if not text:
        return [""]

    out = []
    for paragraph in str(text).split("\n"):
        p = paragraph.strip()
        if not p:
            out.append("")
            continue
         
        words = p.split()
        current_line = ""
        
        for word in words:
             
            if current_line and len(current_line) + 1 + len(word) > max_chars:
                 
                if len(word) > max_chars:
                    if current_line:
                        out.append(current_line)
                        current_line = ""
                     
                    while len(word) > max_chars - 1:
                        out.append(word[:max_chars - 1] + "-")
                        word = word[max_chars - 1:]
                    current_line = word
                else:
                    out.append(current_line)
                    current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        if current_line:
            out.append(current_line)
    
    return out if out else [""]

def _normalize_summary(text: str) -> str:
    """Turn multi-line LLM output into a single flowing paragraph"""
    if not text:
        return ""
    s = " ".join(str(text).splitlines())
     
    s = " ".join(s.split())
    return s.strip()

def build_pdf_bytes(payload: dict, analytics: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    x = 16 * mm
    y = height - 16 * mm
    line_h = 5.2 * mm

    doc = payload.get("document", {}) or {}
    doc_id = doc.get("id", "")
    wellbore = doc.get("wellbore_name", "")
    period_start = doc.get("period_start", "")
    period_end = doc.get("period_end", "")
    filename = doc.get("source_filename", "")

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, "DDR Daily Analytics Report")
    y -= 2 * line_h

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Document ID: {doc_id}   Wellbore: {wellbore}")
    y -= line_h
    c.drawString(x, y, f"Period: {period_start}  →  {period_end}")
    y -= line_h
    c.drawString(x, y, f"Source file: {filename}")
    y -= 1.5 * line_h

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Daily Short Summary")
    y -= line_h

    c.setFont("Helvetica", 10)
    summary_raw = (analytics.get("daily_short_summary") or "")
    summary = _normalize_summary(summary_raw) or "(No summary returned.)"
    for ln in _wrap_lines(summary, max_chars=110):
        if y < 55 * mm:
            break
        c.drawString(x, y, ln)
        y -= line_h

    y -= 0.8 * line_h

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Detected Events & Anomalies")
    y -= line_h

     
    col_idx_x = x
    col_time_x = x + 10 * mm
    col_event_x = x + 38 * mm
    col_anom_x = x + 85 * mm   
    col_sev_x = x + 170 * mm

    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_idx_x, y, "Idx")
    c.drawString(col_time_x, y, "Time")
    c.drawString(col_event_x, y, "Event")
    c.drawString(col_anom_x, y, "Anomalies")
    c.drawString(col_sev_x, y, "Sev")
    y -= line_h * 0.9

    c.setFont("Helvetica", 9)

    ops = payload.get("operations", []) or []
    events = analytics.get("events", []) or []

    ev_map = {}
    for e in events:
        if isinstance(e, dict) and "op_row_index" in e:
            ev_map[int(e["op_row_index"])] = e

    row_line_h = line_h * 0.85   

     
    IDX_CH = 3
    TIME_CH = 20
    EVENT_CH = 35   
    ANOM_CH = 50    
    SEV_CH = 8

    for i in range(len(ops)):
        if y < 22 * mm:
            break

        op = ops[i]
        e = ev_map.get(i, {})

        start_t = str(op.get("start_time", "") or "")
        end_t = str(op.get("end_time", "") or "")
        time_s = (start_t + "-" + end_t).strip("-")

        event_type = str(e.get("event_type", "")) or str(op.get("main_activity", "") or "OTHER")

        anomalies_list = e.get("anomalies", [])
        sev = "NONE"
        anomalies_s = ""

        if isinstance(anomalies_list, list) and anomalies_list:
            labels = []
            severities = []
            for a in anomalies_list:
                if isinstance(a, dict):
                    lab = str(a.get("label", "")).strip()
                    if lab:
                        labels.append(lab)
                    s = str(a.get("severity", "")).strip()
                    if s:
                        severities.append(s)

            anomalies_s = ", ".join(labels)
            sev = " / ".join(severities) if severities else "MED"

        idx_lines = _wrap_lines(str(i), IDX_CH)
        time_lines = _wrap_lines(time_s, TIME_CH)
        event_lines = _wrap_lines(event_type, EVENT_CH)
        anom_lines = _wrap_lines(anomalies_s, ANOM_CH)
        sev_lines = _wrap_lines(sev, SEV_CH)

        max_lines = max(len(idx_lines), len(time_lines), len(event_lines), 
                       len(anom_lines), len(sev_lines))
         
        if y - (max_lines - 1) * row_line_h < 22 * mm:
            break
         
        for li in range(max_lines):
            yy = y - li * row_line_h
            c.drawString(col_idx_x, yy, idx_lines[li] if li < len(idx_lines) else "")
            c.drawString(col_time_x, yy, time_lines[li] if li < len(time_lines) else "")
            c.drawString(col_event_x, yy, event_lines[li] if li < len(event_lines) else "")
            c.drawString(col_anom_x, yy, anom_lines[li] if li < len(anom_lines) else "")
            c.drawString(col_sev_x, yy, sev_lines[li] if li < len(sev_lines) else "")
         
        y -= max_lines * row_line_h + (row_line_h * 0.25)

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(x, 12 * mm, f"Generated: {datetime.now().isoformat(timespec='seconds')}")

    c.showPage()
    c.save()
    return buf.getvalue()