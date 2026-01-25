import re
from datetime import datetime
from typing import Optional, Tuple

def parse_wellbore(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(
        r'Wellbore\s*:\s*([0-9/]+(?:-[A-Za-z0-9]+)*(?:\s+[A-Z]{1,3})?)\b',
        text,
        flags=re.IGNORECASE
    )
    return m.group(1).strip() if m else None

def _norm_time(t: str) -> str:
    # "00.00" -> "00:00"
    return t.replace(".", ":").strip()

def parse_period(text: str) -> Optional[Tuple[datetime, datetime]]:
    if not text:
        return None

    # Accept:
    # Period: YYYY-MM-DD [HH:MM or HH.MM] [ - or whitespace ] YYYY-MM-DD [time optional]
    m = re.search(
        r'Period\s*:\s*'
        r'(\d{4}-\d{2}-\d{2})'                # start date
        r'(?:\s+(\d{2}[:.]\d{2}))?'           # optional start time
        r'\s*(?:[-–—]\s*)?'                   # optional dash
        r'\s*'
        r'(\d{4}-\d{2}-\d{2})'                # end date
        r'(?:\s+(\d{2}[:.]\d{2}))?',          # optional end time
        text,
        flags=re.IGNORECASE
    )
    if not m:
        return None

    d1, t1, d2, t2 = m.group(1), m.group(2), m.group(3), m.group(4)
    t1 = _norm_time(t1) if t1 else "00:00"
    t2 = _norm_time(t2) if t2 else "00:00"

    start = datetime.fromisoformat(f"{d1} {t1}")
    end   = datetime.fromisoformat(f"{d2} {t2}")
    return start, end