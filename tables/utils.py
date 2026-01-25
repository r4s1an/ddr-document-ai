import re
from typing import Any, Dict, Optional, List, Tuple

def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("\n", " ")).strip()

TS_FIELDS = {
    "report_creation_ts",
    "spud_ts",
    "date_well_complete",
}

def coerce_timestamp(v: Any) -> Optional[str]:
    if v is None:
        return None

    if isinstance(v, int):
        print(f"[DEBUG] coercing int: {v}") 
        v = str(v)

    s = str(v).strip()

    # Reject pure year (still unsafe)
    if re.fullmatch(r"\d{4}", s):
        return None

    # Normalize single-digit month/day
    # 2023-6-4 -> 2023-06-04
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    # Normalize date + time without leading zeros
    # 2023-6-4 8:3 -> 2023-06-04 08:03
    m = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?",
        s
    )
    if m:
        y, mo, d, hh, mm, ss = m.groups()
        return (
            f"{y}-{int(mo):02d}-{int(d):02d} "
            f"{int(hh):02d}:{int(mm):02d}"
            + (f":{int(ss):02d}" if ss else "")
        )

    # Accept already well-formed ISO strings
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}(\s+\d{2}:\d{2}(:\d{2})?)?",
        s
    ):
        return s

    # DD/MM/YYYY or DD.MM.YYYY → ISO
    m = re.fullmatch(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", s)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    return None

def coerce_summary_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)

    # timestamps
    for k in TS_FIELDS:
        if k in out:
            out[k] = coerce_timestamp(out[k])
    return out

def parse_val(v: str):
    v = clean(v)
    if not v:
        return None
    
    # Preserve ISO-like date/timestamp patterns
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", v):
        return v
    
    # Preserve DD/MM/YYYY or DD.MM.YYYY patterns
    if re.match(r"^\d{1,2}[./]\d{1,2}[./]\d{4}", v):
        return v
    
    # pure numeric
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return float(v) if "." in v else int(v)
    
    # first numeric inside
    m = re.search(r"-?\d+(\.\d+)?", v)
    if m:
        s = m.group(0)
        return float(s) if "." in s else int(s)
    
    return v

def group_rows(
    rec_texts: List[str],
    rec_boxes: List[List[int]],
    row_threshold: int = 18,
) -> List[Tuple[int, List[Tuple[int, str]]]]:
    items = []
    for t, b in zip(rec_texts, rec_boxes):
        if not t or not b or len(b) < 4:
            continue
        x1, y1, *_ = map(int, b[:4])
        txt = clean(t)
        if txt:
            items.append((y1, x1, txt))

    items.sort(key=lambda z: (z[0], z[1]))

    rows = []
    cur = []
    cur_y = None

    for y1, x1, txt in items:
        if cur_y is None or abs(y1 - cur_y) <= row_threshold:
            cur.append((x1, txt))
            cur_y = y1 if cur_y is None else cur_y
        else:
            cur.sort(key=lambda z: z[0])
            rows.append((cur_y, cur))
            cur = [(x1, txt)]
            cur_y = y1

    if cur:
        cur.sort(key=lambda z: z[0])
        rows.append((cur_y or 0, cur))

    return rows

def split_label_value_row(row_text: str) -> Tuple[str, str]:

    s = clean(row_text).replace("：", ":")  
    if ":" in s:
        a, b = s.split(":", 1)
        return clean(a + ":"), clean(b)
    
    match = re.match(r"^(.+?)\s+([YN]|[A-Z]{1,3}|\d+(?:\.\d+)?)$", s, re.IGNORECASE)
    if match:
        label = clean(match.group(1)) + ":"  
        value = clean(match.group(2))
        return (label, value)

    return s, ""

def normalize_label(s: str) -> str:
    s = clean(s).replace("：", ":")
    s = re.sub(r"\([^)]*\)", "", s)  # removes "(m/h)" and "()"
    s = s.rstrip(":").strip().lower()
    return s