from typing import Optional, List, Tuple
import re


def split_main_sub_activity(cell: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Splits 'Main -- Sub' activity cell into (main, sub).
    Removes newlines WITHOUT adding spaces (sidetra\\nck → sidetrack).
    """
    if not cell:
        return None, None

    # Remove newlines completely
    text = cell.replace("\n", "")

    # Normalize multiple spaces (but do not add new ones)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if "--" in text:
        main, sub = text.split("--", 1)
        return main.strip(), sub.strip()

    return text.strip(), None

def _normalize_nullable_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None

    t = s.strip()

    # collapse whitespace (optional)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # treat common "empty" markers as NULL
    if t == "":
        return None

    # whole-string markers
    if t.lower() in {"none", "null", "n/a", "na", "nil", "-", "--"}:
        return None

    # sometimes PDFs contain just "None." or "(None)"
    if re.fullmatch(r"[\(\[]?\s*none\s*[\)\]]?\.?", t, flags=re.IGNORECASE):
        return None

    return t

def norm(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def find_y_of_text(page, target: str):
    """
    Return the 'top' y-coordinate of the first occurrence of target text.
    Supports single- and multi-word targets.
    """
    target = target.lower()
    words = page.extract_words(use_text_flow=True) or []

    tokens = [w["text"].lower() for w in words]
    target_tokens = target.split()

    for i in range(len(tokens)):
        if tokens[i:i + len(target_tokens)] == target_tokens:
            return words[i]["top"]

    return None

def table_to_kv(table):
    """
    Convert a 2-column table to a key-value dict.
    Keeps last non-null value for duplicated keys.
    """
    result = {}

    for row in table:
        if not row or len(row) < 2:
            continue

        key = norm(row[0])
        val = norm(row[1])

        if key is None:
            continue

        result[key] = val

    return result

def map_table_by_order(table, columns):
    """
    Map table rows to predefined columns by row order.
    Extra rows are ignored. Missing rows become NULL.
    """
    mapped = {col: None for col in columns}

    for idx, row in enumerate(table):
        if idx >= len(columns):
            break

        if not row or len(row) < 2:
            continue

        value = norm(row[1])
        mapped[columns[idx]] = value

    return mapped

def drop_empty_rows(table):
    cleaned = []
    for row in table:
        if not row:
            continue
        # strip and keep only non-empty rows
        cells = [(c or "").strip() for c in row]
        if any(cells):
            cleaned.append(row)
    return cleaned

def extract_text_between(page, y_start: float, y_end: float) -> str:
    """
    Extract text between two vertical Y coordinates.
    """
    words = page.extract_words(use_text_flow=True) or []

    lines = {}
    for w in words:
        top = w["top"]
        if y_start < top < y_end:
            key = round(top, 1)  # group by visual line
            lines.setdefault(key, []).append(w["text"])

    return "\n".join(
        " ".join(words) for _, words in sorted(lines.items())
    ).strip()

def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _cluster_words_into_lines(words: List[dict], y_tol: float = 2.5) -> List[dict]:
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: List[List[dict]] = []

    for w in words:
        placed = False
        for line in lines:
            if abs(line[0]["top"] - w["top"]) <= y_tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    out = []
    for line in lines:
        line = sorted(line, key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line if w.get("text"))
        out.append({
            "top": float(line[0]["top"]),
            "x0": float(min(w["x0"] for w in line)),
            "text": text.strip()
        })
    out.sort(key=lambda d: (d["top"], d["x0"]))
    return out

def _is_headerish(line_text: str) -> bool:
    t = line_text.strip()
    if not t:
        return False
    if len(t) > 60: 
        return False

    # reject lines with many digits (tables)
    digit_ratio = sum(ch.isdigit() for ch in t) / max(len(t), 1)
    if digit_ratio > 0.15:
        return False

    # allow common header punctuation
    if not re.fullmatch(r"[A-Za-z0-9 ()/&\-]+", t):
        return False

    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)

    return (upper_ratio > 0.8) or (t == t.title())

def find_next_header_line_y(
    page,
    y_after: float,
    x0_max: float = 160.0,
    stop_headers: Optional[List[str]] = None,
) -> Optional[Tuple[float, str]]:
    """
    Find the next header-like line below y_after.
    If stop_headers provided, prefer the earliest match among:
      - explicit stop_headers (by text match, robust normalization)
      - generic headerish lines
    """
    stop_headers = stop_headers or []

    words = page.extract_words(use_text_flow=True) or []
    lines = _cluster_words_into_lines(words, y_tol=2.5)

    # Pre-normalize stop headers for robust compare
    stop_norm = [_norm_token(h) for h in stop_headers]

    candidates = []
    for ln in lines:
        if ln["top"] <= y_after + 2:
            continue
        if ln["x0"] > x0_max:
            continue

        ln_norm = _norm_token(ln["text"])

        # If this line matches an explicit stop header, accept immediately as candidate
        if stop_norm and ln_norm in stop_norm:
            candidates.append((ln["top"], ln["text"], 0))  # priority 0
            continue

        # Otherwise accept generic headerish lines
        if _is_headerish(ln["text"]):
            candidates.append((ln["top"], ln["text"], 1))  # priority 1

    if not candidates:
        return None

    # Sort by y, then priority (explicit stop header wins if same y)
    candidates.sort(key=lambda t: (t[0], t[2]))
    top, text, _prio = candidates[0]
    return top, text

def _strip_leading_header(text: str, header: str) -> str:
    # remove a header line if it appears at the start of extracted block
    return re.sub(rf"^\s*{re.escape(header)}\s*\n?", "", text, flags=re.IGNORECASE).strip()

def _strip_trailing_headers(text: str, headers: List[str]) -> str:
    # remove any of these headers if they accidentally appear inside the extracted text
    for h in headers:
        # remove header and anything after it (common when crop end is a bit late)
        text = re.split(rf"\n?\s*{re.escape(h)}\s*\n?", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return text