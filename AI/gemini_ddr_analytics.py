import json
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Gemini (Google) key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

# Use a real Gemini model ID (examples: gemini-2.5-flash, gemini-2.0-flash)
MODEL_NAME_DEFAULT = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

GEMINI_ENDPOINT_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of the FIRST valid JSON object from model text.
    1) Try full json.loads
    2) If fails, scan for balanced {...} and parse the first valid object.
    """
    if not text:
        return None

    s = text.strip()

    # 1) strict parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) balanced-brace scan
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    # continue scanning for next object
                    nxt = s.find("{", i + 1)
                    if nxt == -1:
                        return None
                    start = nxt
                    depth = 0
                    in_str = False
                    esc = False

    return None


def run_ddr_analytics(payload: dict, model_name: str = MODEL_NAME_DEFAULT) -> dict:
    """
    Gemini version.
    Input: payload from fetch_ddr_payload()
    Output: STRICT JSON dict
      {
        "daily_short_summary": "...",
        "events": [
          {
            "op_row_index": 0,
            "event_type": "string",
            "evidence": "1-2 sentences why this event type",
            "anomalies": [
              {"label": "string", "severity": "LOW|MED|HIGH"}
            ]
          }
        ]
      }
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY (or API_KEY).")

    url = GEMINI_ENDPOINT_TMPL.format(model=model_name, key=GEMINI_API_KEY)

    system_msg = (
        "You are a drilling operations analyst.\n"
        "Return STRICT JSON ONLY. No markdown. No extra text.\n"
        "Do NOT invent facts. Use only the provided data.\n"
        "Output MUST include exactly one events item per operations row.\n"
        "If a row has no anomaly: anomalies must be []."
    )

    user_msg = """
You will receive structured DDR data from SQL tables:
- document metadata
- summary texts
- operations (time intervals with main_activity, sub_activity, state, remark)
- optional gas readings, drilling fluid, survey, lithology, stratigraphy

TASKS:
1) Produce a daily summary (10-16 lines max).
2) For EACH operations row, classify the event type (free-text label).
3) Detect anomalies (if any) for that row. For EACH anomaly give:
   - label (free text)
   - severity: LOW / MED / HIGH

Return STRICT JSON in this schema exactly:

{
  "daily_short_summary": "string",
  "events": [
    {
      "op_row_index": int,
      "event_type": "string",
      "evidence": "string",
      "anomalies": [
        {
          "label": "string",
          "severity": "LOW|MED|HIGH"
        }
      ]
    }
  ]
}
""".strip()

    # Gemini request format
    req_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system_msg},
                    {"text": user_msg},
                    {"text": "DDR_PAYLOAD_JSON:\n" + json.dumps(payload, ensure_ascii=False)},
                ],
            }
        ],
        "generationConfig": {
            # Keep this on — it helps a lot
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 8000,
        },
    }

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=req_payload,
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Gemini text location
    try:
        text_out = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
    except Exception:
        text_out = ""

    parsed = _extract_json_object(text_out)

    # One deterministic retry if non-JSON
    if not parsed:
        req_payload2 = dict(req_payload)
        req_payload2["generationConfig"] = dict(req_payload["generationConfig"])
        req_payload2["generationConfig"]["temperature"] = 0.0

        resp2 = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=req_payload2,
            timeout=60,
        )
        if resp2.status_code != 200:
            raise RuntimeError(f"Gemini retry error {resp2.status_code}: {resp2.text}")

        data2 = resp2.json()
        text2 = (
            data2.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        parsed = _extract_json_object(text2)

    if not parsed:
        raise ValueError(f"Gemini returned non-JSON. Raw text (first 2000 chars): {text_out[:2000]}")

    # Minimal validation (MVP)
    if "daily_short_summary" not in parsed or "events" not in parsed:
        raise ValueError("Model output missing keys: daily_short_summary/events")

    return parsed
