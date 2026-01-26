import json
import os
from typing import Any, Dict, Optional
import requests

GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash" 

DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 1
DEFAULT_MAX_OUTPUT_TOKENS = 2048

MAX_STR_CHARS = 8000
MAX_LIST_ROWS_PER_QUERY = 200   

def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)

def _truncate_str(s: str, max_chars: int = MAX_STR_CHARS) -> str:
    if s is None:
        return s
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...(truncated,{len(s)} chars)"

def _shrink_payload(obj: Any) -> Any:
    """
    Ensure payload stays compact:
    - truncate long strings (raw_json can be massive)
    - clamp long row lists again (belt & suspenders)
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        return _truncate_str(obj)

    if isinstance(obj, list):
         
        out = obj[:MAX_LIST_ROWS_PER_QUERY]
        return [_shrink_payload(x) for x in out]

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
             
            if k in ("raw_json", "interpretation") and isinstance(v, str):
                out[k] = _truncate_str(v)
            else:
                out[k] = _shrink_payload(v)
        return out

    return obj

def _build_prompt(question: str, retrieved_payload: Dict[str, Any]) -> str:
    payload_small = _shrink_payload(retrieved_payload)

    return (
        "User question:\n"
        f"{question}\n\n"
        "Retrieved SQL payload (ground truth):\n"
        f"{_safe_json_dumps(payload_small)}\n\n"
        "Return ONE JSON object only with keys exactly:\n"
        "answer, assumptions_or_limits, followups\n"
    )

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of the FIRST valid JSON object from text.
    1) Try full-text json.loads
    2) If fails, scan for a balanced {...} object and try parsing each candidate.
    """
    if not text:
        return None

    s = text.strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
     
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
                     
                     
                    nxt = s.find("{", i + 1)
                    if nxt == -1:
                        return None
                    start = nxt
                    depth = 0
                    in_str = False
                    esc = False

    return None

def _validate_response_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure required keys exist and types are sane.
    If something is missing, fill with safe defaults.
    """
    if not isinstance(obj, dict):
        return {
            "answer": "Model returned an invalid response (not a JSON object).",
            "sql_used": [],
            "tables_preview": {},
            "assumptions_or_limits": ["Model output was not a valid JSON object."],
            "followups": [],
        }
     
    obj.setdefault("answer", "")
    obj.setdefault("assumptions_or_limits", [])
    obj.setdefault("followups", [])
     
    if not isinstance(obj["answer"], str):
        obj["answer"] = str(obj["answer"])

    if not isinstance(obj["assumptions_or_limits"], list):
        obj["assumptions_or_limits"] = [str(obj["assumptions_or_limits"])]

    if not isinstance(obj["followups"], list):
        obj["followups"] = [str(obj["followups"])]

    return obj

def generate_tag_answer(
    question: str,
    retrieved_payload: Dict[str, Any],
    model_name: str = GEMINI_DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_s: int = 30,
) -> Dict[str, Any]:
    """
    Calls Gemini and returns STRICT JSON dict.
    IMPORTANT: does NOT execute SQL; only uses retrieved_payload.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not key:
        return {
            "answer": "Missing API key. Set GEMINI_API_KEY (or API_KEY) in environment.",
            "sql_used": [],
            "tables_preview": {},
            "assumptions_or_limits": ["No Gemini API key found in environment."],
            "followups": ["Set GEMINI_API_KEY and retry."],
        }

    model = model_name or GEMINI_DEFAULT_MODEL
    url = GEMINI_ENDPOINT_TMPL.format(model=model)

    prompt_text = _build_prompt(question=question, retrieved_payload=retrieved_payload)

    system_text = (
           "You are a read-only TAG assistant. "
        "Return STRICT JSON ONLY. No markdown. No extra text. "
        "Keys exactly: answer, assumptions_or_limits, followups. "
        "Use ONLY the retrieved payload. If unclear, ask in followups.")

    body = {
         
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": [
            {"role": "user", "parts": [{"text": prompt_text}]}
        ],
        "generationConfig": {
             
            "responseMimeType": "application/json",
            "temperature": float(temperature),
            "topP": float(DEFAULT_TOP_P),
            "topK": int(DEFAULT_TOP_K),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    try:
        resp = requests.post(
            url,
            headers={
                "x-goog-api-key": key,               
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout_s,
        )

        if resp.status_code != 200:
            print("=== GEMINI API ERROR ===")
            print(resp.status_code, resp.reason)
            print(resp.text)
            print("=== END GEMINI ERROR ===")

        resp.raise_for_status()
        data = resp.json()

    except Exception as e:
        return {
            "answer": f"Gemini API error: {type(e).__name__}: {str(e)}",
            "sql_used": [],
            "tables_preview": {},
            "assumptions_or_limits": ["Gemini call failed."],
            "followups": [],
        }

     
    try:
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return {
            "answer": "Gemini returned no usable text.",
            "sql_used": [],
            "tables_preview": {},
            "assumptions_or_limits": [f"Unexpected response structure: keys={list(data.keys())}"],
            "followups": [],
        }

    parsed = _extract_json_object(text_out)

     
    if not parsed:
        try:
            body2 = dict(body)
            body2["generationConfig"] = dict(body["generationConfig"])
            body2["generationConfig"]["temperature"] = 0.0
            body2["generationConfig"]["topP"] = 1.0
            body2["generationConfig"]["topK"] = 1

            resp2 = requests.post(
                url,
                headers={
                    "x-goog-api-key": key,
                    "Content-Type": "application/json",
                },
                json=body2,
                timeout=timeout_s,
            )
            resp2.raise_for_status()
            data2 = resp2.json()

            text2 = data2["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _extract_json_object(text2)
        except Exception:
            parsed = None

    if not parsed:
        return {
            "answer": "Model returned a non-JSON response. I cannot use it safely.",
            "sql_used": [],
            "tables_preview": {},
            "assumptions_or_limits": [
                "Model output was not valid JSON after retries, so it was rejected to avoid hallucinations.",
            ],
            "followups": ["Try re-asking with a simpler question, or reduce requested scope."],
        }

    return _validate_response_schema(parsed)
