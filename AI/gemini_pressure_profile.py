import base64
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = "gemini-flash-latest"

URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent?key={API_KEY}"
)

def _encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def extract_pressure_profile(image_bytes: bytes, mime_type: str) -> dict:
    """
    Extracts pressure vs depth profile data.
    Returns STRICT JSON compatible with pressure_profile tables.
    """

    image_b64 = _encode_image_bytes(image_bytes)

    prompt = """
Analyze this engineering pressure profile plot (Pressure vs Depth).

Rules:
- Extract each curve separately.
- Extract each data point exactly once.
- Do NOT repeat points.
- Do NOT invent values.
- If a value is unclear, omit that point.

Axes:
- X-axis: Pressure (psi)
- Y-axis: Depth (may be TVD, TVDSS, or MD)

Tasks:
1. Identify all curves on the plot (e.g. Measured Pressure, Virgin Pressure, Min, Max).
2. For each curve, extract pressure vs depth data points.
3. Identify the depth unit used on the plot.
4. Provide a short interpretation of the pressure profile.

Return STRICT JSON ONLY in this schema:
{
  "chart_title": "string",
  "curves": [
    {
      "curve_name": "string",
      "points": [
        {
          "pressure_psi": number,
          "depth_value": number,
          "depth_unit": "string"
        }
      ]
    }
  ],
  "interpretation": "string"
}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 8192,
        },
    }

    response = requests.post(
        URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    result = response.json()

    raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw_text)
