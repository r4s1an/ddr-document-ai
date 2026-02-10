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

def extract_pressure_plot(image_bytes: bytes, mime_type: str) -> dict:
    """
    Extracts pressure-vs-time plot data using Gemini.
    MUST return a dict compatible with save_pressure_plot().
    """

    image_b64 = _encode_image_bytes(image_bytes)

    prompt = """
    Analyze this engineering pressure vs time plot.

    Rules:
    - Extract each visible data point exactly once
    - Do NOT repeat points
    - Do NOT invent values
    - If a value is unclear, omit it

    Tasks:
    1. Extract pressure vs time data
    2. Identify well names from legend
    3. Provide a short interpretation

    Return STRICT JSON ONLY in this schema:
    {
    "chart_title": "string",
    "data_points": [
        {
        "x_value": "YYYY-MM",
        "y_value": number,
        "group_name": "Well name"
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
                            "mime_type": "image/png",
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
