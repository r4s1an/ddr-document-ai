from datetime import date
import json
import re
from sqlalchemy import text

_YM_RE = re.compile(r"^(\d{4})-(\d{2})$")

def _parse_x_date(x_text: str):
    m = _YM_RE.match(x_text)
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), 1)


def save_pressure_plot(
    conn,
    *,
    source_key: str,
    extracted: dict,
) -> int:

    # --- 1. Basic validation ---
    title = extracted.get("chart_title")
    if not title:
        raise ValueError("chart_title missing")

    data_points = extracted.get("data_points")
    if not isinstance(data_points, list) or not data_points:
        raise ValueError("data_points missing or empty")

    interpretation = extracted.get("interpretation")

    # --- 1.5 Re-ingest safety (DELETE FIRST) ---
    conn.execute(
        text("""
            DELETE FROM pressure_plot
            WHERE source_key = :source_key

        """),
        {"source_key": source_key},
    )

    # --- 2. Insert plot ---
    plot_id = conn.execute(
        text("""
            INSERT INTO pressure_plot
                (chart_title, interpretation, raw_json, source_key)
            VALUES
                (:title, :interpretation, :raw_json, :source_key)
            RETURNING id
        """),
        {
            "title": title,
            "interpretation": interpretation,
            "raw_json": json.dumps(extracted, ensure_ascii=False),
            "source_key": source_key,
        },
    ).scalar_one()

    # --- 3. Clean + collect wells ---
    wells = []
    seen = set()
    cleaned_points = []

    for p in data_points:
        well = p.get("group_name")
        x_text = str(p.get("x_value"))
        y = p.get("y_value")

        if not well or not isinstance(y, (int, float)):
            continue

        key = (well, x_text, float(y))
        if key in seen:
            continue

        seen.add(key)
        cleaned_points.append(p)

        if well not in wells:
            wells.append(well)

    # --- 4. Insert series ---
    series_id_map = {}

    for well in wells:
        series_id = conn.execute(
            text("""
                INSERT INTO pressure_series (plot_id, well_name)
                VALUES (:plot_id, :well_name)
                RETURNING id
            """),
            {"plot_id": plot_id, "well_name": well},
        ).scalar_one()
        series_id_map[well] = series_id

    # --- 5. Insert points ---
    order = 0
    for p in cleaned_points:
        well = p["group_name"]
        x_text = str(p["x_value"])
        y = float(p["y_value"])

        conn.execute(
            text("""
                INSERT INTO pressure_point
                    (plot_id, series_id, x_text, x_date, pressure_psi, point_order)
                VALUES
                    (:plot_id, :series_id, :x_text, :x_date, :pressure, :order)
            """),
            {
                "plot_id": plot_id,
                "series_id": series_id_map[well],
                "x_text": x_text,
                "x_date": _parse_x_date(x_text),
                "pressure": y,
                "order": order,
            },
        )

        order += 1

    return plot_id