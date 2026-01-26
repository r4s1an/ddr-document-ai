import json
from sqlalchemy import text


def save_pressure_profile(
    conn,
    *,
    source_key: str,
    extracted: dict,
) -> int:
    """
    Saves a Pressure Profile (Pressure vs Depth) plot.
    Returns pressure_profile_plot.id
    """

     
    title = extracted.get("chart_title")
    if not title:
        raise ValueError("chart_title missing")

    curves = extracted.get("curves")
    if not isinstance(curves, list) or not curves:
        raise ValueError("curves missing or empty")

    interpretation = extracted.get("interpretation")
     
    conn.execute(
        text("""
            DELETE FROM pressure_profile_plot
            WHERE source_key = :source_key
        """),
        {"source_key": source_key},
    )

     
    plot_id = conn.execute(
        text("""
            INSERT INTO pressure_profile_plot
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

     
    curve_id_map = {}

    for curve in curves:
        name = curve.get("curve_name")
        if not name:
            continue

        curve_id = conn.execute(
            text("""
                INSERT INTO pressure_profile_curve
                    (plot_id, curve_name)
                VALUES
                    (:plot_id, :curve_name)
                RETURNING id
            """),
            {
                "plot_id": plot_id,
                "curve_name": name,
            },
        ).scalar_one()

        curve_id_map[name] = curve_id

     
    seen = set()
    order = 0

    for curve in curves:
        curve_name = curve.get("curve_name")
        curve_id = curve_id_map.get(curve_name)
        if not curve_id:
            continue

        points = curve.get("points", [])
        if not isinstance(points, list):
            continue

        for p in points:
            pressure = p.get("pressure_psi")
            depth = p.get("depth_value")
            unit = p.get("depth_unit")

            if not isinstance(pressure, (int, float)):
                continue
            if not isinstance(depth, (int, float)):
                continue

            key = (curve_name, float(pressure), float(depth))
            if key in seen:
                continue
            seen.add(key)

            conn.execute(
                text("""
                    INSERT INTO pressure_profile_point
                        (plot_id, curve_id, pressure_psi, depth_value, depth_unit, point_order)
                    VALUES
                        (:plot_id, :curve_id, :pressure, :depth, :unit, :order)
                """),
                {
                    "plot_id": plot_id,
                    "curve_id": curve_id,
                    "pressure": float(pressure),
                    "depth": float(depth),
                    "unit": unit,
                    "order": order,
                },
            )

            order += 1

    return plot_id
