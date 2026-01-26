
QUERY_CATALOG = {
    "doc_overview": """
    SELECT
        d.id,
        d.source_filename,
        d.file_sha256,
        d.uploaded_at,
        d.wellbore_name,
        d.period_start,
        d.period_end,
        d.report_number AS document_report_number,

        s.status,
        s.report_creation_ts,
        s.report_number AS summary_report_number,
        s.days_ahead_behind,
        s.operator,
        s.rig_name,
        s.drilling_contractor,
        s.spud_ts,
        s.wellbore_type,
        s.elevation_rkb_msl_m,
        s.water_depth_msl_m,
        s.tight_well,
        s.hpht,
        s.temperature_degc,
        s.pressure_psig,
        s.date_well_complete,
        s.dist_drilled_m,
        s.penetration_rate_mph,
        s.hole_dia_in,
        s.pressure_test_type,
        s.formation_strength_g_cm3,
        s.dia_last_casing,
        s.depth_kickoff_mmd,
        s.depth_kickoff_mtvd,
        s.depth_mmd,
        s.depth_mtvd,
        s.plug_back_depth_mmd,
        s.depth_formation_strength_mmd,
        s.depth_formation_strength_mtvd,
        s.depth_last_casing_mmd,
        s.depth_last_casing_mtvd,
        s.extracted_at,

        a.activities_24h_text,
        a.planned_24h_text
    FROM ddr_documents d
    LEFT JOIN ddr_summary_report s
        ON s.document_id = d.id
    LEFT JOIN ddr_activity_summary a
        ON a.document_id = d.id
    WHERE d.id = :document_id
    LIMIT :limit
    """,
    "daily_metrics_by_doc": """
    SELECT
        document_id,
        wellbore_name,
        day,
        total_ops,
        total_hours,
        fail_ops,
        fail_hours,
        interruption_hours,
        repair_hours,
        avg_op_minutes
    FROM v_ddr_daily_ops_metrics
    WHERE document_id = :document_id
    ORDER BY day DESC
    LIMIT :limit
    """,
    "ops_by_doc": """
    SELECT
        id,
        document_id,
        start_time,
        end_time,
        end_depth_mmd,
        main_activity,
        sub_activity,
        state,
        remark
    FROM ddr_operations
    WHERE document_id = :document_id
    ORDER BY
        start_time NULLS LAST,
        end_time NULLS LAST,
        id
    LIMIT :limit
    """,

    "fail_ops_by_doc": """
    SELECT
        id,
        document_id,
        start_time,
        end_time,
        end_depth_mmd,
        main_activity,
        sub_activity,
        state,
        remark
    FROM ddr_operations
    WHERE document_id = :document_id
      AND lower(COALESCE(state, '')) = 'fail'
    ORDER BY
        start_time NULLS LAST,
        end_time NULLS LAST,
        id
    LIMIT :limit
    """,

    "top_remarks_by_doc": """
    SELECT
        remark,
        COUNT(*) AS n
    FROM ddr_operations
    WHERE document_id = :document_id
      AND remark IS NOT NULL
      AND NULLIF(BTRIM(remark), '') IS NOT NULL
    GROUP BY remark
    ORDER BY n DESC, remark
    LIMIT :limit
    """,
    "gas_by_doc": """
    SELECT
        id,
        document_id,
        time,
        class,
        depth_to_top_mmd,
        depth_to_bottom_md,
        depth_to_top_mtvd,
        depth_to_bottom_tvd,
        highest_gas_percent,
        lowest_gas_percent,
        c1_ppm,
        c2_ppm,
        c3_ppm,
        ic4_ppm,
        ic5_ppm
    FROM ddr_gas_reading_information
    WHERE document_id = :document_id
    ORDER BY time NULLS LAST, id
    LIMIT :limit
    """,
    "gas_spikes_by_doc": """
    SELECT
        id,
        document_id,
        time,
        class,
        depth_to_top_mmd,
        highest_gas_percent,
        c1_ppm,
        c2_ppm,
        c3_ppm,
        ic4_ppm,
        ic5_ppm
    FROM ddr_gas_reading_information
    WHERE document_id = :document_id
      AND highest_gas_percent IS NOT NULL
    ORDER BY highest_gas_percent DESC NULLS LAST, time NULLS LAST, id
    LIMIT :limit
    """,
    "drilling_fluid_by_doc": """
    SELECT
        id,
        document_id,
        sample_time,
        sample_point,
        sample_depth_mmd,
        fluid_type,
        fluid_density_g_cm3
    FROM ddr_drilling_fluid
    WHERE document_id = :document_id
    ORDER BY sample_time NULLS LAST, id
    LIMIT :limit
    """,
    "drilling_fluid_by_well_day": """
    SELECT
        f.id,
        f.document_id,
        f.sample_time,
        f.sample_point,
        f.sample_depth_mmd,
        f.fluid_type,
        f.fluid_density_g_cm3
    FROM ddr_drilling_fluid f
    JOIN ddr_documents d
    ON d.id = f.document_id
    WHERE d.wellbore_name = :wellbore_name
    AND d.period_start::date = :day
    ORDER BY f.sample_time NULLS LAST, f.id
    LIMIT :limit
    """,
    "docs_by_well_day": """
    SELECT
        id AS document_id,
        wellbore_name,
        period_start,
        period_end,
        source_filename,
        report_number
    FROM ddr_documents
    WHERE wellbore_name = :wellbore_name
      AND period_start IS NOT NULL
      AND (period_start::date) = :day::date
    ORDER BY id DESC
    LIMIT :limit
    """,
    "daily_metrics_by_well_day": """
    SELECT
        document_id,
        wellbore_name,
        day,
        total_ops,
        total_hours,
        fail_ops,
        fail_hours,
        interruption_hours,
        repair_hours,
        avg_op_minutes
    FROM v_ddr_daily_ops_metrics
    WHERE wellbore_name = :wellbore_name
      AND day = :day::date
    ORDER BY document_id DESC
    LIMIT :limit
    """,

    "ops_by_well_day": """
    SELECT
        o.id,
        o.document_id,
        d.wellbore_name,
        (d.period_start::date) AS day,
        o.start_time,
        o.end_time,
        o.end_depth_mmd,
        o.main_activity,
        o.sub_activity,
        o.state,
        o.remark
    FROM ddr_operations o
    JOIN ddr_documents d
      ON d.id = o.document_id
    WHERE d.wellbore_name = :wellbore_name
      AND d.period_start IS NOT NULL
      AND (d.period_start::date) = :day::date
    ORDER BY
        o.start_time NULLS LAST,
        o.end_time NULLS LAST,
        o.id
    LIMIT :limit
    """,
    "fail_ops_by_well_day": """
    SELECT
        o.id,
        o.document_id,
        d.wellbore_name,
        (d.period_start::date) AS day,
        o.start_time,
        o.end_time,
        o.end_depth_mmd,
        o.main_activity,
        o.sub_activity,
        o.state,
        o.remark
    FROM ddr_operations o
    JOIN ddr_documents d
      ON d.id = o.document_id
    WHERE d.wellbore_name = :wellbore_name
      AND d.period_start IS NOT NULL
      AND (d.period_start::date) = :day::date
      AND lower(COALESCE(o.state, '')) = 'fail'
    ORDER BY
        o.start_time NULLS LAST,
        o.end_time NULLS LAST,
        o.id
    LIMIT :limit
    """,

    "top_remarks_by_well_day": """
    SELECT
        o.remark,
        COUNT(*) AS n
    FROM ddr_operations o
    JOIN ddr_documents d
      ON d.id = o.document_id
    WHERE d.wellbore_name = :wellbore_name
      AND d.period_start IS NOT NULL
      AND (d.period_start::date) = :day::date
      AND o.remark IS NOT NULL
      AND NULLIF(BTRIM(o.remark), '') IS NOT NULL
    GROUP BY o.remark
    ORDER BY n DESC, o.remark
    LIMIT :limit
    """,

    "gas_by_well_day": """
    SELECT
        g.id,
        g.document_id,
        d.wellbore_name,
        (d.period_start::date) AS day,
        g.time,
        g.class,
        g.depth_to_top_mmd,
        g.depth_to_bottom_md,
        g.depth_to_top_mtvd,
        g.depth_to_bottom_tvd,
        g.highest_gas_percent,
        g.lowest_gas_percent,
        g.c1_ppm,
        g.c2_ppm,
        g.c3_ppm,
        g.ic4_ppm,
        g.ic5_ppm
    FROM ddr_gas_reading_information g
    JOIN ddr_documents d
      ON d.id = g.document_id
    WHERE d.wellbore_name = :wellbore_name
      AND d.period_start IS NOT NULL
      AND (d.period_start::date) = :day::date
    ORDER BY g.time NULLS LAST, g.id
    LIMIT :limit
    """,
    "pressure_time_plot_by_id": """
    SELECT
        id,
        chart_title,
        interpretation,
        raw_json,
        source_key
    FROM pressure_plot
    WHERE id = :plot_id
    LIMIT :limit
    """,

    "pressure_profile_plot_by_id": """
    SELECT
        id,
        chart_title,
        interpretation,
        raw_json,
        source_key
    FROM pressure_profile_plot
    WHERE id = :plot_id
    LIMIT :limit
    """,

    "pressure_time_plots_by_source_key": """
    SELECT
        id,
        chart_title,
        interpretation,
        raw_json,
        source_key
    FROM pressure_plot
    WHERE source_key = :source_key
    ORDER BY id DESC
    LIMIT :limit
    """,

    "pressure_profile_plots_by_source_key": """
    SELECT
        id,
        chart_title,
        interpretation,
        raw_json,
        source_key
    FROM pressure_profile_plot
    WHERE source_key = :source_key
    ORDER BY id DESC
    LIMIT :limit
    """,

    "pressure_time_plots_by_title": """
    SELECT
        id,
        chart_title,
        interpretation,
        raw_json,
        source_key
    FROM pressure_plot
    WHERE chart_title ILIKE :title_pattern
    ORDER BY id DESC
    LIMIT :limit
    """,

    "pressure_profile_plots_by_title": """
    SELECT
        id,
        chart_title,
        interpretation,
        raw_json,
        source_key
    FROM pressure_profile_plot
    WHERE chart_title ILIKE :title_pattern
    ORDER BY id DESC
    LIMIT :limit
    """,
    "latest_plot_any": """
    SELECT 'pressure_plot' AS plot_type, id, chart_title, interpretation, raw_json, source_key, created_at
    FROM pressure_plot
    UNION ALL
    SELECT 'pressure_profile_plot' AS plot_type, id, chart_title, interpretation, raw_json, source_key, created_at
    FROM pressure_profile_plot
    ORDER BY created_at DESC
    LIMIT :limit
    """,
    "pressure_time_series_by_plot_id": """
    SELECT id AS series_id, plot_id, well_name
    FROM pressure_series
    WHERE plot_id = :plot_id
    ORDER BY id
    LIMIT :limit
    """,

    "pressure_time_points_by_plot_id": """
    SELECT id, plot_id, series_id, x_text, x_date, pressure_psi, point_order
    FROM pressure_point
    WHERE plot_id = :plot_id
    ORDER BY series_id, point_order
    LIMIT :limit
    """,
    "pressure_profile_curves_by_plot_id": """
    SELECT id AS curve_id, plot_id, curve_name
    FROM pressure_profile_curve
    WHERE plot_id = :plot_id
    ORDER BY id
    LIMIT :limit
    """,

    "pressure_profile_points_by_plot_id": """
    SELECT id, plot_id, curve_id, pressure_psi, depth_value, depth_unit, point_order
    FROM pressure_profile_point
    WHERE plot_id = :plot_id
    ORDER BY curve_id, point_order
    LIMIT :limit
    """,

}
