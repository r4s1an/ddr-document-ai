import pandas as pd
from sqlalchemy import text

def fetch_daily_ops_metrics(engine, wellbore_name: str | None = None):
    sql = """
    SELECT *
    FROM v_ddr_daily_ops_metrics
    WHERE (:wb IS NULL OR wellbore_name = :wb)
    ORDER BY day, document_id;
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"wb": wellbore_name})
    return df
