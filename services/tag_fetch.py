import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from sqlalchemy import text
from sqlalchemy.engine import Engine


DEFAULT_LIMIT = 200
MAX_LIMIT = 1000
 
STATEMENT_TIMEOUT_MS = 4000   
LOCK_TIMEOUT_MS = 1500        
 

_DENY_TOKENS = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "grant", "revoke", "commit", "rollback",
    "vacuum", "analyze", "reindex", "cluster",
    "copy", "execute", "prepare", "deallocate",
    "call", "do", "set ", "reset ",
]
_DENY_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in _DENY_TOKENS) + r")\b", re.IGNORECASE)

 
_SELECT_WITH_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


@dataclass
class Plan:
    name: str
    sql: str
    params: Dict[str, Any]
    limit: int = DEFAULT_LIMIT


def _normalize_limit(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    try:
        n = int(limit)
    except Exception:
        return DEFAULT_LIMIT
    if n < 1:
        return DEFAULT_LIMIT
    if n > MAX_LIMIT:
        return MAX_LIMIT
    return n


def _validate_sql_is_read_only(sql: str) -> None:
    s = (sql or "").strip()
    if not s:
        raise ValueError("Empty SQL.")
    if not _SELECT_WITH_RE.match(s):
        raise ValueError("Only SELECT/WITH queries are allowed.")
     
    if _DENY_RE.search(s):
        raise ValueError("Query contains forbidden (non read-only) tokens.")
     
     
    if ";" in s.strip().rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed.")


def _jsonable(v: Any) -> Any:
    """
    Convert common non-JSON types into JSON-serializable values.
    - datetime/date/time -> ISO strings
    - Decimal -> float
    """
     
    try:
        import datetime as _dt
        from decimal import Decimal
    except Exception:
        _dt = None
        Decimal = None   

    if v is None:
        return None

    if Decimal is not None and isinstance(v, Decimal):
         
        return float(v)

    if _dt is not None:
        if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
            return v.isoformat()

     
    if isinstance(v, (bytes, bytearray)):
        return v.hex()

    return v


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row._mapping)
    return {k: _jsonable(v) for k, v in d.items()}


def run_one(engine: Engine, plan: Plan) -> Dict[str, Any]:
    """
    Execute one safe read-only query plan and return:
      { "rows": [...], "row_count": int, "truncated": bool }
    """
    _validate_sql_is_read_only(plan.sql)

    limit = _normalize_limit(plan.limit)

    params = dict(plan.params or {})
    params["limit"] = limit   

    with engine.connect() as conn:
         
        conn.execute(text("SET LOCAL statement_timeout = :ms"), {"ms": STATEMENT_TIMEOUT_MS})
        conn.execute(text("SET LOCAL lock_timeout = :ms"), {"ms": LOCK_TIMEOUT_MS})

        res = conn.execute(text(plan.sql), params)
        rows = res.fetchall()

    row_dicts = [_row_to_dict(r) for r in rows]
    row_count = len(row_dicts)

     
    truncated = row_count >= limit

    return {
        "rows": row_dicts,
        "row_count": row_count,
        "truncated": truncated,
    }

def run_many(engine: Engine, plans: Sequence[Any]) -> Dict[str, Any]:
    """
    Execute many plans and return:
    {
      "results": {
        plan_name: {"rows": [...], "row_count": int, "truncated": bool}
      },
      "sql_used": [
        {"name": ..., "sql": ..., "params": {...}, "row_count": N}
      ]
    }

    `plans` may be a list of Plan or objects with Plan-like attributes.
    """
    results: Dict[str, Any] = {}
    sql_used: List[Dict[str, Any]] = []

    for p in plans:
        if isinstance(p, Plan):
            plan = p
        else:
             
            plan = Plan(
                name=getattr(p, "name"),
                sql=getattr(p, "sql"),
                params=getattr(p, "params", {}) or {},
                limit=getattr(p, "limit", DEFAULT_LIMIT),
            )

        out = run_one(engine, plan)

        results[plan.name] = out
        sql_used.append(
            {
                "name": plan.name,
                "sql": plan.sql.strip(),
                "params": dict(plan.params or {}) | {"limit": _normalize_limit(plan.limit)},
                "row_count": out["row_count"],
            }
        )

    return {"results": results, "sql_used": sql_used}