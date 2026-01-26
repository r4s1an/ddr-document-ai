# services/tag_chat_orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# -----------------------------
# Minimal contracts (MVP)
# -----------------------------

@dataclass
class QueryPlan:
    name: str
    sql: str
    params: Dict[str, Any]
    limit: int = 200


@dataclass
class QueryUsed:
    name: str
    sql: str
    params: Dict[str, Any]
    row_count: int


def _error_json(msg: str, extra_limits: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "answer": msg,
        "sql_used": [],
        "tables_preview": {},
        "assumptions_or_limits": (extra_limits or []),
        "followups": [],
    }


def run_tag_chat_turn(
    engine,
    question: str,
    document_id: Optional[int] = None,
    wellbore_name: Optional[str] = None,
    day: Optional[str] = None,  # ISO date string "YYYY-MM-DD"
    model_name: str = "gemini-flash-latest",
) -> Dict[str, Any]:
    """
    Orchestrates one TAG chat turn:
      router -> SQL fetch -> build retrieved payload -> Gemini -> strict JSON response

    Notes:
    - This module is UI-agnostic (no streamlit here).
    - Lazy imports are used so app.py can run while we build other modules.
    """

    question = (question or "").strip()
    if not question:
        return _error_json("Empty question.")

    # -----------------------------
    # 1) Route question -> QueryPlans OR clarification
    # -----------------------------
    try:
        from domain.tag_router import route_question  # we will create next
    except Exception as e:
        return _error_json(
            f"Backend not ready: missing domain.tag_router ({type(e).__name__}: {e})",
            ["Create domain/tag_router.py next."],
        )

    router_decision = route_question(
        question=question,
        document_id=document_id,
        wellbore_name=wellbore_name,
        day=day,
    )

    # Expected router_decision keys:
    # {
    #   "needs_clarification": bool,
    #   "clarifying_question": str | None,
    #   "query_plans": list[QueryPlan-like dict],
    #   "assumptions_or_limits": list[str]
    # }

    if router_decision.get("needs_clarification"):
        clarifying = router_decision.get("clarifying_question") or "Can you clarify?"
        return {
            "answer": clarifying,
            "sql_used": [],
            "tables_preview": {},
            "assumptions_or_limits": router_decision.get("assumptions_or_limits", []),
            "followups": router_decision.get("followups", []),
        }

    raw_plans = router_decision.get("query_plans") or []
    if not raw_plans:
        return _error_json(
            "I couldn't map your question to any supported query yet.",
            router_decision.get("assumptions_or_limits", []),
        )

    # Normalize plans into QueryPlan objects
    plans: List[QueryPlan] = []
    for p in raw_plans:
        plans.append(
            QueryPlan(
                name=p["name"],
                sql=p["sql"],
                params=p.get("params", {}) or {},
                limit=int(p.get("limit", 200)),
            )
        )

    # -----------------------------
    # 2) Execute SQL safely
    # -----------------------------
    try:
        from services.tag_fetch import run_many  # we will create next
    except Exception as e:
        return _error_json(
            f"Backend not ready: missing services.tag_fetch ({type(e).__name__}: {e})",
            ["Create services/tag_fetch.py next."],
        )

    # run_many should return:
    # {
    #   "results": { plan_name: {"rows": [...], "row_count": int, "truncated": bool} },
    #   "sql_used": [ {name, sql, params, row_count} ... ],
    # }
    fetch_out = run_many(engine=engine, plans=plans)

    results = fetch_out.get("results", {}) or {}
    sql_used = fetch_out.get("sql_used", []) or []

    # Build a compact preview for UI + LLM grounding
    # (We keep at most 25 rows per query in preview to stay small.)
    tables_preview: Dict[str, Any] = {}
    for qname, qres in results.items():
        rows = qres.get("rows", []) or []
        tables_preview[qname] = rows[:25]

    retrieved_payload = {
        "question": question,
        "context_overrides": {
            "document_id": document_id,
            "wellbore_name": wellbore_name,
            "day": day,
        },
        "results": results,  # full (already limited by SQL)
        "tables_preview": tables_preview,
        "assumptions_or_limits": router_decision.get("assumptions_or_limits", []),
    }

    # -----------------------------
    # 3) Ask Gemini to produce STRICT JSON answer grounded in payload
    # -----------------------------
    try:
        from AI.gemini_tag_chat import generate_tag_answer  # we will create after fetch/router
    except Exception as e:
        return _error_json(
            f"Backend not ready: missing ai.gemini_tag_chat ({type(e).__name__}: {e})",
            ["Create ai/gemini_tag_chat.py next (after router + fetch)."],
        )

    final_json = generate_tag_answer(
        question=question,
        retrieved_payload=retrieved_payload,
        model_name=model_name,
    )

    # Ensure required keys exist (defensive)
    final_json.setdefault("sql_used", sql_used)
    final_json.setdefault("tables_preview", tables_preview)
    final_json.setdefault("assumptions_or_limits", router_decision.get("assumptions_or_limits", []))
    final_json.setdefault("followups", router_decision.get("followups", []))

    return final_json
