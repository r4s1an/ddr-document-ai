from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
        "assumptions_or_limits": (extra_limits or []),
        "followups": [],
    }

def run_tag_chat_turn(
    engine,
    question: str,
    document_id: Optional[int] = None,
    wellbore_name: Optional[str] = None,
    day: Optional[str] = None,   
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

    try:
        from domain.tag_router import route_question   
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

    if router_decision.get("needs_clarification"):
        clarifying = router_decision.get("clarifying_question") or "Can you clarify?"
        return {
            "answer": clarifying,
            "assumptions_or_limits": router_decision.get("assumptions_or_limits", []),
            "followups": router_decision.get("followups", []),
        }

    raw_plans = router_decision.get("query_plans") or []
    if not raw_plans:
        return _error_json(
            "I couldn't map your question to any supported query yet.",
            router_decision.get("assumptions_or_limits", []),
        )

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

    try:
        from services.tag_fetch import run_many   
    except Exception as e:
        return _error_json(
            f"Backend not ready: missing services.tag_fetch ({type(e).__name__}: {e})",
            ["Create services/tag_fetch.py next."],
        )
     
    fetch_out = run_many(engine=engine, plans=plans)

    results = fetch_out.get("results", {}) or {}

    retrieved_payload = {
        "question": question,
        "context_overrides": {
            "document_id": document_id,
            "wellbore_name": wellbore_name,
            "day": day,
        },
        "results": results,   
        "assumptions_or_limits": router_decision.get("assumptions_or_limits", []),
    }

    try:
        from AI.gemini_tag_chat import generate_tag_answer   
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
    final_json.setdefault("assumptions_or_limits", router_decision.get("assumptions_or_limits", []))
    final_json.setdefault("followups", router_decision.get("followups", []))

    return final_json
