# domain/tag_router.py

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from domain.query_catalog import QUERY_CATALOG


# -----------------------------
# Helpers
# -----------------------------

_DOC_ID_RE = re.compile(r"\b(?:document[_\s-]*id|doc(?:ument)?)[\s:=#]*([0-9]{1,12})\b", re.IGNORECASE)
_ANY_INT_RE = re.compile(r"\b([0-9]{1,12})\b")
# Wellbore examples: 15/9-F-11 T2, 15/9-F-14, etc.
_WELLBORE_RE = re.compile(r"\b(\d{1,3}/\d{1,3}-[A-Z]-\d{1,3}(?:\s*[A-Z0-9]{1,6})?)\b", re.IGNORECASE)
# ISO date: 2013-04-23
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_SHA256_RE = re.compile(r"\b([a-f0-9]{64})\b", re.IGNORECASE)
_PLOT_ID_RE = re.compile(r"\b(?:plot[_\s-]*id|plot)[\s:=#]*([0-9]{1,12})\b", re.IGNORECASE)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _lower(s: str) -> str:
    return (s or "").strip().lower()


def _parse_iso_date(s: str) -> Optional[str]:
    """Return ISO date string YYYY-MM-DD if valid."""
    s = (s or "").strip()
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        return d.isoformat()
    except Exception:
        return None

def _extract_source_key(question: str) -> Optional[str]:
    m = _SHA256_RE.search(question or "")
    return m.group(1).lower() if m else None

def _extract_plot_id(question: str) -> Optional[int]:
    m = _PLOT_ID_RE.search(question or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _extract_document_id(question: str) -> Optional[int]:
    m = _DOC_ID_RE.search(question or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _extract_iso_date(question: str) -> Optional[str]:
    m = _ISO_DATE_RE.search(question or "")
    if m:
        return _parse_iso_date(m.group(1))
    return None


def _extract_wellbore(question: str) -> Optional[str]:
    m = _WELLBORE_RE.search(question or "")
    if m:
        return _norm_ws(m.group(1))
    return None


def _intent_keywords(q: str) -> Dict[str, bool]:
    ql = _lower(q)
    return {
        "compare": any(k in ql for k in ["compare", "vs", "versus", "difference", "between"]),
        "failed": any(k in ql for k in ["fail", "failed", "failure", "error"]),
        "downtime": any(k in ql for k in ["downtime", "npt", "interruption", "repair", "stuck", "breakdown"]),
        "operations": any(k in ql for k in ["operation", "ops", "timeline", "what happened", "activities"]),
        "gas": any(k in ql for k in ["gas", "c1", "c2", "c3", "ic4", "ic5", "ppm", "%"]),
        "top_reasons": any(k in ql for k in ["top", "recurring", "common", "most frequent", "reasons", "remarks"]),
        "summary": any(k in ql for k in ["summary", "overview", "report", "status"]),
        "image": any(k in ql for k in ["plot", "chart", "curve", "profile", "pressure plot", "pressure profile"]),

    }


def _add_followups_for_missing_entities(
    have_doc: bool,
    have_well: bool,
    have_day: bool,
) -> List[str]:
    f: List[str] = []
    if not have_doc and not have_well:
        f.append("Which document_id or wellbore_name should I use?")
    if have_well and not have_day:
        f.append("Do you mean a specific day (YYYY-MM-DD) or all days for that wellbore?")
    if have_doc and not have_day:
        f.append("Do you mean this document only (its day), or filter operations by a specific day?")
    return f


# -----------------------------
# Router entry point
# -----------------------------

def route_question(
    question: str,
    document_id: Optional[int] = None,
    wellbore_name: Optional[str] = None,
    day: Optional[str] = None,  # ISO date string "YYYY-MM-DD"
) -> Dict[str, Any]:
    """
    Deterministic router:
    - Extract entities from question (doc_id, wellbore, day)
    - Choose an intent via keyword rules
    - Return query plans using allowlisted query templates from QUERY_CATALOG

    Returns dict:
    {
      "needs_clarification": bool,
      "clarifying_question": str|None,
      "query_plans": [ {name, sql, params, limit} ... ],
      "assumptions_or_limits": [str...],
      "followups": [str...]
    }
    """
    q = _norm_ws(question)

    # entity extraction (question overrides > UI overrides, but UI overrides can fill missing)
    q_doc = _extract_document_id(q)
    q_well = _extract_wellbore(q)
    q_day = _extract_iso_date(q)

    doc_id = q_doc if q_doc is not None else document_id
    well = q_well if q_well is not None else (wellbore_name.strip() if wellbore_name else None)
    day_iso = q_day if q_day is not None else _parse_iso_date(day) if day else None

    kw = _intent_keywords(q)

    assumptions: List[str] = [
        "SQL retrieval is read-only and limited (row limits applied).",
        "Failure is defined as ddr_operations.state = 'fail' (case-insensitive) for MVP.",
        "Day filtering uses ddr_documents.period_start::date = day.",
    ]

    followups = _add_followups_for_missing_entities(
        have_doc=doc_id is not None,
        have_well=bool(well),
        have_day=day_iso is not None,
    )

    # If nothing to anchor retrieval, ask for doc_id or wellbore
    if doc_id is None and not well:
        return {
            "needs_clarification": True,
            "clarifying_question": "Which document_id or wellbore_name should I use?",
            "query_plans": [],
            "assumptions_or_limits": assumptions,
            "followups": followups,
        }

    # Intent selection (MVP)
    # Priority: compare > image > downtime > failed > gas > top_reasons > operations/summary

    intent = "DOC_OVERVIEW"
    if kw["compare"]:
        intent = "COMPARE"
    elif kw["image"]:
        intent = "IMAGE"
    elif kw["downtime"]:
        intent = "DOWNTIME"
    elif kw["failed"]:
        intent = "FAILED_OPS"
    elif kw["gas"]:
        intent = "GAS"
    elif kw["top_reasons"]:
        intent = "TOP_REMARKS"
    elif kw["operations"]:
        intent = "OPS_TIMELINE"
    elif kw["summary"]:
        intent = "DOC_OVERVIEW"

    # Build query plans from catalog
    plans: List[Dict[str, Any]] = []
    if intent == "IMAGE":
        plot_id = _extract_plot_id(q)
        source_key = _extract_source_key(q)

        ql = _lower(q)
        is_profile = any(k in ql for k in ["profile", "pressure profile", "vs depth", "depth"])
        is_time = any(k in ql for k in ["time", "vs time", "pressure vs time"])

        if plot_id is not None:
            if is_profile and not is_time:
                plans.append(_make_plan("pressure_profile_plot_by_id", {"plot_id": plot_id}, limit=5))
            elif is_time and not is_profile:
                plans.append(_make_plan("pressure_time_plot_by_id", {"plot_id": plot_id}, limit=5))
            else:
                # unknown which type; try both
                plans.append(_make_plan("pressure_time_plot_by_id", {"plot_id": plot_id}, limit=5))
                plans.append(_make_plan("pressure_profile_plot_by_id", {"plot_id": plot_id}, limit=5))

            return {
                "needs_clarification": False,
                "clarifying_question": None,
                "query_plans": plans,
                "assumptions_or_limits": assumptions + ["Image intent: summarizing stored interpretation + raw_json from SQL."],
                "followups": ["Do you want a comparison between plots or a single plot explanation?"],
            }

        if source_key:
            if is_profile and not is_time:
                plans.append(_make_plan("pressure_profile_plots_by_source_key", {"source_key": source_key}, limit=20))
            elif is_time and not is_profile:
                plans.append(_make_plan("pressure_time_plots_by_source_key", {"source_key": source_key}, limit=20))
            else:
                plans.append(_make_plan("pressure_time_plots_by_source_key", {"source_key": source_key}, limit=20))
                plans.append(_make_plan("pressure_profile_plots_by_source_key", {"source_key": source_key}, limit=20))

            return {
                "needs_clarification": False,
                "clarifying_question": None,
                "query_plans": plans,
                "assumptions_or_limits": assumptions + ["Image intent: summarizing stored interpretation + raw_json from SQL."],
                "followups": ["If multiple plots are returned, tell me which plot_id to focus on."],
            }

        # Fallback: title search using ILIKE
        # Use a conservative pattern (first ~40 chars) to avoid crazy wide matches
        title_snip = q[:40]
        title_pattern = f"%{title_snip}%"

        if is_profile and not is_time:
            plans.append(_make_plan("pressure_profile_plots_by_title", {"title_pattern": title_pattern}, limit=20))
        elif is_time and not is_profile:
            plans.append(_make_plan("pressure_time_plots_by_title", {"title_pattern": title_pattern}, limit=20))
        else:
            plans.append(_make_plan("pressure_time_plots_by_title", {"title_pattern": title_pattern}, limit=20))
            plans.append(_make_plan("pressure_profile_plots_by_title", {"title_pattern": title_pattern}, limit=20))

        return {
            "needs_clarification": False,
            "clarifying_question": None,
            "query_plans": plans,
            "assumptions_or_limits": assumptions + [
                "Image intent fallback used chart_title ILIKE matching; results may include multiple plots.",
                "Provide plot_id or source_key for more precise retrieval."
            ],
            "followups": ["If you know it, provide plot_id or source_key for exact match."],
        }
    # --- Compare: needs two document_ids. MVP approach:
    # If question contains "12 and 13" or "12 vs 13" etc., extract two ints.
    if intent == "COMPARE":
        ints = [int(x) for x in _ANY_INT_RE.findall(q)]
        # prefer doc_id if present as first
        doc_ids = []
        if doc_id is not None:
            doc_ids.append(doc_id)
        for x in ints:
            if x not in doc_ids:
                doc_ids.append(x)
        doc_ids = doc_ids[:2]

        if len(doc_ids) < 2:
            return {
                "needs_clarification": True,
                "clarifying_question": "Which two document_ids should I compare? Example: 'compare document_id 12 vs 13'.",
                "query_plans": [],
                "assumptions_or_limits": assumptions,
                "followups": ["Provide two document_ids to compare."],
            }

        a, b = doc_ids[0], doc_ids[1]
        plans.append(_make_plan("doc_overview", {"document_id": a}, limit=1))
        plans.append(_make_plan("doc_overview", {"document_id": b}, limit=1))
        plans.append(_make_plan("daily_metrics_by_doc", {"document_id": a}, limit=10))
        plans.append(_make_plan("daily_metrics_by_doc", {"document_id": b}, limit=10))
        plans.append(_make_plan("fail_ops_by_doc", {"document_id": a}, limit=200))
        plans.append(_make_plan("fail_ops_by_doc", {"document_id": b}, limit=200))

        return {
            "needs_clarification": False,
            "clarifying_question": None,
            "query_plans": plans,
            "assumptions_or_limits": assumptions + ["Compare intent: using v_ddr_daily_ops_metrics + fail ops lists."],
            "followups": [
                "Do you want the comparison by totals, by day, or both?",
                "Do you want top failure remarks for each document?",
            ],
        }

    # --- Resolve day/doc/well for non-compare intents ---
    # For well+day queries, we need document ids for that well/day.
    # We'll use a lookup query, then other queries can use that result later (we’ll do that in fetch layer later).
    # MVP: we directly query by (well, day) in templates where possible.

    if intent in ("DOC_OVERVIEW", "OPS_TIMELINE", "FAILED_OPS", "DOWNTIME", "TOP_REMARKS", "GAS"):
        # If doc_id is present, we anchor everything to doc_id
        if doc_id is not None:
            plans.append(_make_plan("doc_overview", {"document_id": doc_id}, limit=1))

            if intent in ("OPS_TIMELINE", "DOWNTIME"):
                plans.append(_make_plan("ops_by_doc", {"document_id": doc_id}, limit=300))
                plans.append(_make_plan("daily_metrics_by_doc", {"document_id": doc_id}, limit=10))

            if intent == "FAILED_OPS":
                plans.append(_make_plan("fail_ops_by_doc", {"document_id": doc_id}, limit=300))

            if intent == "TOP_REMARKS":
                plans.append(_make_plan("top_remarks_by_doc", {"document_id": doc_id}, limit=30))

            if intent == "GAS":
                plans.append(_make_plan("gas_by_doc", {"document_id": doc_id}, limit=300))

            return {
                "needs_clarification": False,
                "clarifying_question": None,
                "query_plans": plans,
                "assumptions_or_limits": assumptions,
                "followups": followups,
            }

        # If no doc_id but we have well/day, use well/day templates
        if well and day_iso:
            plans.append(_make_plan("docs_by_well_day", {"wellbore_name": well, "day": day_iso}, limit=50))

            if intent in ("OPS_TIMELINE", "DOWNTIME"):
                plans.append(_make_plan("daily_metrics_by_well_day", {"wellbore_name": well, "day": day_iso}, limit=50))
                plans.append(_make_plan("ops_by_well_day", {"wellbore_name": well, "day": day_iso}, limit=400))

            if intent == "FAILED_OPS":
                plans.append(_make_plan("fail_ops_by_well_day", {"wellbore_name": well, "day": day_iso}, limit=400))

            if intent == "TOP_REMARKS":
                plans.append(_make_plan("top_remarks_by_well_day", {"wellbore_name": well, "day": day_iso}, limit=50))

            if intent == "GAS":
                plans.append(_make_plan("gas_by_well_day", {"wellbore_name": well, "day": day_iso}, limit=400))

            return {
                "needs_clarification": False,
                "clarifying_question": None,
                "query_plans": plans,
                "assumptions_or_limits": assumptions + ["Well+day intent: may match multiple documents on same day; results will show all matches."],
                "followups": [
                    "If multiple documents match, tell me which document_id you want to focus on.",
                ],
            }

        # If only well without day: ask whether all days or which day
        if well and not day_iso:
            return {
                "needs_clarification": True,
                "clarifying_question": f"I have wellbore_name='{well}'. Which day (YYYY-MM-DD) should I use, or should I summarize all days?",
                "query_plans": [],
                "assumptions_or_limits": assumptions,
                "followups": [
                    "Example: '2013-04-23'",
                    "Or say: 'all days' (we can implement this next).",
                ],
            }

    # Default fallback
    return {
        "needs_clarification": True,
        "clarifying_question": "I couldn't fully understand the request. Can you specify document_id, wellbore_name, or day?",
        "query_plans": [],
        "assumptions_or_limits": assumptions,
        "followups": followups,
    }


def _make_plan(name: str, params: Dict[str, Any], limit: int) -> Dict[str, Any]:
    if name not in QUERY_CATALOG:
        raise KeyError(f"Unknown query template: {name}")
    return {
        "name": name,
        "sql": QUERY_CATALOG[name],
        "params": params,
        "limit": int(limit),
    }