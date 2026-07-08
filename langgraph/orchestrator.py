"""
DNR Copilot – LangGraph Orchestrator
=====================================
Import path: from langgraph.orchestrator import run_dnr_copilot, build_graph
(works when dnr-copilot/ is on sys.path, which api/main.py ensures)
"""

from __future__ import annotations

import sys
import os
import json
import logging
from typing import Any

# Ensure dnr-copilot/ root is on sys.path when this module is imported directly
_here = os.path.dirname(os.path.abspath(__file__))   # dnr-copilot/langgraph/
_root = os.path.dirname(_here)                         # dnr-copilot/
if _root not in sys.path:
    sys.path.insert(0, _root)

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from langgraph.state.dnr_state import DNRState
from langgraph.prompts.agent_prompts import (
    PLANNER_SYSTEM, PLANNER_USER,
    ANALYSIS_SYSTEM, ANALYSIS_USER,
    RECOMMENDATION_SYSTEM, RECOMMENDATION_USER,
)
from clients.mcp_clients import AnalysisClient, MemoryClient, PredictionClient

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)         
analysis_client   = AnalysisClient()
memory_client     = MemoryClient()
prediction_client = PredictionClient()


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_llm_json(text: str) -> dict:
    """Strip markdown fences then parse JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]          # drop opening fence line
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


# =============================================================================
# NODE 1 – PLANNER
# =============================================================================
async def planner_node(state: DNRState) -> DNRState:
    logger.info("[Planner] Starting")
    user_msg = PLANNER_USER.format(
        topology=json.dumps(state["topology"], indent=2),
        switch_states=json.dumps(state["switch_states"], indent=2),
        delta_load=json.dumps(state["delta_load"], indent=2),
    )
    try:
        resp = await llm.ainvoke([
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": user_msg},
        ])
        plan = _parse_llm_json(resp.content)
        logger.info("[Planner] feasible=%s", plan.get("feasible"))
        return {**state, "plan": plan, "errors": state.get("errors", [])}
    except json.JSONDecodeError as exc:
        err = f"Planner JSON parse error – {exc}"
        logger.error(err)
        return {**state,
                "plan": {"feasible": False, "reason": "LLM output was not valid JSON"},
                "errors": state.get("errors", []) + [err]}
    except Exception as exc:
        err = f"Planner error – {exc}"
        logger.exception(err)
        return {**state,
                "plan": {"feasible": False, "reason": str(exc)},
                "errors": state.get("errors", []) + [err]}


# =============================================================================
# NODE 2 – ANALYSIS
# =============================================================================
async def analysis_node(state: DNRState) -> DNRState:
    logger.info("[Analysis] Running")
    plan     = state.get("plan", {})
    required = plan.get("required_analyses",
                        ["power_flow", "voltage_profile", "loss_calculation",
                         "constraint_check", "radiality_check"])
    raw: dict[str, Any] = {}
    errors = list(state.get("errors", []))

    try:
        if "power_flow"      in required:
            raw["power_flow"]      = await analysis_client.run_powerflow(
                state["topology"], state["switch_states"], state["delta_load"])
        if "voltage_profile" in required:
            raw["voltage_profile"] = await analysis_client.voltage_profile(
                state["topology"], state["switch_states"])
        if "loss_calculation" in required:
            raw["loss_calculation"]= await analysis_client.loss_calculation(
                state["topology"], state["switch_states"], state["delta_load"])
        if "constraint_check" in required:
            raw["constraint_check"]= await analysis_client.constraint_check(
                state["topology"], state["switch_states"])
        if "radiality_check" in required:
            raw["radiality_check"] = await analysis_client.radiality_check(
                state["topology"], state["switch_states"])
    except Exception as exc:
        err = f"Analysis MCP error – {exc}"
        logger.exception(err)
        return {**state, "analysis_results": raw, "errors": errors + [err]}

    try:
        resp = await llm.ainvoke([
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {"role": "user",   "content": ANALYSIS_USER.format(
                raw_analysis=json.dumps(raw, indent=2),
                plan=json.dumps(plan, indent=2))},
        ])
        analysis_results = _parse_llm_json(resp.content)
        logger.info("[Analysis] %s", analysis_results.get("summary"))
    except Exception as exc:
        err = f"Analysis LLM error – {exc}"
        logger.error(err)
        errors.append(err)
        analysis_results = raw

    return {**state, "analysis_results": analysis_results, "errors": errors}


# =============================================================================
# NODE 3 – MEMORY
# =============================================================================
async def memory_node(state: DNRState) -> DNRState:
    logger.info("[Memory] Retrieving cases")
    plan   = state.get("plan", {})
    errors = list(state.get("errors", []))

    if not plan.get("retrieve_memory", True):
        return {**state, "similar_cases": []}

    try:
        similar_cases = await memory_client.retrieve_similar_cases(
            topology=state["topology"],
            delta_load=state["delta_load"],
            query_hint=plan.get("memory_query_hint", ""),
            top_k=5,
        )
        logger.info("[Memory] %d cases", len(similar_cases))
    except Exception as exc:
        err = f"Memory MCP error – {exc}"
        logger.exception(err)
        errors.append(err)
        similar_cases = []

    return {**state, "similar_cases": similar_cases, "errors": errors}


# =============================================================================
# NODE 4 – PREDICTION
# =============================================================================
async def prediction_node(state: DNRState) -> DNRState:
    logger.info("[Prediction] Generating actions")
    plan   = state.get("plan", {})
    errors = list(state.get("errors", []))

    try:
        recommendations = await prediction_client.predict_switch_actions(
            topology=state["topology"],
            switch_states=state["switch_states"],
            delta_load=state["delta_load"],
            analysis_results=state.get("analysis_results", {}),
            objective=plan.get("prediction_params", {}).get("objective", "minimize_loss"),
            priority=plan.get("prediction_params", {}).get("priority", "safety"),
        )
        logger.info("[Prediction] %d candidates", len(recommendations))
    except Exception as exc:
        err = f"Prediction MCP error – {exc}"
        logger.exception(err)
        errors.append(err)
        recommendations = []

    return {**state, "recommendations": recommendations, "errors": errors}


# =============================================================================
# NODE 5 – RECOMMENDATION
# =============================================================================
async def recommendation_node(state: DNRState) -> DNRState:
    logger.info("[Recommendation] Generating explanation")
    errors = list(state.get("errors", []))

    try:
        resp = await llm.ainvoke([
            {"role": "system", "content": RECOMMENDATION_SYSTEM},
            {"role": "user",   "content": RECOMMENDATION_USER.format(
                analysis_results=json.dumps(state.get("analysis_results", {}), indent=2),
                similar_cases=json.dumps(state.get("similar_cases", []), indent=2),
                predictions=json.dumps(state.get("recommendations", []), indent=2),
                plan=json.dumps(state.get("plan", {}), indent=2))},
        ])
        output          = _parse_llm_json(resp.content)
        recommendations = output.get("recommended_actions", state.get("recommendations", []))
        explanation     = output.get("explanation", "No explanation produced.")
        logger.info("[Recommendation] %s", explanation[:80])
    except Exception as exc:
        err = f"Recommendation error – {exc}"
        logger.exception(err)
        errors.append(err)
        recommendations = state.get("recommendations", [])
        explanation     = "Unable to generate explanation due to an internal error."

    return {**state, "recommendations": recommendations,
            "explanation": explanation, "errors": errors}


# =============================================================================
# ROUTING
# =============================================================================
def route_after_planner(state: DNRState) -> str:
    if not state.get("plan", {}).get("feasible", True):
        return "end_infeasible"
    return "analysis"


async def infeasible_node(state: DNRState) -> DNRState:
    reason = state.get("plan", {}).get("reason", "Unknown reason")
    return {**state,
            "explanation": f"Reconfiguration is not feasible: {reason}",
            "recommendations": []}


# =============================================================================
# GRAPH
# =============================================================================
def build_graph():
    builder = StateGraph(DNRState)
    builder.add_node("planner",        planner_node)
    builder.add_node("analysis",       analysis_node)
    builder.add_node("memory",         memory_node)
    builder.add_node("prediction",     prediction_node)
    builder.add_node("recommendation", recommendation_node)
    builder.add_node("infeasible",     infeasible_node)

    builder.set_entry_point("planner")
    builder.add_conditional_edges("planner", route_after_planner,
                                  {"analysis": "analysis", "end_infeasible": "infeasible"})
    builder.add_edge("analysis",       "memory")
    builder.add_edge("memory",         "prediction")
    builder.add_edge("prediction",     "recommendation")
    builder.add_edge("recommendation", END)
    builder.add_edge("infeasible",     END)
    return builder.compile()


async def run_dnr_copilot(topology: dict, switch_states: dict, delta_load: dict) -> DNRState:
    return await build_graph().ainvoke({
        "topology":      topology,
        "switch_states": switch_states,
        "delta_load":    delta_load,
        "errors":        [],
    })