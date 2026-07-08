"""
Prediction MCP Server  –  FastMCP 3.x compatible
=================================================
Rule-based switching action recommender (no ML model).
Run: python mcp/prediction_server/server.py
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("prediction-server")


# =============================================================================
# RULE ENGINE
# =============================================================================

def _generate_candidates(
    topology: dict,
    switch_states: dict,
    analysis: dict,
    objective: str,
    priority: str,
    max_candidates: int,
) -> list[dict]:

    open_sw   = [sw for sw, closed in switch_states.items() if not closed]
    closed_sw = [sw for sw, closed in switch_states.items() if closed]

    v_violations  = analysis.get("voltage_violations",  [])
    th_violations = analysis.get("thermal_violations",   [])
    total_loss    = analysis.get("total_losses_kw",      0.0)

    candidates: list[dict] = []

    # R1 – voltage restore
    if v_violations and open_sw and closed_sw:
        low = [v["bus_id"] for v in v_violations if v.get("limit") == "min"]
        for tie, sec in itertools.islice(itertools.product(open_sw, closed_sw), 3):
            if tie != sec:
                candidates.append({
                    "rule_applied":        "R1_voltage_restore",
                    "open_switches":       [sec],
                    "close_switches":      [tie],
                    "target_violation":    f"Low voltage: {low[:2]}",
                    "expected_improvement":"Voltage recovery via tie switch",
                    "confidence":          "medium",
                    "radiality_safe":      True,
                })

    # R2 – thermal relief
    if th_violations and open_sw and closed_sw:
        overloaded = [v["branch_id"] for v in th_violations]
        for tie, sec in itertools.islice(itertools.product(open_sw, closed_sw), 2):
            if tie != sec:
                candidates.append({
                    "rule_applied":        "R2_thermal_relief",
                    "open_switches":       [sec],
                    "close_switches":      [tie],
                    "target_violation":    f"Overload: {overloaded[:2]}",
                    "expected_improvement":"Load redistribution",
                    "confidence":          "medium",
                    "radiality_safe":      True,
                })

    # R3 – loss minimisation
    if total_loss > 10.0 and open_sw and closed_sw:
        candidates.append({
            "rule_applied":        "R3_loss_minimise",
            "open_switches":       [closed_sw[-1]],
            "close_switches":      [open_sw[0]],
            "target_violation":    f"Losses: {total_loss:.1f} kW",
            "expected_improvement":"~5-15% loss reduction",
            "confidence":          "low",
            "radiality_safe":      True,
        })

    # De-duplicate
    seen: set[tuple] = set()
    unique = []
    for c in candidates:
        key = (tuple(sorted(c["open_switches"])), tuple(sorted(c["close_switches"])))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Rank
    for i, c in enumerate(unique[:max_candidates], 1):
        c["rank"] = i

    if not unique:
        unique = [{
            "rank": 1, "open_switches": [], "close_switches": [],
            "rule_applied": "NO_ACTION", "target_violation": "None",
            "expected_improvement": "No reconfiguration needed",
            "confidence": "high", "radiality_safe": True,
        }]

    return unique[:max_candidates]


# =============================================================================
# TOOL  –  plain parameters
# =============================================================================

@mcp.tool()
async def predict_switch_actions(
    topology: dict,
    switch_states: dict,
    delta_load: dict,
    analysis_results: dict,
    objective: str = "minimize_loss",
    priority: str = "safety",
    max_candidates: int = 5,
) -> list:
    """
    Generate ranked switching action candidates using rule-based heuristics.
    Rules: R1 voltage restore, R2 thermal relief, R3 loss minimisation.
    """
    candidates = _generate_candidates(
        topology, switch_states, analysis_results,
        objective, priority, max_candidates,
    )
    logger.info("[Prediction] %d candidates (objective=%s)", len(candidates), objective)
    return candidates


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)