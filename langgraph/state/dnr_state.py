"""
DNR Copilot – LangGraph State Schema
Python 3.12 | LangGraph
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class DNRState(TypedDict, total=False):
    # ── Inputs (set by caller before graph runs) ──────────────────────────────
    topology: dict[str, Any]          # Node/branch/switch topology map
    switch_states: dict[str, bool]    # switch_id → open/closed
    delta_load: dict[str, float]      # node_id → ΔP or ΔQ in kW / kVAR

    # ── Intermediate / produced by nodes ─────────────────────────────────────
    analysis_results: dict[str, Any]  # power-flow, voltages, losses, violations
    similar_cases: list[dict]         # historical cases from Memory MCP
    recommendations: list[dict]       # ranked switching actions from Prediction MCP

    # ── Final output ──────────────────────────────────────────────────────────
    explanation: str                  # natural-language summary for operator

    # ── Internal bookkeeping ──────────────────────────────────────────────────
    errors: list[str]                 # accumulated non-fatal errors / warnings
    plan: dict[str, Any]             # planner's structured plan