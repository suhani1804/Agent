"""
MCP Client Wrappers 
================================================
Uses fastmcp.Client (MCP protocol) instead of raw HTTP.

  AnalysisClient   → Analysis MCP  (port 8001)
  MemoryClient     → Memory MCP    (port 8002)
  PredictionClient → Prediction MCP (port 8003)
"""


from __future__ import annotations

import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

logger = logging.getLogger(__name__)

ANALYSIS_URL   = "http://localhost:8001/mcp"
MEMORY_URL     = "http://localhost:8002/mcp"
PREDICTION_URL = "http://localhost:8003/mcp"


async def _call(url: str, tool: str, args: dict) -> Any:
    """Call a single tool on a FastMCP server and return the parsed result."""
    transport = StreamableHttpTransport(url)
    async with Client(transport) as client:
        result = await client.call_tool(tool, args)
    # result is a CallToolResult; content is a list of TextContent / etc.
    # For our servers every tool returns JSON-serialisable dicts/lists.
    if not result.content:
        return {}
    first = result.content[0]
    # TextContent has .text; parse it if it looks like JSON, else return raw
    text = getattr(first, "text", str(first))
    try:
        import json
        return json.loads(text)
    except (ValueError, TypeError):
        return text


# =============================================================================
# ANALYSIS CLIENT
# =============================================================================

class AnalysisClient:
    def __init__(self, url: str = ANALYSIS_URL):
        self._url = url

    async def run_powerflow(self, topology, switch_states, delta_load=None) -> dict:
        return await _call(self._url, "run_powerflow", {
            "topology":       topology,
            "switch_states":  switch_states,
            "delta_load":     delta_load or {},
        })

    async def voltage_profile(self, topology, switch_states,
                               v_min_pu=0.95, v_max_pu=1.05) -> dict:
        return await _call(self._url, "voltage_profile", {
            "topology":      topology,
            "switch_states": switch_states,
            "v_min_pu":      v_min_pu,
            "v_max_pu":      v_max_pu,
        })

    async def loss_calculation(self, topology, switch_states, delta_load=None) -> dict:
        return await _call(self._url, "loss_calculation", {
            "topology":      topology,
            "switch_states": switch_states,
            "delta_load":    delta_load or {},
        })

    async def constraint_check(self, topology, switch_states,
                                v_min_pu=0.95, v_max_pu=1.05,
                                thermal_limit_pct=100.0) -> dict:
        return await _call(self._url, "constraint_check", {
            "topology":           topology,
            "switch_states":      switch_states,
            "v_min_pu":           v_min_pu,
            "v_max_pu":           v_max_pu,
            "thermal_limit_pct":  thermal_limit_pct,
        })

    async def radiality_check(self, topology, switch_states) -> dict:
        return await _call(self._url, "radiality_check", {
            "topology":      topology,
            "switch_states": switch_states,
        })


# =============================================================================
# MEMORY CLIENT
# =============================================================================

class MemoryClient:
    def __init__(self, url: str = MEMORY_URL):
        self._url = url

    async def retrieve_similar_cases(self, topology, delta_load,
                                      query_hint="", top_k=5) -> list:
        result = await _call(self._url, "retrieve_similar_cases", {
            "topology":   topology,
            "delta_load": delta_load,
            "query_hint": query_hint,
            "top_k":      top_k,
        })
        return result if isinstance(result, list) else []

    async def save_case(self, topology_id, switch_states, delta_load,
                         analysis_results, recommendations, explanation,
                         tags=None) -> dict:
        return await _call(self._url, "save_case", {
            "topology_id":      topology_id,
            "switch_states":    switch_states,
            "delta_load":       delta_load,
            "analysis_results": analysis_results,
            "recommendations":  recommendations,
            "explanation":      explanation,
            "tags":             tags or [],
        })

    async def topology_lookup(self, topology_id: str) -> dict:
        return await _call(self._url, "topology_lookup", {
            "topology_id": topology_id,
        })


# =============================================================================
# PREDICTION CLIENT
# =============================================================================

class PredictionClient:
    def __init__(self, url: str = PREDICTION_URL):
        self._url = url

    async def predict_switch_actions(self, topology, switch_states, delta_load,
                                      analysis_results, objective="minimize_loss",
                                      priority="safety", max_candidates=5) -> list:
        result = await _call(self._url, "predict_switch_actions", {
            "topology":         topology,
            "switch_states":    switch_states,
            "delta_load":       delta_load,
            "analysis_results": analysis_results,
            "objective":        objective,
            "priority":         priority,
            "max_candidates":   max_candidates,
        })
        return result if isinstance(result, list) else []