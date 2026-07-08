"""
Memory MCP Server  –  FastMCP 3.x compatible
=============================================
Run: python mcp/memory_server/server.py
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

STORAGE_DIR   = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CASES_FILE    = STORAGE_DIR / "cases.json"
TOPOLOGY_FILE = STORAGE_DIR / "topologies.json"

mcp = FastMCP("memory-server")


def _load(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return []
    return []


def _save(path: Path, data: list):
    path.write_text(json.dumps(data, indent=2))


def _vec(delta_load: dict) -> list[float]:
    vals = []
    for k in sorted(delta_load):
        vals.append(delta_load[k].get("delta_p_kw",   0.0))
        vals.append(delta_load[k].get("delta_q_kvar", 0.0))
    return vals


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    ma   = math.sqrt(sum(x**2 for x in a))
    mb   = math.sqrt(sum(x**2 for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


# =============================================================================
# TOOLS  –  plain parameters
# =============================================================================

@mcp.tool()
async def retrieve_similar_cases(
    topology: dict,
    delta_load: dict,
    query_hint: str = "",
    top_k: int = 5,
) -> list:
    """Return top-k most similar historical cases by cosine similarity on load vectors."""
    cases = _load(CASES_FILE)
    if not cases:
        return []
    qv = _vec(delta_load)
    scored = []
    for case in cases:
        cv    = _vec(case.get("delta_load", {}))
        score = _cosine(qv, cv)
        topo_match = 1.0 if case.get("topology_id") == topology.get("topology_id") else 0.0
        total = 0.6 * score + 0.4 * topo_match
        if total > 0:
            scored.append({**case, "similarity_score": round(total, 4)})
    scored.sort(key=lambda c: c["similarity_score"], reverse=True)
    return scored[:top_k]


@mcp.tool()
async def save_case(
    topology_id: str,
    switch_states: dict,
    delta_load: dict,
    analysis_results: dict,
    recommendations: list,
    explanation: str,
    tags: list | None = None,
) -> dict:
    """Persist a solved DNR case. Returns {case_id, status}."""
    cases = _load(CASES_FILE)
    record = {
        "case_id":          str(uuid.uuid4()),
        "timestamp":        time.time(),
        "topology_id":      topology_id,
        "switch_states":    switch_states,
        "delta_load":       delta_load,
        "analysis_results": analysis_results,
        "recommendations":  recommendations,
        "explanation":      explanation,
        "tags":             tags or [],
    }
    cases.append(record)
    _save(CASES_FILE, cases)
    logger.info("Saved case %s", record["case_id"])
    return {"case_id": record["case_id"], "status": "saved"}


@mcp.tool()
async def topology_lookup(topology_id: str) -> dict:
    """Return stored topology metadata by ID."""
    for t in _load(TOPOLOGY_FILE):
        if t.get("topology_id") == topology_id:
            return t
    return {"error": f"Topology '{topology_id}' not found"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8002)