"""
Analysis MCP Server  –  FastMCP 3.x compatible
===============================================
Tools use plain function parameters (no Pydantic request wrapper).
Run: python mcp/analysis_server/server.py
"""

from __future__ import annotations

import logging
import math
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("analysis-server")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _active_branches(branches: list, switch_states: dict) -> list:
    return [b for b in branches
            if b.get("switch_id") is None or switch_states.get(b["switch_id"], True)]


def _apply_delta(buses: list, delta_load: dict) -> list:
    bus_map = {b["id"]: dict(b) for b in buses}
    for node_id, delta in delta_load.items():
        if node_id in bus_map:
            bus_map[node_id]["p_load_kw"]   = bus_map[node_id].get("p_load_kw",   0.0) + delta.get("delta_p_kw",   0.0)
            bus_map[node_id]["q_load_kvar"]  = bus_map[node_id].get("q_load_kvar", 0.0) + delta.get("delta_q_kvar", 0.0)
    return list(bus_map.values())


def _bfs(buses: list, branches: list, substation: str,
         max_iter: int = 50, tol: float = 1e-4) -> dict:
    bus_map = {b["id"]: b for b in buses}
    v       = {b["id"]: 1.0 for b in buses}
    angle   = {b["id"]: 0.0 for b in buses}

    children: dict[str, list] = {b["id"]: [] for b in buses}
    for br in branches:
        children.setdefault(br["from_bus"], []).append(br)

    bp: dict[str, float] = {}
    bq: dict[str, float] = {}

    def backward(node: str):
        p = bus_map[node].get("p_load_kw",   0.0) / 1000.0
        q = bus_map[node].get("q_load_kvar", 0.0) / 1000.0
        for br in children.get(node, []):
            child = br["to_bus"]
            cp, cq = backward(child)
            r, x = br.get("r_ohm", 0.1), br.get("x_ohm", 0.1)
            i2 = (cp**2 + cq**2) / max(v[child]**2, 1e-6)
            p += cp + r * i2
            q += cq + x * i2
            bp[br["id"]] = cp
            bq[br["id"]] = cq
        return p, q

    def forward(node: str, vp: float, ap: float):
        for br in children.get(node, []):
            child = br["to_bus"]
            r, x = br.get("r_ohm", 0.1), br.get("x_ohm", 0.1)
            p, q = bp.get(br["id"], 0.0), bq.get(br["id"], 0.0)
            v[child] = max(vp - (r * p + x * q) / max(vp, 1e-6), 0.5)
            angle[child] = ap - math.degrees(math.atan2(x * p - r * q, vp**2))
            forward(child, v[child], angle[child])

    converged = False
    for it in range(max_iter):
        v_old = dict(v)
        backward(substation)
        v[substation] = 1.0
        forward(substation, 1.0, 0.0)
        if max(abs(v[b] - v_old[b]) for b in v) < tol:
            converged = True
            break

    flows = {}
    for br in branches:
        p_kw   = bp.get(br["id"], 0.0) * 1000.0
        q_kvar = bq.get(br["id"], 0.0) * 1000.0
        s_kva  = math.sqrt(p_kw**2 + q_kvar**2)
        rating = br.get("rating_kva", 100.0)
        flows[br["id"]] = {
            "p_kw": round(p_kw, 3),
            "q_kvar": round(q_kvar, 3),
            "loading_pct": round(s_kva / rating * 100, 2) if rating else 0.0,
        }

    return {
        "converged": converged,
        "iterations": it + 1,
        "bus_voltages": {bid: round(vv, 6) for bid, vv in v.items()},
        "bus_angles_deg": {bid: round(aa, 4) for bid, aa in angle.items()},
        "branch_flows": flows,
    }


# =============================================================================
# TOOLS  –  plain parameters, no Pydantic request wrapper
# =============================================================================

@mcp.tool()
async def run_powerflow(
    topology: dict,
    switch_states: dict,
    delta_load: dict | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
) -> dict:
    """Run BFS power-flow. Returns bus voltages, branch flows, convergence."""
    active = _active_branches(topology["branches"], switch_states)
    buses  = _apply_delta(topology["buses"], delta_load or {})
    try:
        r = _bfs(buses, active, topology["substation_bus"], max_iterations, tolerance)
    except RecursionError:
        return {"converged": False, "iterations": 0, "bus_voltages": {},
                "bus_angles_deg": {}, "branch_flows": {},
                "message": "Failed: network may not be radial."}
    r["message"] = "Converged" if r["converged"] else f"Did not converge in {r['iterations']} iterations"
    return r


@mcp.tool()
async def voltage_profile(
    topology: dict,
    switch_states: dict,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
) -> dict:
    """Per-bus voltage magnitudes (p.u.) with violation flags."""
    active = _active_branches(topology["branches"], switch_states)
    r = _bfs(topology["buses"], active, topology["substation_bus"])
    voltages = r["bus_voltages"]
    violations = [
        {"bus_id": bid, "voltage_pu": v, "limit": "min" if v < v_min_pu else "max"}
        for bid, v in voltages.items() if v < v_min_pu or v > v_max_pu
    ]
    return {
        "bus_voltages_pu": voltages,
        "min_voltage": min(voltages.values(), default=0.0),
        "max_voltage": max(voltages.values(), default=0.0),
        "violation_buses": violations,
    }


@mcp.tool()
async def loss_calculation(
    topology: dict,
    switch_states: dict,
    delta_load: dict | None = None,
) -> dict:
    """Total and per-branch real/reactive power losses."""
    active = _active_branches(topology["branches"], switch_states)
    buses  = _apply_delta(topology["buses"], delta_load or {})
    r = _bfs(buses, active, topology["substation_bus"])

    branch_losses = {}
    total_p = total_q = 0.0
    for br in active:
        flows  = r["branch_flows"].get(br["id"], {})
        p, q   = flows.get("p_kw", 0.0), flows.get("q_kvar", 0.0)
        v_to   = r["bus_voltages"].get(br["to_bus"], 1.0)
        rr, x  = br.get("r_ohm", 0.1), br.get("x_ohm", 0.1)
        i2     = (p**2 + q**2) / max(v_to**2 * 1e6, 1e-6)
        pl, ql = rr * i2 / 1e3, x * i2 / 1e3
        branch_losses[br["id"]] = {"p_loss_kw": round(pl, 4), "q_loss_kvar": round(ql, 4)}
        total_p += pl; total_q += ql

    return {
        "total_real_loss_kw": round(total_p, 4),
        "total_reactive_loss_kvar": round(total_q, 4),
        "branch_losses": branch_losses,
    }


@mcp.tool()
async def constraint_check(
    topology: dict,
    switch_states: dict,
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    thermal_limit_pct: float = 100.0,
) -> dict:
    """Thermal and voltage constraint violations."""
    active = _active_branches(topology["branches"], switch_states)
    r = _bfs(topology["buses"], active, topology["substation_bus"])

    thermal = [
        {"branch_id": bid, "loading_pct": f["loading_pct"],
         "rating_kva": next((b["rating_kva"] for b in topology["branches"] if b["id"] == bid), None)}
        for bid, f in r["branch_flows"].items() if f["loading_pct"] > thermal_limit_pct
    ]
    voltage = [
        {"bus_id": bid, "voltage_pu": v, "limit": "min" if v < v_min_pu else "max"}
        for bid, v in r["bus_voltages"].items() if v < v_min_pu or v > v_max_pu
    ]
    return {
        "thermal_violations": thermal,
        "voltage_violations": voltage,
        "all_constraints_satisfied": not thermal and not voltage,
    }


@mcp.tool()
async def radiality_check(
    topology: dict,
    switch_states: dict,
) -> dict:
    """Verify radial (tree) topology via DFS cycle detection."""
    active   = _active_branches(topology["branches"], switch_states)
    bus_ids  = {b["id"] for b in topology["buses"]}
    adj: dict[str, list] = {b: [] for b in bus_ids}
    for br in active:
        adj[br["from_bus"]].append((br["to_bus"],   br["id"]))
        adj[br["to_bus"]].append((br["from_bus"], br["id"]))

    visited: set[str] = set()
    loops: list[list[str]] = []

    def dfs(node: str, parent: str | None, path: list[str]):
        visited.add(node)
        for nb, br_id in adj[node]:
            if nb == parent:
                continue
            if nb in visited:
                loops.append(path + [br_id])
            else:
                dfs(nb, node, path + [br_id])

    dfs(topology["substation_bus"], None, [])
    return {
        "is_radial": len(loops) == 0,
        "loops_detected": loops,
        "isolated_buses": [b for b in bus_ids if b not in visited],
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)