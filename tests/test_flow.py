"""
DNR Copilot – End-to-End Integration Test
==========================================
Demonstrates the full request flow described in Part 5:

  Step 1  User submits topology
  Step 2  Planner executes  →  produces plan
  Step 3  Analysis MCP tools execute  →  produces analysis_results
  Step 4  Memory MCP retrieves similar cases  →  produces similar_cases
  Step 5  Prediction MCP generates actions  →  produces recommendations
  Step 6  Recommendation Agent generates explanation
  Step 7  Final response returned

Run (with all three MCP servers already running):
    pytest tests/test_flow.py -v
    # or directly:
    python tests/test_flow.py
"""

from __future__ import annotations

import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)

# ── Sample inputs ──────────────────────────────────────────────────────────────

TOPOLOGY = {
    "topology_id": "feeder-A-33bus",
    "substation_bus": "bus_0",
    "buses": [
        {"id": "bus_0",  "voltage_kv": 11.0, "p_load_kw": 0,   "q_load_kvar": 0},
        {"id": "bus_1",  "voltage_kv": 11.0, "p_load_kw": 200, "q_load_kvar": 80},
        {"id": "bus_2",  "voltage_kv": 11.0, "p_load_kw": 300, "q_load_kvar": 120},
        {"id": "bus_3",  "voltage_kv": 11.0, "p_load_kw": 150, "q_load_kvar": 60},
        {"id": "bus_4",  "voltage_kv": 11.0, "p_load_kw": 250, "q_load_kvar": 100},
    ],
    "branches": [
        {"id": "br_01", "from_bus": "bus_0", "to_bus": "bus_1",
         "r_ohm": 0.5,  "x_ohm": 0.4, "rating_kva": 800, "switch_id": "sw_01"},
        {"id": "br_12", "from_bus": "bus_1", "to_bus": "bus_2",
         "r_ohm": 0.6,  "x_ohm": 0.5, "rating_kva": 600, "switch_id": "sw_12"},
        {"id": "br_23", "from_bus": "bus_2", "to_bus": "bus_3",
         "r_ohm": 0.4,  "x_ohm": 0.3, "rating_kva": 500, "switch_id": "sw_23"},
        {"id": "br_34", "from_bus": "bus_3", "to_bus": "bus_4",
         "r_ohm": 0.3,  "x_ohm": 0.25,"rating_kva": 400, "switch_id": "sw_34"},
        # Tie branch (normally open) – connects feeder tail to another substation
        {"id": "br_tie","from_bus": "bus_4", "to_bus": "bus_0",
         "r_ohm": 0.8,  "x_ohm": 0.6, "rating_kva": 600, "switch_id": "sw_tie"},
    ],
}

SWITCH_STATES: dict[str, bool] = {
    "sw_01":  True,   # closed (in-service)
    "sw_12":  True,
    "sw_23":  True,
    "sw_34":  True,
    "sw_tie": False,  # normally open tie switch
}

# Load spike at bus_2 and bus_3 (e.g., hot summer afternoon)
DELTA_LOAD: dict[str, dict[str, float]] = {
    "bus_2": {"delta_p_kw": 180.0, "delta_q_kvar": 70.0},
    "bus_3": {"delta_p_kw": 120.0, "delta_q_kvar": 50.0},
}


# =============================================================================
# STEP-BY-STEP TRACE
# =============================================================================

async def run_traced_flow() -> None:
    """
    Execute the full DNR orchestration and print each state transition.
    Requires all three MCP servers to be running (ports 8001-8003).
    """
    from dnr_copilot.langgraph.orchestrator import build_graph
    from dnr_copilot.langgraph.state.dnr_state import DNRState

    graph = build_graph()

    # ── Step 1: Initial state ──────────────────────────────────────────────────
    initial_state: DNRState = {
        "topology":     TOPOLOGY,
        "switch_states": SWITCH_STATES,
        "delta_load":   DELTA_LOAD,
        "errors":       [],
    }

    print("\n" + "═" * 70)
    print("STEP 1 – User submits topology + switch states + load changes")
    print("═" * 70)
    print(json.dumps({
        "topology_id":   TOPOLOGY["topology_id"],
        "buses":         len(TOPOLOGY["buses"]),
        "branches":      len(TOPOLOGY["branches"]),
        "closed_switches": [k for k, v in SWITCH_STATES.items() if v],
        "open_switches":   [k for k, v in SWITCH_STATES.items() if not v],
        "delta_load":    DELTA_LOAD,
    }, indent=2))

    # ── Stream through graph nodes ─────────────────────────────────────────────
    step = 1
    final_state: DNRState = initial_state

    async for event in graph.astream(initial_state):
        for node_name, node_output in event.items():
            step += 1
            final_state = {**final_state, **node_output}

            print(f"\n{'═'*70}")
            print(f"STEP {step} – Node: {node_name.upper()}")
            print("═" * 70)

            if node_name == "planner":
                plan = node_output.get("plan", {})
                print(json.dumps({
                    "feasible":         plan.get("feasible"),
                    "affected_zones":   plan.get("affected_zones"),
                    "required_analyses":plan.get("required_analyses"),
                    "retrieve_memory":  plan.get("retrieve_memory"),
                    "objective":        plan.get("prediction_params", {}).get("objective"),
                }, indent=2))

            elif node_name == "analysis":
                ar = node_output.get("analysis_results", {})
                print(json.dumps({
                    "converged":          ar.get("power_flow_converged"),
                    "voltage_violations": len(ar.get("voltage_violations", [])),
                    "thermal_violations": len(ar.get("thermal_violations", [])),
                    "total_loss_kw":      ar.get("total_losses_kw"),
                    "radiality_ok":       ar.get("radiality_ok"),
                    "summary":            ar.get("summary"),
                }, indent=2))

            elif node_name == "memory":
                cases = node_output.get("similar_cases", [])
                print(json.dumps({
                    "similar_cases_found": len(cases),
                    "top_case_score": cases[0].get("similarity_score") if cases else None,
                }, indent=2))

            elif node_name == "prediction":
                recs = node_output.get("recommendations", [])
                print(json.dumps({
                    "candidates_generated": len(recs),
                    "top_recommendation": {
                        "rank":            recs[0].get("rank"),
                        "open_switches":   recs[0].get("open_switches"),
                        "close_switches":  recs[0].get("close_switches"),
                        "rule":            recs[0].get("rule_applied"),
                        "confidence":      recs[0].get("confidence"),
                    } if recs else None,
                }, indent=2))

            elif node_name == "recommendation":
                print(json.dumps({
                    "final_actions":  len(node_output.get("recommendations", [])),
                    "explanation":    node_output.get("explanation"),
                    "errors":         node_output.get("errors", []),
                }, indent=2))

    # ── Step 7: Final response ─────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("STEP 7 – Final Response (returned to API caller)")
    print("═" * 70)
    print(json.dumps({
        "analysis_results":  final_state.get("analysis_results", {}),
        "recommendations":   final_state.get("recommendations", []),
        "explanation":       final_state.get("explanation", ""),
        "errors":            final_state.get("errors", []),
    }, indent=2))


# =============================================================================
# UNIT-TESTABLE HELPERS (no MCP servers needed)
# =============================================================================

def test_bfs_powerflow_isolated() -> None:
    """Unit test: BFS power-flow on 3-bus radial feeder (no MCP servers needed)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from dnr_copilot.mcp.analysis_server.server import _bfs_powerflow

    buses = [
        {"id": "b0", "p_load_kw": 0,   "q_load_kvar": 0},
        {"id": "b1", "p_load_kw": 100, "q_load_kvar": 40},
        {"id": "b2", "p_load_kw": 80,  "q_load_kvar": 30},
    ]
    branches = [
        {"id": "br01", "from_bus": "b0", "to_bus": "b1", "r_ohm": 0.5, "x_ohm": 0.4, "rating_kva": 500},
        {"id": "br12", "from_bus": "b1", "to_bus": "b2", "r_ohm": 0.3, "x_ohm": 0.2, "rating_kva": 400},
    ]
    result = _bfs_powerflow(buses, branches, "b0", max_iter=50, tol=1e-4)

    assert result["converged"], "Power flow should converge on simple 3-bus feeder"
    assert 0.9 <= result["bus_voltages"]["b2"] <= 1.0, "Voltage at b2 must be reasonable"
    print("✅ test_bfs_powerflow_isolated PASSED")


def test_similarity_scoring() -> None:
    """Unit test: cosine similarity scoring (no MCP servers needed)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from dnr_copilot.mcp.memory_server.server import _cosine_similarity, _load_vector

    d1 = {"bus_1": {"delta_p_kw": 100.0, "delta_q_kvar": 40.0}}
    d2 = {"bus_1": {"delta_p_kw": 100.0, "delta_q_kvar": 40.0}}
    d3 = {"bus_1": {"delta_p_kw": 0.0,   "delta_q_kvar": 0.0}}

    assert _cosine_similarity(_load_vector(d1), _load_vector(d2)) == 1.0, "Identical should be 1.0"
    assert _cosine_similarity(_load_vector(d1), _load_vector(d3)) == 0.0, "Zero vector should be 0.0"
    print("✅ test_similarity_scoring PASSED")


def test_rule_engine() -> None:
    """Unit test: rule engine produces candidates for voltage violation (no MCP)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from dnr_copilot.mcp.prediction_server.server import RuleEngine

    engine = RuleEngine(
        topology=TOPOLOGY,
        switch_states=SWITCH_STATES,
        analysis={
            "voltage_violations": [{"bus_id": "bus_4", "voltage_pu": 0.92, "limit": "min"}],
            "thermal_violations": [],
            "total_losses_kw": 5.0,
        },
        objective="restore_voltage",
        priority="safety",
    )
    candidates = engine.generate_candidates(max_candidates=5)
    assert len(candidates) >= 1, "Should produce at least one candidate"
    assert candidates[0].rank == 1
    print(f"✅ test_rule_engine PASSED – {len(candidates)} candidates generated")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n── Running isolated unit tests (no MCP servers needed) ──")
    test_bfs_powerflow_isolated()
    test_similarity_scoring()
    test_rule_engine()

    print("\n── Running full traced flow (requires MCP servers on 8001-8003) ──")
    print("   Set SKIP_INTEGRATION=1 to skip this step.\n")
    import os
    if not os.getenv("SKIP_INTEGRATION"):
        asyncio.run(run_traced_flow())
    else:
        print("   Integration flow skipped.")