"""
DNR Copilot + IEEE 33-Bus Visualizer — Unified FastAPI Entry Point
==================================================================

Endpoints
---------
  GET  /              → Visualizer dashboard (http://localhost:8000)
  GET  /health        → Health check
  POST /reconfigure   → DNR LangGraph orchestration (LLM pipeline)

  GET  /viz/scenarios         → List IEEE 33-bus test scenarios
  GET  /viz/topology          → Bus positions + branch data
  POST /viz/run               → Real pandapower BFS before/after analysis
  GET  /viz/rag/{topic}       → RAG knowledge-base lookup

HOW TO RUN (from dnr-copilot/ directory)
-----------------------------------------
  uvicorn api.main:app --reload --port 8000
  → Dashboard:   http://localhost:8000
  → API docs:    http://localhost:8000/docs
"""

from __future__ import annotations

import sys, os, logging

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_api_dir = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_api_dir)
for _p in (_root, os.path.dirname(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App (defined before any imports that might fail) ──────────────────────────
app = FastAPI(
    title="DNR Copilot + IEEE 33-Bus Visualizer",
    description="LangGraph orchestration · Pandapower BFS · RAG explainability",
    version="2.0.0",
)

# ── Serve visualizer frontend ─────────────────────────────────────────────────
_frontend = os.path.join(_root, "visualizer", "frontend")
_static   = os.path.join(_frontend, "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.get("/", include_in_schema=False)
async def dashboard():
    idx = os.path.join(_frontend, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "Visualizer dashboard not found — check visualizer/frontend/index.html"}


# =============================================================================
# LAZY IMPORTS
# =============================================================================

_runner = None
def _get_runner():
    global _runner
    if _runner is None:
        from langgraph.orchestrator import run_dnr_copilot
        _runner = run_dnr_copilot
    return _runner

_engine = None
def _get_engine():
    global _engine
    if _engine is None:
        from visualizer.engine.ieee33_engine import (
            run_powerflow, compute_delta, rag_explain,
            SCENARIOS, BUS_POSITIONS, BRANCH_DATA, N, RAG_KB
        )
        _engine = {
            "run_powerflow": run_powerflow,
            "compute_delta": compute_delta,
            "rag_explain":   rag_explain,
            "SCENARIOS":     SCENARIOS,
            "BUS_POSITIONS": BUS_POSITIONS,
            "BRANCH_DATA":   BRANCH_DATA,
            "N":             N,
            "RAG_KB":        RAG_KB,
        }
    return _engine

# Cache base power flow result
_base_result = None
def _get_base():
    global _base_result
    if _base_result is None:
        e = _get_engine()
        _base_result = e["run_powerflow"]({})
    return _base_result


# =============================================================================
# SCHEMAS
# =============================================================================

class ReconfigureRequest(BaseModel):
    topology: dict[str, Any] = Field(..., description="Network topology")
    switch_states: dict[str, bool] = Field(..., description="switch_id → closed/open")
    delta_load: dict[str, dict[str, float]] = Field(default_factory=dict)

class ReconfigureResponse(BaseModel):
    analysis_results: dict[str, Any]
    recommendations: list[dict[str, Any]]
    explanation: str
    errors: list[str]

class VizRunRequest(BaseModel):
    scenario_key: str = "base"
    custom_changes: dict[str, Any] | None = None


# =============================================================================
# CORE ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "services": ["dnr-copilot", "ieee33-visualizer"]}


@app.post("/reconfigure", response_model=ReconfigureResponse,
          summary="DNR LangGraph orchestration (LLM pipeline)")
async def reconfigure(request: ReconfigureRequest):
    logger.info("Reconfigure | topology=%s | switches=%d",
                request.topology.get("topology_id"), len(request.switch_states))
    try:
        state = await _get_runner()(
            topology=request.topology,
            switch_states=request.switch_states,
            delta_load=request.delta_load,
        )
    except Exception as exc:
        logger.exception("Orchestration failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ReconfigureResponse(
        analysis_results=state.get("analysis_results", {}),
        recommendations=state.get("recommendations", []),
        explanation=state.get("explanation", ""),
        errors=state.get("errors", []),
    )


# =============================================================================
# VISUALIZER ENDPOINTS  (/viz/*)
# =============================================================================

@app.get("/viz/scenarios", summary="List IEEE 33-bus test scenarios")
def viz_scenarios():
    e = _get_engine()
    return {
        k: {"label": v["label"], "description": v["description"]}
        for k, v in e["SCENARIOS"].items()
    }


@app.get("/viz/topology", summary="IEEE 33-bus bus positions and branches")
def viz_topology():
    e = _get_engine()
    return {
        "buses":    list(range(1, e["N"] + 1)),
        "positions":{str(k): list(v) for k, v in e["BUS_POSITIONS"].items()},
        "branches": [
            {"from": fb, "to": tb, "R": R, "X": X}
            for fb, tb, R, X, P, Q in e["BRANCH_DATA"]
        ],
    }


@app.post("/viz/run", summary="Run pandapower BFS before/after + RAG explanation")
def viz_run(req: VizRunRequest):
    e  = _get_engine()
    sc = e["SCENARIOS"].get(req.scenario_key)
    if not sc:
        raise HTTPException(404, f"Unknown scenario: {req.scenario_key}")

    changes = dict(sc["changes"])
    if req.custom_changes:
        changes.update(req.custom_changes)

    before = _get_base()
    after  = e["run_powerflow"](changes)

    if not after.get("converged"):
        raise HTTPException(500, "Power flow did not converge.")

    delta        = e["compute_delta"](before, after)
    explanations = e["rag_explain"](req.scenario_key, before, after, delta)

    return {
        "scenario":     req.scenario_key,
        "label":        sc["label"],
        "description":  sc["description"],
        "before":       before,
        "after":        after,
        "delta":        delta,
        "explanations": explanations,
    }


@app.get("/viz/rag/topics", summary="List RAG knowledge-base topics")
def viz_rag_topics():
    e = _get_engine()
    return {k: {"title": v["title"], "severity": v["severity"]} for k, v in e["RAG_KB"].items()}


@app.get("/viz/rag/{topic}", summary="Look up a RAG knowledge entry")
def viz_rag(topic: str):
    e = _get_engine()
    if topic not in e["RAG_KB"]:
        raise HTTPException(404, "Topic not found")
    return e["RAG_KB"][topic]


# =============================================================================
# DIRECT EXECUTION
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n  DNR Copilot + Visualizer")
    print("  → http://localhost:8000  (dashboard)")
    print("  → http://localhost:8000/docs  (API)\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# =============================================================================
# VISUALIZER SECTION ENDPOINTS  (/viz/s1 … /viz/s10 and /viz/full)
# =============================================================================

class SectionRequest(BaseModel):
    p_load_kw:   list[float] | None = None
    q_load_kvar: list[float] | None = None

class S3Request(BaseModel):
    perturb_pct: list[float] | None = None
    p_load_kw:   list[float] | None = None
    q_load_kvar: list[float] | None = None

class S4Request(BaseModel):
    delta_q_kvar: float = 50.0
    p_load_kw:    list[float] | None = None
    q_load_kvar:  list[float] | None = None

class S5Request(BaseModel):
    near_slack:  list[int]   | None = None
    mid_feeder:  list[int]   | None = None
    far_end:     list[int]   | None = None
    perturb_pct: float = 0.20
    p_load_kw:   list[float] | None = None
    q_load_kvar: list[float] | None = None

class S6Request(BaseModel):
    p_load_kw:   list[float] | None = None
    q_load_kvar: list[float] | None = None
    eci_weights: dict        | None = None

class S8Request(BaseModel):
    p_load_kw:              list[float] | None = None
    q_load_kvar:            list[float] | None = None
    instability_threshold:  float = 0.5


_eng_cache = None

def _eng():
    """Load section functions directly from file to avoid mcp namespace clash."""
    global _eng_cache
    if _eng_cache is not None:
        return _eng_cache
    import importlib.util
    _srv = os.path.join(_root, "mcp", "visualizer_server", "server.py")
    spec = importlib.util.spec_from_file_location("viz_server", _srv)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _eng_cache = dict(
        s1=mod.s1_base_powerflow,
        s2=mod.s2_electrical_distance,
        s3=mod.s3_disturbance_propagation,
        s4=mod.s4_reactive_sensitivity,
        s5=mod.s5_feeder_zone_vulnerability,
        s6=mod.s6_influence_indices,
        s9=mod.s9_advanced_viz_data,
        s10=mod.s10_research_insights,
        full=mod.run_full_analysis,
    )
    return _eng_cache


@app.post("/viz/s1", summary="S1 Base power flow")
async def viz_s1(r: SectionRequest):
    return await _eng()["s1"](r.p_load_kw, r.q_load_kvar)

@app.post("/viz/s2", summary="S2 Electrical distance")
async def viz_s2(r: SectionRequest):
    return await _eng()["s2"](r.p_load_kw, r.q_load_kvar)

@app.post("/viz/s3", summary="S3 Active power disturbance propagation")
async def viz_s3(r: S3Request):
    return await _eng()["s3"](r.perturb_pct, r.p_load_kw, r.q_load_kvar)

@app.post("/viz/s4", summary="S4 Reactive sensitivity dV/dQ")
async def viz_s4(r: S4Request):
    return await _eng()["s4"](r.delta_q_kvar, r.p_load_kw, r.q_load_kvar)

@app.post("/viz/s5", summary="S5 Feeder zone vulnerability")
async def viz_s5(r: S5Request):
    return await _eng()["s5"](r.near_slack, r.mid_feeder, r.far_end,
                               r.perturb_pct, r.p_load_kw, r.q_load_kvar)

@app.post("/viz/s6", summary="S6 BII, VFI, DPF, ECI, Instability indices")
async def viz_s6(r: S6Request):
    return await _eng()["s6"](r.p_load_kw, r.q_load_kvar, r.eci_weights)

@app.post("/viz/s7", summary="S7 Graph-theoretic analysis")
async def viz_s7(r: SectionRequest):
    return await _eng()["s7"](r.p_load_kw, r.q_load_kvar)

@app.post("/viz/s8", summary="S8 GNN node+edge feature dataset")
async def viz_s8(r: S8Request):
    return await _eng()["s8"](r.p_load_kw, r.q_load_kvar, r.instability_threshold)

@app.post("/viz/s9", summary="S9 Advanced composite visualization data")
async def viz_s9(r: SectionRequest):
    return await _eng()["s9"](r.p_load_kw, r.q_load_kvar)

@app.post("/viz/s10", summary="S10 Automated research insights")
async def viz_s10(r: SectionRequest):
    return await _eng()["s10"](r.p_load_kw, r.q_load_kvar)

@app.post("/viz/full", summary="Run all 10 sections in one call")
async def viz_full(r: SectionRequest):
    return await _eng()["full"](r.p_load_kw, r.q_load_kvar)