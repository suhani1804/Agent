"""
DNR Copilot — Unified FastAPI
==============================
Endpoints:
  GET  /                    → Dashboard
  GET  /health
  POST /reconfigure         → LangGraph DNR orchestration
  POST /viz/run_all         → BFS once + all sections fast + background S3/S4
  GET  /viz/status/{qid}    → Poll background job status
  GET  /viz/background/{qid}→ Fetch S3/S4 when ready
  POST /viz/compare         → Compare two query IDs
  POST /viz/chat            → Chatbot (LLM interprets query → runs analysis)
  GET  /viz/memory          → Query history from Redis
  GET  /viz/memory/{qid}    → Full result for one query (from checkpointer)
  GET  /viz/topology        → Bus positions + branches
  GET  /viz/rag/query       → Query RAG knowledge base
"""

from __future__ import annotations
import sys, os, json, uuid, asyncio, time, logging, re

_api_dir = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_api_dir)
for _p in (_root, os.path.dirname(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DNR Copilot + Visualizer", version="3.0.0")

# ── Serve frontend ─────────────────────────────────────────────────────────────
_frontend = os.path.join(_root, "visualizer", "frontend")
_static   = os.path.join(_frontend, "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.get("/", include_in_schema=False)
async def dashboard():
    idx = os.path.join(_frontend, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else {"error": "index.html not found"}


# ── Lazy imports ───────────────────────────────────────────────────────────────
_engine = None
def eng():
    global _engine
    if _engine is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ieee33_engine",
            os.path.join(_root, "visualizer", "engine", "ieee33_engine.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _engine = mod
    return _engine

_runner = None
def get_runner():
    global _runner
    if _runner is None:
        from langgraph.orchestrator import run_dnr_copilot
        _runner = run_dnr_copilot
    return _runner

_redis = None
def rc():
    global _redis
    if _redis is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "redis_cache",
            os.path.join(_root, "memory", "redis_cache.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _redis = mod
    return _redis

_rag = None
def rag():
    global _rag
    if _rag is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rag_store",
            os.path.join(_root, "memory", "rag_store.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.seed_knowledge_base()
        _rag = mod
    return _rag

# In-process result store (fallback when checkpointer fails)
_result_store: dict = {}

# LangGraph SQLite checkpointer helpers
_CP_DB = None
def _cp_db_path():
    db_dir = os.path.join(_root, "memory", "checkpoints")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "queries.db")

def cp_save(query_id: str, data: dict) -> None:
    """Save full result into SQLite checkpointer."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.base import empty_checkpoint
    db_path = _cp_db_path()
    with SqliteSaver.from_conn_string(db_path) as saver:
        config = {"configurable": {
            "thread_id": query_id,
            "checkpoint_ns": "",
            "checkpoint_id": "",
        }}
        ckpt = empty_checkpoint()
        ckpt["channel_values"] = {"result": data}
        saver.put(config, ckpt,
                  {"source":"input","step":0,"writes":{},"parents":{}}, {})

def cp_load(query_id: str) -> dict | None:
    """Load full result from SQLite checkpointer."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    db_path = _cp_db_path()
    with SqliteSaver.from_conn_string(db_path) as saver:
        snaps = list(saver.list({"configurable": {"thread_id": query_id}}))
        if not snaps:
            return None
        return snaps[-1].checkpoint.get("channel_values", {}).get("result")

# Cache base BFS result
_base_bfs = None
def get_base_bfs():
    global _base_bfs
    if _base_bfs is None:
        cached = rc().get_pf_cache("base", {})
        if cached:
            import numpy as np
            for k in ["V_pu","V_ang","Im","loading","Pl","Ql","p_load","q_load"]:
                if k in cached:
                    cached[k] = np.array(cached[k])
            _base_bfs = cached
        else:
            _base_bfs = eng().run_bfs({})
            rc().set_pf_cache("base", {}, {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in _base_bfs.items()
            })
    return _base_bfs

_pred_server = None
def _pred():
    global _pred_server
    if _pred_server is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pred_server",
            os.path.join(_root, "mcp", "prediction_server", "server.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _pred_server = mod
    return _pred_server

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class RunAllRequest(BaseModel):
    scenario_key: str = "base"
    custom_changes: dict[str, Any] | None = None
    session_id: str = "default"

class CompareRequest(BaseModel):
    query_id_a: str
    query_id_b: str
    session_id: str = "default"

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    thread_id: str | None = None

class ReconfigureRequest(BaseModel):
    topology: dict[str, Any]
    switch_states: dict[str, bool]
    delta_load: dict[str, dict[str, float]] = {}

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3


class GNNPredictRequest(BaseModel):
    scenario_key: str = "base"
    custom_changes: dict | None = None
    top_k: int = 3



# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_background(query_id: str, bfs: dict, changes: dict):
    """Runs S3 + S4 in background thread, stores result in Redis."""
    import numpy as np
    rc_ = rc(); e = eng()

    rc_.set_background_status(query_id, "running")
    try:
        # Restore numpy arrays
        bfs2 = dict(bfs)
        for k in ["V_pu","V_ang","Im","loading","Pl","Ql","p_load","q_load"]:
            if k in bfs2 and isinstance(bfs2[k], list):
                bfs2[k] = np.array(bfs2[k])

        s3 = e.compute_s3_background(bfs2, changes)
        s4 = e.compute_s4_background(bfs2)

        # Recompute S6/S9/S10 with real S3 data (replaces proxy)
        s6_real  = e.compute_s6(bfs2, s3)
        s9_real  = e.compute_s9(bfs2, s3, s6_real)
        s10_real = e.compute_s10(bfs2, s3, s6_real)

        result = {"s3": s3, "s4": s4, "s6": s6_real,
                  "s9": s9_real, "s10": s10_real}

        rc_.set_background_result(query_id, result)
        rc_.set_background_status(query_id, "done")
        logger.info("[BG] query %s background complete", query_id)

    except Exception as exc:
        logger.exception("[BG] query %s failed: %s", query_id, exc)
        rc_.set_background_status(query_id, "error")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/viz/run_all")
async def run_all(req: RunAllRequest, bg: BackgroundTasks):
    """
    Main analysis endpoint.
    1. BFS runs exactly once
    2. Fast sections (S1 S2 S5 S6proxy S9proxy S10proxy) returned immediately
    3. S3 S4 run in background — poll /viz/status/{query_id}
    4. Result stored in Redis + LangGraph checkpointer
    """
    import numpy as np
    e  = rc_mod = rc(); e = eng()
    sc = e.SCENARIOS.get(req.scenario_key)
    if not sc:
        raise HTTPException(404, f"Unknown scenario: {req.scenario_key}")

    changes = dict(sc["changes"])
    if req.custom_changes:
        changes.update(req.custom_changes)

    _ts = int(time.time())
    _lbl = re.sub(r"[^a-zA-Z0-9_-]", "_", sc["label"][:20]).strip("_")
    query_id  = f"q{_ts}_{_lbl}"
    clean_qid = query_id

    # Check Redis cache first
    cached_full = rc_mod.get_pf_cache(req.scenario_key, changes)

    if cached_full:
        logger.info("[Cache HIT] %s", req.scenario_key)
        import numpy as np
        scen_bfs = dict(cached_full)
        for k in ["V_pu","V_ang","Im","loading","Pl","Ql","p_load","q_load"]:
            if k in scen_bfs and isinstance(scen_bfs[k], list):
                scen_bfs[k] = np.array(scen_bfs[k])
    else:
        logger.info("[BFS] running once for %s", req.scenario_key)
        scen_bfs = e.run_bfs(changes)
        serialisable = {k: v.tolist() if hasattr(v,"tolist") else v
                        for k, v in scen_bfs.items()}
        rc_mod.set_pf_cache(req.scenario_key, changes, serialisable)

    base_bfs = get_base_bfs()
    bl = [f"{fb}-{tb}" for fb, tb, *_ in e.BRANCH_DATA]

    # Fast sections — instant
    s1  = e.compute_s1(base_bfs, scen_bfs, bl)
    s2  = e.compute_s2(scen_bfs)
    s5  = e.compute_s5(scen_bfs)
    s6p = e.compute_s6(scen_bfs)          # proxy (no S3 yet)
    s9p = e.compute_s9(scen_bfs, None, s6p)
    s10p= e.compute_s10(scen_bfs, None, s6p)

    # RAG knowledge retrieval
    rag_mod = rag()
    scenario_text = (
        f"{sc['label']} "
        f"min voltage {s1['after']['min_V']} pu "
        f"losses {s1['after']['totPl']} kW "
        f"buses below 0.95: {s1['after']['buses_below_095']}"
    )
    knowledge = rag_mod.query_knowledge(scenario_text, top_k=4)
    similar   = rag_mod.retrieve_similar_cases(scenario_text, top_k=3)

    # Save to Redis query history
    rc_mod.save_query_meta(clean_qid, {
        "label":         query_id,
        "scenario_key":  req.scenario_key,
        "timestamp":     time.time(),
        "min_V":         s1["after"]["min_V"],
        "total_losses":  s1["after"]["totPl"],
        "max_loading":   s1["after"]["max_loading"],
        "buses_below_095": s1["after"]["buses_below_095"],
        "thread_id":     clean_qid,
    })

    # Save to RAG case store
    rag_mod.save_case(
        query_id=clean_qid,
        label=sc["label"],
        scenario_key=req.scenario_key,
        analysis=s1["after"],
        recommendations=[],
        explanation=f"Scenario: {sc['description']}",
    )

    # Store full result in checkpointer (LangGraph SQLite)
    full_result = {
        "query_id": clean_qid,
        "scenario_key": req.scenario_key,
        "label": sc["label"],
        "s1": s1, "s2": s2, "s5": s5,
        "s6": s6p, "s9": s9p, "s10": s10p,
        "knowledge": knowledge,
        "similar_cases": similar,
    }
    try:
        cp_save(clean_qid, full_result)
    except Exception as exc:
        logger.warning("Checkpointer save failed: %s", exc)
    # Always save to in-process store as ultimate fallback
    _result_store[clean_qid] = full_result

    # Launch background S3/S4
    bfs_serialisable = {k: v.tolist() if hasattr(v,"tolist") else v
                        for k, v in scen_bfs.items()}
    rc_mod.set_background_status(clean_qid, "running")
    bg.add_task(_run_background, clean_qid, bfs_serialisable, changes)

    return {
        "query_id":    clean_qid,
        "label":       sc["label"],
        "description": sc["description"],
        "s1": s1, "s2": s2, "s5": s5,
        "s6": s6p, "s9": s9p, "s10": s10p,
        "knowledge":     knowledge,
        "similar_cases": similar,
        "background_status": "running",
    }


@app.get("/viz/status/{query_id}")
async def bg_status(query_id: str):
    status = rc().get_background_status(query_id)
    return {"query_id": query_id, "status": status}


@app.get("/viz/background/{query_id}")
async def bg_result(query_id: str):
    status = rc().get_background_status(query_id)
    if status != "done":
        return {"status": status, "ready": False}
    data = rc().get_background_result(query_id)
    if not data:
        raise HTTPException(404, "Background result not found")
    return {"status": "done", "ready": True, **data}


@app.post("/viz/compare")
async def compare(req: CompareRequest):
    """Pull two query results and compute overlay delta."""
    rc_mod = rc(); e = eng()

    # Store selection in Redis
    rc_mod.set_comparison(req.session_id, [req.query_id_a, req.query_id_b])

    # Try checkpointer first, fall back to Redis BFS cache
    results = {}
    for qid in [req.query_id_a, req.query_id_b]:
        try:
            data = cp_load(qid)
            if data:
                results[qid] = data
                continue
        except Exception as exc:
            logger.warning("Checkpointer read %s failed: %s", qid, exc)
        # Fallback 1: in-process store
        if qid in _result_store:
            results[qid] = _result_store[qid]
            continue
        # Fallback 2: Redis query metadata only
        try:
            meta = rc().get_query_meta(qid)
            if meta:
                results[qid] = {"query_id": qid, "label": meta.get("label", qid),
                                 "s1": {}, "_meta_only": True}
        except Exception:
            pass

    if len(results) < 2:
        found = list(results.keys())
        missing = [q for q in [req.query_id_a, req.query_id_b] if q not in found]
        raise HTTPException(404,
            f"Could not load queries: {missing}. "
            f"These queries may have been lost when the server restarted "
            f"(fakeredis is in-memory only). Run both scenarios again first.")

    ra = results[req.query_id_a]
    rb = results[req.query_id_b]
    delta = e.compare_queries(ra, rb)

    return {
        "query_id_a":  req.query_id_a,
        "query_id_b":  req.query_id_b,
        "label_a":     ra.get("label", req.query_id_a),
        "label_b":     rb.get("label", req.query_id_b),
        "delta":       delta,
        "s1_a":        ra.get("s1", {}),
        "s1_b":        rb.get("s1", {}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

CHAT_SYSTEM = """\
You are DNR Copilot, an AI assistant for IEEE 33-bus Distribution Network
Reconfiguration analysis.

You help operators analyse network scenarios. When a user describes a scenario,
extract the parameters and respond with a JSON action object.

SCENARIOS AVAILABLE:
base, load_spike_bus18, load_spike_bus25, dg_bus18, dg_bus33,
cap_bus18, cap_bus30, outage_bus25, heavy_load, light_load

If the user query maps to one of these scenarios, respond with:
{"action": "run_scenario", "scenario_key": "<key>", "explanation": "<brief>"}

If the user describes a CUSTOM scenario (e.g. "add 300kW DG at bus 12"), respond with:
{"action": "run_custom", "changes": {...}, "label": "<short label>", "explanation": "<brief>"}

where changes can contain:
  "global_scale": float
  "bus_load_scale": {bus_number: scale_factor}
  "dg": {bus_number: {"p_mw": float, "q_mvar": float}}
  "capacitor": {bus_number: q_mvar_float}

If information is MISSING or AMBIGUOUS, respond with:
{"action": "ask", "question": "<specific question to ask the operator>"}

If the user asks a QUESTION about results, explain using your knowledge:
{"action": "explain", "answer": "<explanation>"}

If the user wants to COMPARE two past queries:
{"action": "compare", "query_id_a": "<id>", "query_id_b": "<id>"}

Always respond with valid JSON only. No markdown, no prose outside JSON.
"""

# ── Fast keyword router (no LLM needed) ──────────────────────────────────────
def _fast_route(message: str) -> dict | None:
    """
    Instant keyword-based scenario extraction.
    Returns action dict or None if no match found.
    No API call, no latency.
    """
    import re
    m = message.lower().strip()

    # ── Compare intent ──
    if "compare" in m:
        return {"action": "compare_intent",
                "message": "Please select two queries from the history panel on the left and click 'Compare A vs B'."}

    # ── Explicit scenario keywords ──
    KEYWORD_MAP = [
        (["base case", "base", "default", "normal", "standard"],             "base"),
        (["heavy load", "+30", "load +30", "30% load", "peak"],             "heavy_load"),
        (["light load", "-40", "40% less", "night", "low load"],             "light_load"),
        (["bus 18", "b18", "bus18"],                                          None),   # needs disambiguation
        (["bus 25", "b25", "bus25"],                                          None),
        (["bus 33", "b33", "bus33"],                                          None),
        (["outage", "fault", "de-energis", "disconnect bus 25"],              "outage_bus25"),
        (["cap", "capacitor", "reactive comp"],                               None),
    ]

    # ── DG injection ──
    dg_match = re.search(r'(\d+)\s*k[wW].*?(?:dg|pv|solar|inject|generat).*?bus\s*(\d+)', m)
    if not dg_match:
        dg_match = re.search(r'(?:dg|pv|solar|inject|generat).*?(\d+)\s*k[wW].*?bus\s*(\d+)', m)
    if not dg_match:
        dg_match = re.search(r'(?:add|install).*?(\d+)\s*k[wW].*?bus\s*(\d+)', m)
    if not dg_match:
        dg_match = re.search(r'bus\s*(\d+).*?(\d+)\s*k[wW]', m)

    if dg_match:
        grps = dg_match.groups()
        # figure out which group is kW and which is bus
        try:
            g0, g1 = int(grps[0]), int(grps[1])
            # if first number looks like a bus (1-33) and second like kW (>33)
            if g0 <= 33 and g1 > 33:
                bus_num, kw = g0, g1
            else:
                kw, bus_num = g0, g1
            p_mw = kw / 1000.0
            return {
                "action": "run_custom",
                "changes": {"dg": {bus_num: {"p_mw": p_mw, "q_mvar": 0.0}}},
                "label": f"DG {kw}kW at Bus {bus_num}",
                "explanation": f"Adding {kw}kW DG at bus {bus_num}.",
            }
        except Exception:
            pass

    # ── Load spike ──
    load_match = re.search(r'(?:load|demand).*?(\+?\d+)\s*%.*?bus\s*(\d+)', m)
    if not load_match:
        load_match = re.search(r'bus\s*(\d+).*?(?:load|demand).*?(\+?\d+)\s*%', m)
    if load_match:
        try:
            g0, g1 = load_match.groups()
            try:
                bus_num = int(g0); pct = int(g1.replace("+",""))
            except Exception:
                bus_num = int(g1); pct = int(g0.replace("+",""))
            scale = 1.0 + pct / 100.0
            return {
                "action": "run_custom",
                "changes": {"bus_load_scale": {bus_num: scale}},
                "label": f"Load +{pct}% at Bus {bus_num}",
                "explanation": f"Increasing load at bus {bus_num} by {pct}%.",
            }
        except Exception:
            pass

    # ── Capacitor ──
    cap_match = re.search(r'(\d+)\s*k[vV][aA][rR].*?bus\s*(\d+)', m)
    if not cap_match:
        cap_match = re.search(r'(?:cap|capacitor).*?bus\s*(\d+)', m)
    if cap_match:
        grps = cap_match.groups()
        try:
            if len(grps) == 2:
                kvar, bus_num = int(grps[0]), int(grps[1])
            else:
                bus_num, kvar = int(grps[0]), 600
            return {
                "action": "run_custom",
                "changes": {"capacitor": {bus_num: kvar / 1000.0}},
                "label": f"Cap {kvar}kVAR at Bus {bus_num}",
                "explanation": f"Adding {kvar}kVAR capacitor at bus {bus_num}.",
            }
        except Exception:
            pass

    # ── Named scenario keywords ──
    for keywords, key in [
        (["base case","base","default","standard","normal"], "base"),
        (["heavy load","+30%","30 percent","peak demand"],   "heavy_load"),
        (["light load","-40%","low load","night"],           "light_load"),
        (["load spike bus 18","bus 18 load","bus18 load"],   "load_spike_bus18"),
        (["load spike bus 25","bus 25 load","bus25 load"],   "load_spike_bus25"),
        (["dg bus 18","dg at bus 18","bus 18 dg","bus18 dg"],"dg_bus18"),
        (["dg bus 33","dg at bus 33","bus 33 dg","bus33 dg"],"dg_bus33"),
        (["cap bus 18","capacitor bus 18","bus 18 cap"],      "cap_bus18"),
        (["cap bus 30","capacitor bus 30","bus 30 cap"],      "cap_bus30"),
        (["outage","bus 25 out","disconnect bus 25"],         "outage_bus25"),
    ]:
        if key and any(k in m for k in keywords):
            return {"action": "run_scenario", "scenario_key": key,
                    "explanation": f"Running {key} scenario."}

    return None   # no match → use LLM


@app.post("/viz/chat")
async def chat(req: ChatRequest, bg: BackgroundTasks):
    """
    Chatbot endpoint.
    1. Try instant keyword routing (no LLM, no latency)
    2. Fall back to LLM only if keyword routing fails
    """
    rc_mod = rc()

    # Build conversation history
    history_raw = rc_mod.get_redis().get(f"chat:{req.session_id}")
    history = json.loads(history_raw) if history_raw else []
    history.append({"role": "user", "content": req.message})

    def _save_history(reply: str):
        history.append({"role": "assistant", "content": reply})
        try:
            rc_mod.get_redis().setex(
                f"chat:{req.session_id}", 3600, json.dumps(history[-20:])
            )
        except Exception:
            pass

    async def _dispatch(action: dict) -> dict:
        act = action.get("action")

        if act == "compare_intent":
            _save_history(action["message"])
            return {"type": "question", "message": action["message"], "history": history[-6:]}

        if act == "ask":
            msg = action.get("question", "Could you clarify?")
            _save_history(msg)
            return {"type": "question", "message": msg, "history": history[-6:]}

        if act == "explain":
            knowledge = rag().query_knowledge(req.message, top_k=2)
            msg = action.get("answer", "")
            _save_history(msg)
            return {"type": "explanation", "message": msg,
                    "knowledge": knowledge, "history": history[-6:]}

        if act == "compare":
            compare_req = CompareRequest(
                query_id_a=action.get("query_id_a", ""),
                query_id_b=action.get("query_id_b", ""),
                session_id=req.session_id,
            )
            result = await compare(compare_req)
            _save_history("Comparison ready.")
            return {"type": "comparison", "message": "Comparison ready.",
                    "result": result, "history": history[-6:]}

        if act in ("run_scenario", "run_custom"):
            scenario_key   = action.get("scenario_key", "base")
            custom_changes = action.get("changes")
            label_override = action.get("label")
            run_req = RunAllRequest(
                scenario_key=scenario_key,
                custom_changes=custom_changes,
                session_id=req.session_id,
            )
            result = await run_all(run_req, bg)
            if label_override:
                result["label"] = label_override
            msg = action.get("explanation", f"Running {result['label']}…")
            _save_history(msg)
            return {"type": "analysis", "message": msg,
                    "result": result, "history": history[-6:]}

        return {"type": "unknown", "raw": str(action), "history": history[-6:]}

    # ── Step 1: instant keyword routing ───────────────────────────────────────
    fast = _fast_route(req.message)
    if fast:
        logger.info("[Chat] Fast route matched: %s", fast.get("action"))
        return await _dispatch(fast)

    # ── Step 2: LLM fallback (only for ambiguous queries) ────────────────────
    logger.info("[Chat] No fast route — calling LLM")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        messages = [{"role": "system", "content": CHAT_SYSTEM}] + history[-10:]
        resp = await client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0, max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            _save_history(raw)
            return {"type": "explain", "message": raw, "history": history[-6:]}
        return await _dispatch(action)

    except Exception as exc:
        # If LLM fails (quota, network), ask the user to rephrase
        logger.warning("[Chat] LLM failed: %s", exc)
        msg = ("I couldn't interpret that query automatically. "
               "Try phrases like: 'Base case', 'Add 500kW DG at bus 18', "
               "'Heavy load', 'Load +50% at bus 25', 'Capacitor 600kVAR at bus 30'.")
        _save_history(msg)
        return {"type": "question", "message": msg, "history": history[-6:]}


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/viz/memory")
async def get_memory(limit: int = 20):
    """Return recent query history from Redis sorted set."""
    history = rc().get_query_history(limit)
    return {"queries": history, "count": len(history)}


@app.get("/viz/memory/{query_id}")
async def get_memory_item(query_id: str):
    """Return full result for a specific query from checkpointer."""
    try:
        data = cp_load(query_id)
        if not data:
            raise HTTPException(404, f"Query {query_id} not found")
        return data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.delete("/viz/memory/{query_id}")
async def delete_memory_item(query_id: str):
    rc().delete_query(query_id)
    return {"deleted": query_id}


# ═══════════════════════════════════════════════════════════════════════════════
# RAG ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/viz/rag/query")
async def rag_query(req: RAGQueryRequest):
    knowledge = rag().query_knowledge(req.query, req.top_k)
    similar   = rag().retrieve_similar_cases(req.query, req.top_k)
    return {"knowledge": knowledge, "similar_cases": similar}


@app.get("/viz/rag/cases")
async def rag_cases():
    return {"cases": rag().get_all_cases()}


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY + SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/viz/topology")
def topology():
    e = eng()
    return {
        "buses":    list(range(1, e.N + 1)),
        "positions":{str(k): list(v) for k, v in e.BUS_POSITIONS.items()},
        "branches": [{"from": fb, "to": tb, "R": R, "X": X}
                     for fb, tb, R, X, *_ in e.BRANCH_DATA],
    }

@app.get("/viz/scenarios")
def scenarios():
    e = eng()
    return {k: {"label": v["label"], "description": v["description"]}
            for k, v in e.SCENARIOS.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# DNR COPILOT (LangGraph)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/reconfigure")
async def reconfigure(req: ReconfigureRequest):
    try:
        state = await get_runner()(
            topology=req.topology,
            switch_states=req.switch_states,
            delta_load=req.delta_load,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {
        "analysis_results": state.get("analysis_results", {}),
        "recommendations":  state.get("recommendations", []),
        "explanation":      state.get("explanation", ""),
        "errors":           state.get("errors", []),
    }

@app.post("/viz/gnn_predict")
async def gnn_predict(req: GNNPredictRequest):
    """
    Run GraphTransformer prediction on a scenario.
    Returns predicted losses, min voltage, max loading.
    Also returns switching recommendations.
    """
    e = eng()
    sc = e.SCENARIOS.get(req.scenario_key)
    if not sc:
        raise HTTPException(404, f"Unknown scenario: {req.scenario_key}")

    changes = dict(sc["changes"])
    if req.custom_changes:
        changes.update(req.custom_changes)

    # Check Redis cache first
    rc_mod = rc()
    cached = rc_mod.get_pf_cache(req.scenario_key, changes)
    if cached:
        import numpy as np
        bfs = dict(cached)
        for k in ["V_pu","V_ang","Im","loading","Pl","Ql","p_load","q_load"]:
            if k in bfs and isinstance(bfs[k], list):
                bfs[k] = np.array(bfs[k])
    else:
        bfs = e.run_bfs(changes)
        rc_mod.set_pf_cache(req.scenario_key, changes, {
            k: v.tolist() if hasattr(v,"tolist") else v for k,v in bfs.items()
        })

    bfs_json = {k: v.tolist() if hasattr(v,"tolist") else v for k,v in bfs.items()}
    pred_mod = _pred()

    gnn_result  = await pred_mod.predict_gnn(bfs_json, sc["label"])
    sw_result   = await pred_mod.predict_switching(bfs_json, sc["label"], req.top_k)

    return {
        "scenario_key":  req.scenario_key,
        "label":         sc["label"],
        "gnn_prediction":   gnn_result,
        "switching_recommendation": sw_result,
    }


@app.get("/viz/gnn_info")
async def gnn_info():
    return await _pred().model_info()

if __name__ == "__main__":
    import uvicorn
    print("\n  DNR Copilot v3 — http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")