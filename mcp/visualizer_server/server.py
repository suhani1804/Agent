"""
Visualizer MCP Server  –  IEEE 33-Bus Full Analysis
====================================================
Every section from the notebook is a separate MCP tool.
All inputs are JSON-serialisable dicts/lists.
All calculations happen inside the function.
Port: 8004

Tools (one per notebook section)
---------------------------------
  s1_base_powerflow          – Section 1 : BFS power flow, base case metrics
  s2_electrical_distance     – Section 2 : Cumulative impedance, elec. distance matrix
  s3_disturbance_propagation – Section 3 : Active-power influence matrices, BII
  s4_reactive_sensitivity    – Section 4 : dV/dQ matrix, self-sensitivity, weak buses
  s5_feeder_zone_vulnerability – Section 5 : Zone comparison, feeder-end degradation
  s6_influence_indices       – Section 6 : BII, VFI, DPF, ECI, Instability scores
  s7_graph_analysis          – Section 7 : Betweenness, closeness, Laplacian, Fiedler
  s8_gnn_dataset             – Section 8 : Node + edge feature matrix for GNN
  s9_advanced_viz_data       – Section 9 : Composite multi-metric data for dashboard
  s10_research_insights      – Section 10: Automated correlations and summary metrics
  run_full_analysis          – Run all sections in one call, return everything

Run: python mcp/visualizer_server/server.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg as la

from fastmcp import FastMCP

mcp = FastMCP("visualizer-server")

# ═══════════════════════════════════════════════════════════════════════════════
# IEEE 33-BUS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BRANCH_DATA = [
    (1,  2,  0.0922, 0.0477,   0,   0),
    (2,  3,  0.4930, 0.2511, 100,  60),
    (3,  4,  0.3660, 0.1864,  90,  40),
    (4,  5,  0.3811, 0.1941, 120,  80),
    (5,  6,  0.8190, 0.7070,  60,  30),
    (6,  7,  0.1872, 0.6188,  60,  20),
    (7,  8,  1.7114, 1.2351, 200, 100),
    (8,  9,  1.0300, 0.7400, 200, 100),
    (9, 10,  1.0400, 0.7400,  60,  20),
    (10,11,  0.1966, 0.0650,  60,  20),
    (11,12,  0.3744, 0.1238,  45,  30),
    (12,13,  1.4680, 1.1550,  60,  35),
    (13,14,  0.5416, 0.7129,  60,  35),
    (14,15,  0.5910, 0.5260, 120,  80),
    (15,16,  0.7463, 0.5450,  60,  10),
    (16,17,  1.2890, 1.7210,  60,  20),
    (17,18,  0.7320, 0.5740,  60,  20),
    (2, 19,  0.1640, 0.1565,  90,  40),
    (19,20,  1.5042, 1.3554,  90,  40),
    (20,21,  0.4095, 0.4784,  90,  40),
    (21,22,  0.7089, 0.9373,  90,  40),
    (3, 23,  0.4512, 0.3083,  90,  40),
    (23,24,  0.8980, 0.7091,  90,  50),
    (24,25,  0.8960, 0.7011, 420, 200),
    (6, 26,  0.2030, 0.1034, 420, 200),
    (26,27,  0.2842, 0.1447,  60,  25),
    (27,28,  1.0590, 0.9337,  60,  25),
    (28,29,  0.8042, 0.7006,  60,  20),
    (29,30,  0.5075, 0.2585, 120,  70),
    (30,31,  0.9744, 0.9630, 200, 600),
    (31,32,  0.3105, 0.3619, 150,  70),
    (32,33,  0.3410, 0.5302, 210, 100),
]

N       = 33
BASE_KV = 12.66
I_RATING = 400   # A

# Base nodal loads (0-indexed)
P0 = np.zeros(N)
Q0 = np.zeros(N)
for fb, tb, R, X, P, Q in BRANCH_DATA:
    P0[tb - 1] = P
    Q0[tb - 1] = Q


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pp(p_kw: np.ndarray, q_kvar: np.ndarray) -> dict:
    """Run pandapower BFS and return result dict with numpy arrays."""
    import pandapower as pp
    net = pp.create_empty_network(sn_mva=100)
    for i in range(N):
        pp.create_bus(net, vn_kv=BASE_KV, name=f"Bus {i+1}")
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)
    for idx, (fb, tb, R, X, P, Q) in enumerate(BRANCH_DATA):
        pp.create_line_from_parameters(
            net, from_bus=fb-1, to_bus=tb-1, length_km=1.0,
            r_ohm_per_km=R, x_ohm_per_km=X, c_nf_per_km=0,
            max_i_ka=I_RATING/1000, name=f"Line {idx}")
    for i in range(1, N):
        pp.create_load(net, bus=i, p_mw=p_kw[i]/1000, q_mvar=q_kvar[i]/1000)
    pp.runpp(net, algorithm="bfsw", max_iteration=200, tolerance_mva=1e-9)
    return dict(
        V_pu   = net.res_bus.vm_pu.values.copy(),
        V_ang  = net.res_bus.va_degree.values.copy(),
        Im     = (net.res_line.i_ka.values * 1000).copy(),
        load   = net.res_line.loading_percent.values.copy(),
        Pl     = (net.res_line.pl_mw.values * 1000).copy(),
        Ql     = (net.res_line.ql_mvar.values * 1000).copy(),
        totPl  = float(net.res_line.pl_mw.values.sum() * 1000),
        totQl  = float(net.res_line.ql_mvar.values.sum() * 1000),
    )


def _norm01(x: np.ndarray) -> np.ndarray:
    r = x - x.min()
    d = r.max() + 1e-12
    return r / d


def _build_graph() -> tuple[nx.DiGraph, np.ndarray]:
    """Return (DiGraph, cumulative_impedance_per_bus)."""
    G = nx.DiGraph()
    for fb, tb, R, X, P, Q in BRANCH_DATA:
        Z = np.sqrt(R**2 + X**2)
        G.add_edge(fb, tb, R=R, X=X, Z=Z)

    # Cumulative impedance from bus 1 (root path sum)
    cumZ = np.zeros(N)
    for fb, tb, R, X, *_ in BRANCH_DATA:
        cumZ[tb-1] = cumZ[fb-1] + np.sqrt(R**2 + X**2)
    return G, cumZ


def _serialise(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BASE POWER FLOW
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s1_base_powerflow(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 1 — IEEE 33-bus base-case BFS power flow.

    Input JSON fields (all optional — defaults to standard IEEE 33-bus loads):
      p_load_kw   : list[float] len=33, per-bus active load in kW
      q_load_kvar : list[float] len=33, per-bus reactive load in kVAR

    Returns per-bus voltages, branch loadings, losses, and summary metrics.
    """
    p = np.array(p_load_kw) if p_load_kw else P0.copy()
    q = np.array(q_load_kvar) if q_load_kvar else Q0.copy()

    t0 = time.time()
    res = _run_pp(p, q)
    elapsed = round(time.time() - t0, 4)

    V = res["V_pu"]
    VSI = []
    for idx, (fb, tb, R, X, *_) in enumerate(BRANCH_DATA):
        vf = V[fb-1]
        pl = res["Pl"][idx] / 1000
        ql = res["Ql"][idx] / 1000
        vsi = 4*(R*pl + X*ql)**2 / max(vf**4, 1e-9)
        VSI.append(round(float(vsi), 8))

    branch_labels = [f"{fb}-{tb}" for fb, tb, *_ in BRANCH_DATA]

    return _serialise({
        "section": 1,
        "title": "Base Case Power Flow",
        "elapsed_s": elapsed,
        "buses": list(range(1, N+1)),
        "branch_labels": branch_labels,
        "V_pu": res["V_pu"],
        "V_ang_deg": res["V_ang"],
        "I_amps": res["Im"],
        "loading_pct": res["load"],
        "Pl_kw": res["Pl"],
        "Ql_kvar": res["Ql"],
        "VSI": VSI,
        "summary": {
            "total_P_load_kw": float(p.sum()),
            "total_Q_load_kvar": float(q.sum()),
            "total_Pl_kw": res["totPl"],
            "total_Ql_kvar": res["totQl"],
            "loss_pct": round(res["totPl"] / max(p.sum(), 1) * 100, 3),
            "min_V_pu": float(V.min()),
            "min_V_bus": int(V.argmin()) + 1,
            "max_V_pu": float(V.max()),
            "buses_below_095": int((V < 0.95).sum()),
            "buses_below_090": int((V < 0.90).sum()),
            "max_loading_pct": float(res["load"].max()),
            "max_loading_branch": branch_labels[int(res["load"].argmax())],
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ELECTRICAL DISTANCE
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s2_electrical_distance(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 2 — Electrical distance analysis.

    Computes:
      - Cumulative path impedance from substation to every bus (cumZ)
      - Full 33×33 electrical distance matrix (|Z_ij| via BFS path sum)
      - Feeder depth per bus (hop count from bus 1)
      - Correlation r(|Z|, V_pu)

    Input JSON fields (optional):
      p_load_kw, q_load_kvar : same as s1_base_powerflow
    """
    p = np.array(p_load_kw) if p_load_kw else P0.copy()
    q = np.array(q_load_kvar) if q_load_kvar else Q0.copy()
    res = _run_pp(p, q)
    G, cumZ = _build_graph()

    # Feeder depth (BFS hop count)
    depth = dict(nx.single_source_shortest_path_length(G.to_undirected(), 1))
    depth_arr = np.array([depth.get(b, 0) for b in range(1, N+1)], dtype=float)

    # Full distance matrix (sum of |Z| along shortest path)
    G_ug = G.to_undirected()
    for u, v, d in G_ug.edges(data=True):
        d["weight"] = d["Z"]
    dist_matrix = np.zeros((N, N))
    for i in range(1, N+1):
        lengths = nx.single_source_dijkstra_path_length(G_ug, i, weight="weight")
        for j in range(1, N+1):
            dist_matrix[i-1, j-1] = lengths.get(j, 0.0)

    V = res["V_pu"]
    r_ZV = float(np.corrcoef(cumZ, V)[0, 1])

    return _serialise({
        "section": 2,
        "title": "Electrical Distance Analysis",
        "buses": list(range(1, N+1)),
        "cumulative_Z_ohm": cumZ,           # path impedance from bus 1
        "feeder_depth_hops": depth_arr,
        "elec_dist_matrix": dist_matrix,    # 33×33
        "V_pu": V,
        "summary": {
            "r_Z_Vpu": round(r_ZV, 4),
            "max_elec_dist_bus": int(cumZ.argmax()) + 1,
            "max_elec_dist_ohm": round(float(cumZ.max()), 4),
            "max_depth_bus": int(depth_arr.argmax()) + 1,
            "max_depth_hops": int(depth_arr.max()),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ACTIVE POWER DISTURBANCE PROPAGATION
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s3_disturbance_propagation(
    perturb_pct: list[float] | None = None,
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 3 — Active power disturbance propagation.

    For each non-slack bus i and each perturbation level, increases P_i by
    the given percentage and records |ΔV_j| at every bus j → 33×33 influence matrix.

    Input JSON fields:
      perturb_pct  : list of perturbation fractions, e.g. [0.05, 0.10, 0.20]
                     (default [0.05, 0.10, 0.20])
      p_load_kw, q_load_kvar : base loads (optional)

    Returns:
      influence_matrices : dict keyed by pct string → 33×33 matrix
      BII               : Bus Influence Index per bus (row sum of M_20)
      VFI               : Voltage Fragility Index per bus (col sum of M_20)
      propagation_profiles : |ΔV| profile for key buses [2,8,18,24,30]
    """
    perturbs = perturb_pct if perturb_pct else [0.05, 0.10, 0.20]
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()

    t0 = time.time()
    influ = {}
    for pct in perturbs:
        M = np.zeros((N, N))
        for i in range(1, N):
            Pi = p_base.copy()
            Pi[i] *= (1 + pct)
            pf = _run_pp(Pi, q_base.copy())
            M[i, :] = np.abs(pf["V_pu"] - base_V)
        influ[str(pct)] = M

    elapsed = round(time.time() - t0, 3)

    M20 = influ.get("0.2", influ[str(perturbs[-1])])
    BII = M20.sum(axis=1)
    VFI = M20.sum(axis=0)

    # Propagation profiles for key buses
    key_buses = [2, 8, 18, 24, 30]
    profiles = {f"bus_{b}": M20[b-1].tolist() for b in key_buses if b <= N}

    return _serialise({
        "section": 3,
        "title": "Active Power Disturbance Propagation",
        "elapsed_s": elapsed,
        "buses": list(range(1, N+1)),
        "perturb_levels": perturbs,
        "influence_matrices": {k: v for k, v in influ.items()},
        "BII": BII,
        "VFI": VFI,
        "propagation_profiles_20pct": profiles,
        "summary": {
            "max_influence_pct": round(float(M20.max() * 100), 4),
            "most_influential_bus_BII": int(BII.argmax()) + 1,
            "most_fragile_bus_VFI": int(VFI.argmax()) + 1,
            "top5_BII_buses": (np.argsort(BII)[::-1][:5] + 1).tolist(),
            "top5_VFI_buses": (np.argsort(VFI)[::-1][:5] + 1).tolist(),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — REACTIVE POWER SENSITIVITY (dV/dQ)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s4_reactive_sensitivity(
    delta_q_kvar: float = 50.0,
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 4 — Reactive power sensitivity (dV/dQ matrix).

    Perturbs Q at each bus by delta_q_kvar and measures ΔV at all buses.
    Diagonal = self-reactive sensitivity (primary weak-bus indicator).

    Input JSON fields:
      delta_q_kvar : perturbation size in kVAR (default 50)
      p_load_kw, q_load_kvar : base loads (optional)

    Returns:
      dVdQ_matrix  : 33×33 sensitivity matrix (pu/kVAR)
      self_sens    : diagonal −dV/dQ per bus (higher = weaker)
      col_sum_sens : total reactive sensitivity exposure per bus
      weak_bus_ranking : top-15 weakest buses by self_sens
    """
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()

    t0 = time.time()
    dVdQ = np.zeros((N, N))
    for i in range(1, N):
        Qi = q_base.copy()
        Qi[i] += delta_q_kvar
        pf = _run_pp(p_base.copy(), Qi)
        dVdQ[i, :] = (pf["V_pu"] - base_V) / delta_q_kvar

    elapsed = round(time.time() - t0, 3)

    self_sens = -np.diag(dVdQ)
    col_sum   = np.abs(dVdQ).sum(axis=0)
    rank_idx  = np.argsort(self_sens)[::-1][:15]

    return _serialise({
        "section": 4,
        "title": "Reactive Power Sensitivity (dV/dQ)",
        "elapsed_s": elapsed,
        "buses": list(range(1, N+1)),
        "delta_q_kvar": delta_q_kvar,
        "dVdQ_matrix": dVdQ,            # 33×33 (pu/kVAR)
        "self_sensitivity": self_sens,   # diagonal (pu/kVAR)
        "col_sum_sensitivity": col_sum,
        "weak_bus_ranking": {
            "bus_numbers": (rank_idx + 1).tolist(),
            "self_sens_values": self_sens[rank_idx].tolist(),
        },
        "summary": {
            "most_sensitive_bus": int(self_sens.argmax()) + 1,
            "max_self_sens_pu_per_kvar": round(float(self_sens.max()), 6),
            "top5_weak_buses": (np.argsort(self_sens)[::-1][:5] + 1).tolist(),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FEEDER ZONE VULNERABILITY
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s5_feeder_zone_vulnerability(
    near_slack: list[int] | None = None,
    mid_feeder: list[int] | None = None,
    far_end: list[int] | None = None,
    perturb_pct: float = 0.20,
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 5 — Feeder zone vulnerability analysis.

    Classifies buses into zones and compares disturbance propagation reach,
    peak amplification, and feeder-end voltage degradation.

    Input JSON fields:
      near_slack   : list of bus numbers near substation (default [2,3,4,19])
      mid_feeder   : list of mid-feeder buses (default [7,8,11,26,27])
      far_end      : list of feeder-end buses (default [17,18,22,25,33])
      perturb_pct  : active power perturbation fraction (default 0.20)
      p_load_kw, q_load_kvar : base loads (optional)
    """
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()
    ns = near_slack or [2, 3, 4, 19]
    mf = mid_feeder or [7, 8, 11, 26, 27]
    fe = far_end    or [17, 18, 22, 25, 33]

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()
    _, cumZ  = _build_graph()

    def zone_profiles(bus_list):
        out = {}
        for b in bus_list:
            Pi = p_base.copy()
            Pi[b-1] *= (1 + perturb_pct)
            pf = _run_pp(Pi, q_base.copy())
            out[b] = np.abs(pf["V_pu"] - base_V) * 100
        return out

    zones = {
        "near_slack": {"buses": ns, "profiles": zone_profiles(ns)},
        "mid_feeder": {"buses": mf, "profiles": zone_profiles(mf)},
        "far_end":    {"buses": fe, "profiles": zone_profiles(fe)},
    }

    # Feeder-end voltage degradation: perturb each main-feeder bus, record V@Bus18
    main_feeder = list(range(1, 19))
    v18_degradation = {}
    for pct in [0.05, 0.10, 0.20]:
        v18_list = []
        for b in main_feeder:
            Pi = p_base.copy(); Pi[b-1] *= (1 + pct)
            pf = _run_pp(Pi, q_base.copy())
            v18_list.append(float(pf["V_pu"][17]))
        v18_degradation[str(pct)] = v18_list

    # Attenuation vs electrical distance for key buses
    attenuation = {}
    for b in [2, 18, 25]:
        Pi = p_base.copy(); Pi[b-1] *= 1.20
        pf = _run_pp(Pi, q_base.copy())
        dV = np.abs(pf["V_pu"] - base_V) * 100
        attenuation[f"bus_{b}"] = {
            "cumZ": cumZ.tolist(),
            "dV_pct": dV.tolist(),
        }

    # Zone means
    zone_means = {}
    for zname, zdata in zones.items():
        profs = list(zdata["profiles"].values())
        zone_means[zname] = np.mean(profs, axis=0).tolist() if profs else []

    return _serialise({
        "section": 5,
        "title": "Feeder Zone Vulnerability Analysis",
        "buses": list(range(1, N+1)),
        "main_feeder_buses": main_feeder,
        "perturb_pct": perturb_pct,
        "zone_buses": {"near_slack": ns, "mid_feeder": mf, "far_end": fe},
        "zone_mean_dV_pct": zone_means,
        "zone_peak_amplification": {
            zname: {b: float(prof.max()) for b, prof in zdata["profiles"].items()}
            for zname, zdata in zones.items()
        },
        "v18_degradation_by_pct": v18_degradation,
        "attenuation_vs_distance": attenuation,
        "base_V18_pu": float(base_V[17]),
        "summary": {
            "highest_peak_zone": max(
                zones,
                key=lambda z: max(v.max() for v in zones[z]["profiles"].values()) if zones[z]["profiles"] else 0
            ),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NOVEL INFLUENCE INDICES (BII, VFI, DPF, ECI, Instability)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s6_influence_indices(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
    eci_weights: dict | None = None,
) -> dict:
    """
    Section 6 — Novel electrical influence indices.

    Computes BII, VFI, DPF, ECI, and Instability score per bus.

    Formulas:
      BII[i]        = Σ_j |ΔV_j| caused by +20% ΔP at bus i   (row sum M20)
      VFI[j]        = Σ_i |ΔV_j| received from all buses       (col sum M20)
      DPF[i]        = off-diagonal sum / self-influence         (propagation reach)
      ECI[i]        = w_cc·CC + w_z·inv(Z) + w_bii·BII + w_vfi·VFI
      Instability[i]= norm(VFI + self_dVdQ + (1 - V_pu))

    Input JSON fields:
      p_load_kw, q_load_kvar : base loads (optional)
      eci_weights : dict with keys cc, z, bii, vfi (default 0.30/0.30/0.25/0.15)
    """
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()
    w = eci_weights or {"cc": 0.30, "z": 0.30, "bii": 0.25, "vfi": 0.15}

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()

    # Influence matrix M20
    M20 = np.zeros((N, N))
    for i in range(1, N):
        Pi = p_base.copy(); Pi[i] *= 1.20
        pf = _run_pp(Pi, q_base.copy())
        M20[i, :] = np.abs(pf["V_pu"] - base_V)

    BII = M20.sum(axis=1)
    VFI = M20.sum(axis=0)
    DPF = np.array([(M20[i, :].sum() - M20[i, i]) / (M20[i, i] + 1e-12) for i in range(N)])

    # Graph centrality
    G, cumZ = _build_graph()
    G_ug = G.to_undirected()
    cc   = nx.closeness_centrality(G_ug)
    cc_arr = np.array([cc[b] for b in range(1, N+1)])
    inv_Z  = _norm01(1.0 / (cumZ + 0.01))

    ECI = (w["cc"]  * _norm01(cc_arr) +
           w["z"]   * inv_Z +
           w["bii"] * _norm01(BII) +
           w["vfi"] * _norm01(VFI))

    # Self reactive sensitivity for instability
    dVdQ_diag = np.zeros(N)
    for i in range(1, N):
        Qi = q_base.copy(); Qi[i] += 50
        pf = _run_pp(p_base.copy(), Qi)
        dVdQ_diag[i] = -(pf["V_pu"][i] - base_V[i]) / 50

    instability = _norm01(_norm01(VFI) + _norm01(dVdQ_diag) + _norm01(1 - base_V))

    # Top 10 table
    top10_idx = np.argsort(ECI)[::-1][:10]

    return _serialise({
        "section": 6,
        "title": "Novel Electrical Influence Indices",
        "buses": list(range(1, N+1)),
        "BII": BII,
        "VFI": VFI,
        "DPF": DPF,
        "ECI": ECI,
        "instability": instability,
        "self_dVdQ": dVdQ_diag,
        "eci_weights": w,
        "top10_by_ECI": {
            "bus_numbers": (top10_idx + 1).tolist(),
            "ECI":         ECI[top10_idx].tolist(),
            "BII":         BII[top10_idx].tolist(),
            "VFI":         VFI[top10_idx].tolist(),
            "instability": instability[top10_idx].tolist(),
            "V_pu":        base_V[top10_idx].tolist(),
        },
        "summary": {
            "top5_BII": (np.argsort(BII)[::-1][:5] + 1).tolist(),
            "top5_VFI": (np.argsort(VFI)[::-1][:5] + 1).tolist(),
            "top5_ECI": (np.argsort(ECI)[::-1][:5] + 1).tolist(),
            "top5_instability": (np.argsort(instability)[::-1][:5] + 1).tolist(),
            "most_propagating_bus_DPF": int(np.argsort(DPF)[::-1][0]) + 1,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ADVANCED VISUALIZATION DATA  (sections 7 & 8 removed)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s7_graph_analysis(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 7 — Graph-theoretic electrical analysis.

    Computes:
      - Betweenness centrality (weight = 1/|Z|)
      - Closeness centrality
      - Degree centrality
      - Laplacian spectrum (all eigenvalues)
      - Fiedler value λ₂ (algebraic connectivity)
      - Correlations: r(betweenness, BII), r(closeness, V_pu)

    Input JSON fields:
      p_load_kw, q_load_kvar : base loads (optional)
    """
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()

    G, cumZ = _build_graph()
    G_ug = G.to_undirected()
    for u, v, d in G_ug.edges(data=True):
        d["weight"] = 1.0 / (d["Z"] + 1e-6)

    bc  = nx.betweenness_centrality(G_ug, weight="weight", normalized=True)
    cc  = nx.closeness_centrality(G_ug)
    dc  = nx.degree_centrality(G_ug)

    bc_arr = np.array([bc[b] for b in range(1, N+1)])
    cc_arr = np.array([cc[b] for b in range(1, N+1)])
    dc_arr = np.array([dc[b] for b in range(1, N+1)])

    # Laplacian spectrum
    L = nx.laplacian_matrix(G_ug, weight="weight").toarray()
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))
    fiedler = float(eigvals[1])

    # BII for correlation
    M20 = np.zeros((N, N))
    base_Vcopy = base_V.copy()
    for i in range(1, N):
        Pi = p_base.copy(); Pi[i] *= 1.20
        pf = _run_pp(Pi, q_base.copy())
        M20[i, :] = np.abs(pf["V_pu"] - base_Vcopy)
    BII = M20.sum(axis=1)
    ECI_proxy = 0.30*_norm01(cc_arr) + 0.30*_norm01(1/(cumZ+0.01)) + \
                0.25*_norm01(BII) + 0.15*_norm01(M20.sum(axis=0))

    r_bc_BII = float(np.corrcoef(bc_arr, BII)[0, 1])
    r_cc_V   = float(np.corrcoef(cc_arr, base_V)[0, 1])
    r_bc_ECI = float(np.corrcoef(bc_arr, ECI_proxy)[0, 1])

    return _serialise({
        "section": 7,
        "title": "Graph-Theoretic Electrical Analysis",
        "buses": list(range(1, N+1)),
        "betweenness_centrality": bc_arr,
        "closeness_centrality": cc_arr,
        "degree_centrality": dc_arr,
        "laplacian_eigenvalues": eigvals,
        "fiedler_value_lambda2": round(fiedler, 6),
        "summary": {
            "fiedler_value": round(fiedler, 6),
            "r_betweenness_BII": round(r_bc_BII, 4),
            "r_closeness_Vpu": round(r_cc_V, 4),
            "r_betweenness_ECI": round(r_bc_ECI, 4),
            "highest_betweenness_bus": int(bc_arr.argmax()) + 1,
            "highest_closeness_bus": int(cc_arr.argmax()) + 1,
            "interpretation": (
                f"λ₂={fiedler:.5f} — "
                + ("fragile topology" if fiedler < 0.02 else "moderate connectivity")
            ),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GNN DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s8_gnn_dataset(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
    instability_threshold: float = 0.5,
) -> dict:
    """
    Section 8 — GNN node and edge feature matrix generation.

    Node features (33×20): V_pu, V_ang, P_load, Q_load, elec_dist,
      feeder_depth, VFI, BII, DPF, ECI, betweenness, closeness, degree,
      self_dVdQ, instability, label_V_drop, label_instab, label_prop, label_weak

    Edge features (32×9): R, X, Z_mag, loading_pct, Pl_kw, Ql_kvar,
      from_bus, to_bus, VSI

    Input JSON fields:
      p_load_kw, q_load_kvar : base loads (optional)
      instability_threshold  : threshold for label_weak (default 0.5)
    """
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()

    G, cumZ = _build_graph()
    G_ug = G.to_undirected()
    depth_d = nx.single_source_shortest_path_length(G_ug, 1)
    depth_arr = np.array([depth_d.get(b, 0) for b in range(1, N+1)], dtype=float)

    # Influence matrices
    M20 = np.zeros((N, N)); dVdQ_diag = np.zeros(N)
    for i in range(1, N):
        Pi = p_base.copy(); Pi[i] *= 1.20
        pf = _run_pp(Pi, q_base.copy()); M20[i, :] = np.abs(pf["V_pu"] - base_V)
        Qi = q_base.copy(); Qi[i] += 50
        pf2 = _run_pp(p_base.copy(), Qi); dVdQ_diag[i] = -(pf2["V_pu"][i] - base_V[i]) / 50

    BII = M20.sum(axis=1); VFI = M20.sum(axis=0)
    DPF = np.array([(M20[i,:].sum()-M20[i,i])/(M20[i,i]+1e-12) for i in range(N)])
    cc_arr = np.array([nx.closeness_centrality(G_ug)[b] for b in range(1, N+1)])
    for u, v, d in G_ug.edges(data=True): d["weight"] = 1/(d["Z"]+1e-6)
    bc_arr = np.array([nx.betweenness_centrality(G_ug, weight="weight")[b] for b in range(1, N+1)])
    dc_arr = np.array([nx.degree_centrality(G_ug)[b] for b in range(1, N+1)])
    inv_Z  = _norm01(1.0/(cumZ+0.01))
    ECI    = 0.30*_norm01(cc_arr)+0.30*inv_Z+0.25*_norm01(BII)+0.15*_norm01(VFI)
    instability = _norm01(_norm01(VFI)+_norm01(dVdQ_diag)+_norm01(1-base_V))

    label_V_drop  = _norm01(1 - base_V)
    label_instab  = instability
    label_prop    = _norm01(DPF)
    label_weak    = (instability > instability_threshold).astype(int)

    node_features = {
        "bus":          list(range(1, N+1)),
        "V_pu":         base_V.tolist(),
        "V_ang":        base_res["V_ang"].tolist(),
        "P_load_kW":    p_base.tolist(),
        "Q_load_kVAR":  q_base.tolist(),
        "elec_dist_Z":  cumZ.tolist(),
        "feeder_depth": depth_arr.tolist(),
        "VFI":          VFI.tolist(),
        "BII":          BII.tolist(),
        "DPF":          DPF.tolist(),
        "ECI":          ECI.tolist(),
        "betweenness":  bc_arr.tolist(),
        "closeness":    cc_arr.tolist(),
        "degree":       dc_arr.tolist(),
        "self_dVdQ":    dVdQ_diag.tolist(),
        "instability":  instability.tolist(),
        "label_V_drop": label_V_drop.tolist(),
        "label_instab": label_instab.tolist(),
        "label_prop":   label_prop.tolist(),
        "label_weak":   label_weak.tolist(),
    }

    # VSI per branch
    VSI_br = []
    for idx, (fb, tb, R, X, *_) in enumerate(BRANCH_DATA):
        vf = base_V[fb-1]
        pl = base_res["Pl"][idx]/1000; ql = base_res["Ql"][idx]/1000
        VSI_br.append(float(4*(R*pl+X*ql)**2 / max(vf**4, 1e-9)))

    edge_features = {
        "from_bus":    [fb for fb,tb,*_ in BRANCH_DATA],
        "to_bus":      [tb for fb,tb,*_ in BRANCH_DATA],
        "R_ohm":       [R  for _,_,R,*_ in BRANCH_DATA],
        "X_ohm":       [X  for _,_,_,X,*_ in BRANCH_DATA],
        "Z_mag":       [round(float(np.sqrt(R**2+X**2)),4) for _,_,R,X,*_ in BRANCH_DATA],
        "loading_pct": base_res["load"].tolist(),
        "Pl_kw":       base_res["Pl"].tolist(),
        "Ql_kvar":     base_res["Ql"].tolist(),
        "VSI":         VSI_br,
    }

    return _serialise({
        "section": 8,
        "title": "GNN Dataset Generation",
        "n_nodes": N,
        "n_edges": len(BRANCH_DATA),
        "node_feature_columns": list(node_features.keys()),
        "edge_feature_columns": list(edge_features.keys()),
        "node_features": node_features,
        "edge_features": edge_features,
        "summary": {
            "weak_bus_count": int(label_weak.sum()),
            "weak_buses": (np.where(label_weak)[0]+1).tolist(),
            "instability_threshold": instability_threshold,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ADVANCED VISUALIZATION DATA
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s9_advanced_viz_data(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 9 — Advanced composite visualization data.

    Returns all data needed for the multi-panel advanced dashboard:
      - Composite risk score per bus (VFI + self_sens + V_drop normalised)
      - Multi-index radar data (BII, VFI, ECI, instability, self_sens)
      - Spectral data (Laplacian eigenvalues)
      - Mean voltage impact per bus across all perturbation levels
      - Top-5 multi-index profile (normalised ranks)

    Input JSON fields:
      p_load_kw, q_load_kvar : base loads (optional)
    """
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()

    # Influence matrices for 3 perturbation levels
    PERTURB = [0.05, 0.10, 0.20]
    influ = {}
    for pct in PERTURB:
        M = np.zeros((N, N))
        for i in range(1, N):
            Pi = p_base.copy(); Pi[i] *= (1+pct)
            pf = _run_pp(Pi, q_base.copy()); M[i,:] = np.abs(pf["V_pu"] - base_V)
        influ[pct] = M

    M20 = influ[0.20]
    BII = M20.sum(axis=1); VFI = M20.sum(axis=0)
    DPF = np.array([(M20[i,:].sum()-M20[i,i])/(M20[i,i]+1e-12) for i in range(N)])

    dVdQ_diag = np.zeros(N)
    for i in range(1, N):
        Qi = q_base.copy(); Qi[i] += 50
        pf = _run_pp(p_base.copy(), Qi); dVdQ_diag[i] = -(pf["V_pu"][i]-base_V[i])/50

    G, cumZ = _build_graph()
    G_ug = G.to_undirected()
    cc_arr = np.array([nx.closeness_centrality(G_ug)[b] for b in range(1, N+1)])
    for u,v,d in G_ug.edges(data=True): d["weight"]=1/(d["Z"]+1e-6)
    bc_arr = np.array([nx.betweenness_centrality(G_ug,weight="weight")[b] for b in range(1, N+1)])
    inv_Z  = _norm01(1/(cumZ+0.01))
    ECI    = 0.30*_norm01(cc_arr)+0.30*inv_Z+0.25*_norm01(BII)+0.15*_norm01(VFI)
    instability = _norm01(_norm01(VFI)+_norm01(dVdQ_diag)+_norm01(1-base_V))

    risk   = (_norm01(VFI)+_norm01(dVdQ_diag)+_norm01(1-base_V))/3

    # Mean voltage impact per bus across all perturbation levels
    mean_impact = {str(pct): (influ[pct].mean(axis=0)*100).tolist() for pct in PERTURB}

    # Top-5 multi-index profile (normalized scores)
    top5_idx = np.argsort(ECI)[::-1][:5]
    norm_data = np.column_stack([_norm01(BII), _norm01(VFI), _norm01(DPF), _norm01(ECI), instability])
    top5_profile = {
        f"bus_{top5_idx[r]+1}": norm_data[top5_idx[r]].tolist()
        for r in range(min(5, len(top5_idx)))
    }

    # Laplacian spectrum
    L = nx.laplacian_matrix(G_ug, weight="weight").toarray()
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))

    # Multi-metric rank comparison (top 10 by ECI)
    top10 = np.argsort(ECI)[::-1][:10]
    import pandas as pd
    rank_BII = pd.Series(BII).rank(ascending=False).values
    rank_VFI = pd.Series(VFI).rank(ascending=False).values
    rank_ECI = pd.Series(ECI).rank(ascending=False).values

    return _serialise({
        "section": 9,
        "title": "Advanced Visualization Data",
        "buses": list(range(1, N+1)),
        "composite_risk_score": risk,
        "BII": BII, "VFI": VFI, "DPF": DPF, "ECI": ECI,
        "instability": instability,
        "self_dVdQ": dVdQ_diag,
        "betweenness": bc_arr,
        "mean_voltage_impact_pct": mean_impact,
        "top5_multi_index_profile": {
            "bus_labels": [f"Bus {top5_idx[r]+1}" for r in range(min(5,len(top5_idx)))],
            "metric_labels": ["BII","VFI","DPF","ECI","Instability"],
            "profiles": top5_profile,
        },
        "multi_metric_ranks_top10": {
            "bus_numbers": (top10+1).tolist(),
            "rank_BII": rank_BII[top10].tolist(),
            "rank_VFI": rank_VFI[top10].tolist(),
            "rank_ECI": rank_ECI[top10].tolist(),
        },
        "laplacian_eigenvalues": eigvals,
        "fiedler_value": round(float(eigvals[1]), 6),
        "VSI_per_branch": [
            round(float(4*(R*(base_res["Pl"][i]/1000)+X*(base_res["Ql"][i]/1000))**2
                        / max(base_V[fb-1]**4, 1e-9)), 8)
            for i,(fb,tb,R,X,*_) in enumerate(BRANCH_DATA)
        ],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — AUTOMATED RESEARCH INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def s10_research_insights(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Section 10 — Automated research insights.

    Computes all cross-metric correlations and returns structured findings:
      1. Weakest buses by voltage
      2. Most influential buses (BII)
      3. Most reactive-sensitive buses (dV/dQ)
      4. Highest ECI buses
      5. Algebraic connectivity λ₂
      6. Electrical distance correlations r(|Z|, V_pu) and r(|Z|, BII)
      7. Graph-electrical correlation r(betweenness, ECI)
      8. Worst VSI branches
      9. Feeder-end vulnerability summary
      10. System summary (losses, violations, loading)

    Input JSON fields:
      p_load_kw, q_load_kvar : base loads (optional)
    """
    p_base = np.array(p_load_kw) if p_load_kw else P0.copy()
    q_base = np.array(q_load_kvar) if q_load_kvar else Q0.copy()

    base_res = _run_pp(p_base, q_base)
    base_V   = base_res["V_pu"].copy()

    G, cumZ = _build_graph()
    G_ug = G.to_undirected()
    for u,v,d in G_ug.edges(data=True): d["weight"]=1/(d["Z"]+1e-6)

    cc_arr = np.array([nx.closeness_centrality(G_ug)[b] for b in range(1,N+1)])
    bc_arr = np.array([nx.betweenness_centrality(G_ug,weight="weight")[b] for b in range(1,N+1)])

    M20 = np.zeros((N,N)); dVdQ_diag = np.zeros(N)
    for i in range(1,N):
        Pi=p_base.copy(); Pi[i]*=1.20
        pf=_run_pp(Pi, q_base.copy()); M20[i,:]=np.abs(pf["V_pu"]-base_V)
        Qi=q_base.copy(); Qi[i]+=50
        pf2=_run_pp(p_base.copy(),Qi); dVdQ_diag[i]=-(pf2["V_pu"][i]-base_V[i])/50

    BII=M20.sum(axis=1); VFI=M20.sum(axis=0)
    inv_Z=_norm01(1/(cumZ+0.01))
    ECI=0.30*_norm01(cc_arr)+0.30*inv_Z+0.25*_norm01(BII)+0.15*_norm01(VFI)
    instability=_norm01(_norm01(VFI)+_norm01(dVdQ_diag)+_norm01(1-base_V))

    L = nx.laplacian_matrix(G_ug, weight="weight").toarray()
    fiedler = float(np.sort(np.real(np.linalg.eigvalsh(L)))[1])

    r_ZV   = float(np.corrcoef(cumZ, base_V)[0,1])
    r_ZBII = float(np.corrcoef(cumZ, BII)[0,1])
    r_bcECI= float(np.corrcoef(bc_arr, ECI)[0,1])

    # Worst VSI branches
    VSI_br=[float(4*(R*(base_res["Pl"][i]/1000)+X*(base_res["Ql"][i]/1000))**2
                  /max(base_V[fb-1]**4,1e-9)) for i,(fb,tb,R,X,*_) in enumerate(BRANCH_DATA)]
    worst_vsi = np.argsort(VSI_br)[:3]
    worst_vsi_labels = [f"{BRANCH_DATA[k][0]}-{BRANCH_DATA[k][1]}" for k in worst_vsi]

    feeder_ends = {18: {}, 25: {}, 33: {}}
    for b in feeder_ends:
        feeder_ends[b] = {
            "V_pu": round(float(base_V[b-1]), 4),
            "instability": round(float(instability[b-1]), 4),
        }

    return _serialise({
        "section": 10,
        "title": "Automated Research Insights",
        "findings": {
            "1_weakest_buses":        (np.argsort(base_V)[:5]+1).tolist(),
            "2_most_influential_BII": (np.argsort(BII)[::-1][:5]+1).tolist(),
            "3_most_reactive_sensitive": (np.argsort(dVdQ_diag)[::-1][:5]+1).tolist(),
            "4_highest_ECI":          (np.argsort(ECI)[::-1][:5]+1).tolist(),
            "5_fiedler_lambda2":      round(fiedler, 5),
            "5_topology_fragility":   "fragile" if fiedler < 0.02 else "moderate",
            "6_r_Z_Vpu":             round(r_ZV, 3),
            "6_r_Z_BII":             round(r_ZBII, 3),
            "7_r_betweenness_ECI":   round(r_bcECI, 3),
            "8_worst_VSI_branches":  worst_vsi_labels,
            "9_feeder_end_buses":    feeder_ends,
            "10_system_summary": {
                "P_loss_kw":       round(base_res["totPl"], 2),
                "P_loss_pct":      round(base_res["totPl"]/max(p_base.sum(),1)*100, 2),
                "Q_loss_kvar":     round(base_res["totQl"], 2),
                "buses_below_095": int((base_V < 0.95).sum()),
                "buses_below_090": int((base_V < 0.90).sum()),
                "max_loading_pct": round(float(base_res["load"].max()), 1),
                "min_V_pu":        round(float(base_V.min()), 4),
                "min_V_bus":       int(base_V.argmin()) + 1,
            },
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS — ALL SECTIONS IN ONE CALL
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def run_full_analysis(
    p_load_kw: list[float] | None = None,
    q_load_kvar: list[float] | None = None,
) -> dict:
    """
    Run all 10 analysis sections in a single call.

    Returns a dict with keys s1 through s10, each containing the full
    result of that section. Use this when you need everything at once;
    call individual section tools when you need only one metric.

    Input JSON fields:
      p_load_kw   : list[float] len=33, per-bus active load in kW (optional)
      q_load_kvar : list[float] len=33, per-bus reactive load in kVAR (optional)
    """
    t0 = time.time()
    results = {}
    results["s1"]  = await s1_base_powerflow(p_load_kw, q_load_kvar)
    results["s2"]  = await s2_electrical_distance(p_load_kw, q_load_kvar)
    results["s3"]  = await s3_disturbance_propagation(None, p_load_kw, q_load_kvar)
    results["s4"]  = await s4_reactive_sensitivity(50.0, p_load_kw, q_load_kvar)
    results["s5"]  = await s5_feeder_zone_vulnerability(None,None,None,0.20,p_load_kw,q_load_kvar)
    results["s6"]  = await s6_influence_indices(p_load_kw, q_load_kvar)
    results["s7"]  = await s7_graph_analysis(p_load_kw, q_load_kvar)
    results["s8"]  = await s8_gnn_dataset(p_load_kw, q_load_kvar)
    results["s9"]  = await s9_advanced_viz_data(p_load_kw, q_load_kvar)
    results["s10"] = await s10_research_insights(p_load_kw, q_load_kvar)
    results["total_elapsed_s"] = round(time.time() - t0, 2)
    return results


if __name__ == "__main__":
    print("\n  IEEE 33-Bus Visualizer MCP Server")
    print("  → http://localhost:8004/mcp\n")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8004)