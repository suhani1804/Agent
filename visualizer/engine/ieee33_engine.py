"""
IEEE 33-Bus DNR Visualizer — Power Engine
==========================================
Real BFS power flow using pandapower.
Computes before/after metrics for any scenario.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Any

# ── IEEE 33-Bus standard branch data ─────────────────────────────────────────
# (from_bus, to_bus, R[Ω], X[Ω], P_load[kW], Q_load[kVAR])
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

N        = 33
BASE_KV  = 12.66
I_RATING = 400   # A

# Bus positions for topology diagram (hierarchical layout)
BUS_POSITIONS = {
    1:  (0, 0),   2:  (1, 0),   3:  (2, 0),   4:  (3, 0),   5:  (4, 0),
    6:  (5, 0),   7:  (6,-1),   8:  (7,-1),   9:  (8,-1),  10:  (9,-1),
    11: (10,-1),  12: (11,-1),  13: (12,-1),  14: (13,-1),  15: (14,-1),
    16: (15,-1),  17: (16,-1),  18: (17,-1),
    19: (1,-2),   20: (2,-2),   21: (3,-2),   22: (4,-2),
    23: (2,-3),   24: (3,-3),   25: (4,-3),
    26: (5,-4),   27: (6,-4),   28: (7,-4),   29: (8,-4),
    30: (9,-4),   31: (10,-4),  32: (11,-4),  33: (12,-4),
}

# ── RAG Knowledge Base ────────────────────────────────────────────────────────
RAG_KB = {
    "voltage_violation": {
        "title": "Voltage Violation",
        "explanation": "A bus voltage below 0.95 p.u. indicates insufficient reactive power support or excessive resistive drop along the feeder. In radial networks like IEEE 33-bus, voltage drops accumulate from the substation outward — buses at the end of long feeders (e.g., bus 18, bus 33) are most vulnerable.",
        "recommendation": "Add capacitor banks near the weakest bus, reconfigure the feeder to reduce electrical distance, or deploy a distributed generator (DG) at that bus.",
        "severity": "high"
    },
    "thermal_overload": {
        "title": "Thermal Overload",
        "explanation": "A branch loading above 80% risks conductor heating, accelerated insulation degradation, and potential outage. Branch 1-2 carries all downstream load and is consistently the most loaded in IEEE 33-bus.",
        "recommendation": "Reconfigure by closing a tie switch to redistribute current, or upgrade the conductor rating on the critical branch.",
        "severity": "high"
    },
    "power_loss": {
        "title": "Power Loss",
        "explanation": "Real power losses (I²R) are highest in branches with large resistance and high current. The IEEE 33-bus system has a base loss of ~245 kW (6.73% of total load). Feeder reconfiguration can reduce losses by 10–30%.",
        "recommendation": "Loss minimisation via optimal switch placement is a classic DNR objective. Closing tie switch 33-37 (between buses 7 and 20) is known to reduce losses significantly.",
        "severity": "medium"
    },
    "load_increase": {
        "title": "Load Increase Disturbance",
        "explanation": "Adding load at a bus reduces that bus's voltage and increases upstream branch currents. The voltage depression propagates back toward the substation with diminishing magnitude — buses electrically close to the disturbance see larger ΔV.",
        "recommendation": "Monitor the voltage stability index (VSI) on branches feeding the disturbed bus. If VSI approaches 1.0, the bus is near its maximum loadability limit.",
        "severity": "medium"
    },
    "dg_injection": {
        "title": "DG / PV Injection",
        "explanation": "Injecting real power at a bus raises local voltage and reduces upstream current flow, decreasing losses. Reactive injection (e.g., from a smart inverter) additionally supports voltage. Over-injection can cause reverse power flow and voltage rise above 1.05 p.u.",
        "recommendation": "Size DG to cover local load plus some upstream load. Use smart inverter reactive control (Q priority mode) to maintain voltage within [0.95, 1.05] p.u.",
        "severity": "low"
    },
    "bus_outage": {
        "title": "Bus Outage / Load Shedding",
        "explanation": "Disconnecting a load bus removes demand, causing voltage rise on neighbouring buses and reduced branch currents. System losses decrease. This simulates planned maintenance or fault isolation.",
        "recommendation": "After isolation, check whether feeder-end buses experience voltage rise above 1.05 p.u. due to reactive power surplus. Reactive load shedding is sometimes needed.",
        "severity": "low"
    },
    "capacitor_bank": {
        "title": "Capacitor Bank",
        "explanation": "Shunt capacitors inject reactive power locally, raising bus voltage and reducing reactive current flow from the substation. This decreases I²R losses and improves power factor. The effect is strongest at the bus of connection.",
        "recommendation": "Optimal capacitor placement targets the bus with the highest dV/dQ sensitivity (bus 18 in IEEE 33-bus). Standard sizes: 300, 600, 900 kVAR.",
        "severity": "low"
    }
}

# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS = {
    "base": {
        "label": "Base case",
        "description": "Standard IEEE 33-bus loading — 3655 kW, 2260 kVAR total.",
        "changes": {}
    },
    "load_spike_bus18": {
        "label": "Load spike at Bus 18 (+50%)",
        "description": "Bus 18 load increased 50% — simulates afternoon peak at feeder end.",
        "changes": {"bus_load_scale": {18: 1.5}}
    },
    "load_spike_bus25": {
        "label": "Load spike at Bus 25 (+80%)",
        "description": "Bus 25 load increased 80% — heavy industrial load on lateral 3.",
        "changes": {"bus_load_scale": {25: 1.8}}
    },
    "dg_bus18": {
        "label": "DG injection at Bus 18 (500 kW)",
        "description": "500 kW rooftop PV farm injected at bus 18 — weakest feeder end.",
        "changes": {"dg": {18: {"p_mw": 0.5, "q_mvar": 0.0}}}
    },
    "dg_bus33": {
        "label": "DG injection at Bus 33 (400 kW)",
        "description": "400 kW DG at bus 33, far end of feeder zone 3.",
        "changes": {"dg": {33: {"p_mw": 0.4, "q_mvar": 0.0}}}
    },
    "cap_bus18": {
        "label": "Capacitor bank at Bus 18 (600 kVAR)",
        "description": "Shunt capacitor at weakest bus — reactive power compensation.",
        "changes": {"capacitor": {18: 0.6}}
    },
    "cap_bus30": {
        "label": "Capacitor bank at Bus 30 (900 kVAR)",
        "description": "Reactive compensation on feeder zone 3 mid-point.",
        "changes": {"capacitor": {30: 0.9}}
    },
    "outage_bus25": {
        "label": "Bus 25 load outage",
        "description": "Bus 25 de-energised — simulates fault isolation or maintenance.",
        "changes": {"bus_load_scale": {25: 0.0}}
    },
    "heavy_load": {
        "label": "System-wide load +30%",
        "description": "All loads scaled up 30% — simulates extreme summer peak demand.",
        "changes": {"global_scale": 1.3}
    },
    "light_load": {
        "label": "System-wide load -40%",
        "description": "All loads scaled down 40% — simulates night-time minimum demand.",
        "changes": {"global_scale": 0.6}
    },
}


def run_powerflow(changes: dict) -> dict[str, Any]:
    """
    Run pandapower BFS on IEEE 33-bus with the given modifications.
    Returns full result dict.
    """
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=100)
    vn_kv = BASE_KV

    for i in range(N):
        pp.create_bus(net, vn_kv=vn_kv, name=f"Bus {i+1}")

    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Grid")

    # Base loads
    P0 = np.zeros(N)
    Q0 = np.zeros(N)
    for fb, tb, R, X, P, Q in BRANCH_DATA:
        P0[tb - 1] = P
        Q0[tb - 1] = Q

    # Apply global scale
    g_scale = changes.get("global_scale", 1.0)
    P0 *= g_scale
    Q0 *= g_scale

    # Apply per-bus load scale
    bus_scale = changes.get("bus_load_scale", {})
    for bus, scale in bus_scale.items():
        P0[bus - 1] *= scale / g_scale   # relative to already-scaled base
        Q0[bus - 1] *= scale / g_scale

    # Create lines
    for idx, (fb, tb, R, X, P, Q) in enumerate(BRANCH_DATA):
        pp.create_line_from_parameters(
            net,
            from_bus=fb - 1, to_bus=tb - 1,
            length_km=1.0,
            r_ohm_per_km=R, x_ohm_per_km=X,
            c_nf_per_km=0,
            max_i_ka=I_RATING / 1000,
            name=f"Line {idx}"
        )

    # Create loads
    for i in range(1, N):
        pp.create_load(net, bus=i, p_mw=P0[i] / 1000, q_mvar=Q0[i] / 1000)

    # Apply DG
    for bus, dg in changes.get("dg", {}).items():
        pp.create_sgen(net, bus=bus - 1,
                       p_mw=dg["p_mw"], q_mvar=dg["q_mvar"],
                       name=f"DG@Bus{bus}")

    # Apply capacitor banks (as static generators with Q injection)
    for bus, q_mvar in changes.get("capacitor", {}).items():
        pp.create_shunt(net, bus=bus - 1,
                        q_mvar=-q_mvar, p_mw=0.0,
                        name=f"Cap@Bus{bus}")

    t0 = time.time()
    try:
        pp.runpp(net, algorithm="bfsw", max_iteration=200, tolerance_mva=1e-9)
        converged = True
    except Exception:
        try:
            pp.runpp(net, algorithm="nr", max_iteration=100, tolerance_mva=1e-6)
            converged = True
        except Exception as e:
            converged = False

    elapsed = time.time() - t0

    if not converged:
        return {"converged": False, "elapsed": elapsed}

    V_pu    = net.res_bus.vm_pu.values.tolist()
    V_ang   = net.res_bus.va_degree.values.tolist()
    Im      = (net.res_line.i_ka.values * 1000).tolist()   # A
    loading = net.res_line.loading_percent.values.tolist()
    Pl      = (net.res_line.pl_mw.values * 1000).tolist()  # kW
    Ql      = (net.res_line.ql_mvar.values * 1000).tolist()

    # VSI per branch: VSI = 4*(R*P+X*Q)^2 / (V_from^2)^2  (simplified)
    vsi = []
    for idx, (fb, tb, R, X, P, Q) in enumerate(BRANCH_DATA):
        vf = V_pu[fb - 1]
        p  = Pl[idx] / 1000
        q  = Ql[idx] / 1000
        vsi_val = 4 * (R * p + X * q) ** 2 / max(vf ** 4, 1e-9)
        vsi.append(round(vsi_val, 6))

    # Branch labels
    branch_labels = [f"{fb}-{tb}" for fb, tb, *_ in BRANCH_DATA]

    return {
        "converged": True,
        "elapsed":   round(elapsed, 4),
        "V_pu":      [round(v, 6) for v in V_pu],
        "V_ang":     [round(a, 4) for a in V_ang],
        "Im":        [round(i, 2) for i in Im],
        "loading":   [round(l, 2) for l in loading],
        "Pl":        [round(p, 4) for p in Pl],
        "Ql":        [round(q, 4) for q in Ql],
        "VSI":       vsi,
        "totPl":     round(sum(Pl), 2),
        "totQl":     round(sum(Ql), 2),
        "min_V":     round(min(V_pu), 4),
        "min_V_bus": int(np.argmin(V_pu)) + 1,
        "max_loading": round(max(loading), 1),
        "buses_below_095": int(sum(1 for v in V_pu if v < 0.95)),
        "branch_labels": branch_labels,
    }


def compute_delta(before: dict, after: dict) -> dict:
    """Compute before→after deltas for all metrics."""
    if not before["converged"] or not after["converged"]:
        return {}

    dV   = [round(a - b, 6) for a, b in zip(after["V_pu"], before["V_pu"])]
    dPl  = [round(a - b, 4) for a, b in zip(after["Pl"],   before["Pl"])]
    dLoad = [round(a - b, 2) for a, b in zip(after["loading"], before["loading"])]

    return {
        "dV":           dV,
        "dPl":          dPl,
        "dLoad":        dLoad,
        "d_totPl":      round(after["totPl"]   - before["totPl"],   2),
        "d_totQl":      round(after["totQl"]   - before["totQl"],   2),
        "d_min_V":      round(after["min_V"]   - before["min_V"],   4),
        "d_max_loading":round(after["max_loading"] - before["max_loading"], 1),
        "d_buses_below":after["buses_below_095"] - before["buses_below_095"],
        "max_dV_bus":   int(np.argmax(np.abs(dV))) + 1,
        "max_dV_mag":   round(max(abs(d) for d in dV), 4),
    }


def rag_explain(scenario_key: str, before: dict, after: dict, delta: dict) -> list[dict]:
    """
    RAG-style explainability: match scenario to knowledge entries,
    return ranked explanations with context from actual results.
    """
    entries = []
    sc = SCENARIOS.get(scenario_key, {})
    changes = sc.get("changes", {})

    # Match knowledge entries to the scenario type
    if "dg" in changes:
        e = dict(RAG_KB["dg_injection"])
        e["context"] = (
            f"DG injection reduced total losses by {-delta.get('d_totPl', 0):.1f} kW. "
            f"Minimum voltage changed by {delta.get('d_min_V', 0):+.4f} p.u. "
            f"({before['min_V_bus']} was the weakest bus before)."
        )
        entries.append(e)

    if "capacitor" in changes:
        e = dict(RAG_KB["capacitor_bank"])
        e["context"] = (
            f"Capacitor injection reduced reactive losses by {-delta.get('d_totQl', 0):.1f} kVAR. "
            f"Voltage at bus {before['min_V_bus']} improved by {delta.get('d_min_V', 0):+.4f} p.u."
        )
        entries.append(e)

    if "global_scale" in changes and changes["global_scale"] > 1:
        e = dict(RAG_KB["load_increase"])
        e["context"] = (
            f"System-wide load increase raised total losses by {delta.get('d_totPl', 0):+.1f} kW. "
            f"Number of buses below 0.95 p.u. changed by {delta.get('d_buses_below', 0):+d}."
        )
        entries.append(e)

    if "bus_load_scale" in changes:
        for bus, scale in changes["bus_load_scale"].items():
            if scale > 1:
                e = dict(RAG_KB["load_increase"])
                e["context"] = (
                    f"Load spike at bus {bus} (×{scale}) caused maximum ΔV of "
                    f"{delta.get('max_dV_mag', 0):.4f} p.u. at bus {delta.get('max_dV_bus', bus)}."
                )
                entries.append(e)
            elif scale == 0:
                e = dict(RAG_KB["bus_outage"])
                e["context"] = (
                    f"Bus {bus} isolation reduced total losses by {-delta.get('d_totPl', 0):.1f} kW. "
                    f"Upstream buses saw voltage rise."
                )
                entries.append(e)

    # Always check for violations in the after state
    if after.get("buses_below_095", 0) > 0:
        e = dict(RAG_KB["voltage_violation"])
        e["context"] = (
            f"{after['buses_below_095']} buses below 0.95 p.u. "
            f"Worst: bus {after['min_V_bus']} at {after['min_V']:.4f} p.u."
        )
        entries.append(e)

    if after.get("max_loading", 0) > 80:
        e = dict(RAG_KB["thermal_overload"])
        e["context"] = f"Maximum branch loading is {after['max_loading']:.1f}% — above the 80% thermal limit."
        entries.append(e)

    if not entries:
        e = dict(RAG_KB["power_loss"])
        e["context"] = (
            f"Total losses: {after['totPl']:.1f} kW ({after['totPl']/3655*100:.1f}% of base load). "
            f"Change from base: {delta.get('d_totPl', 0):+.1f} kW."
        )
        entries.append(e)

    return entries