"""
RAG Store — ChromaDB
====================
Two collections:
  1. dnr_cases     — every solved case stored as embedding
  2. dnr_knowledge — static engineering knowledge base

Uses sentence-transformers locally (no API key needed).
"""

from __future__ import annotations
import json
import time
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = str(Path(__file__).parent / "chroma_db")

_client: chromadb.PersistentClient | None = None
_ef = None


def _get_ef():
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _ef


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _cases_col():
    return get_client().get_or_create_collection(
        name="dnr_cases",
        embedding_function=_get_ef(),
        metadata={"hnsw:space": "cosine"},
    )


def _knowledge_col():
    return get_client().get_or_create_collection(
        name="dnr_knowledge",
        embedding_function=_get_ef(),
        metadata={"hnsw:space": "cosine"},
    )


# ══════════════════════════════════════════════════════════════
# CASE STORE
# ══════════════════════════════════════════════════════════════

def save_case(query_id: str, label: str, scenario_key: str,
              analysis: dict, recommendations: list,
              explanation: str) -> None:
    """
    Store a solved case as a searchable embedding.
    Text = label + scenario + violation summary + recommendations.
    """
    violations = []
    if analysis.get("buses_below_095", 0) > 0:
        violations.append(
            f"{analysis['buses_below_095']} buses below 0.95pu "
            f"(worst bus {analysis.get('min_V_bus')} at {analysis.get('min_V')} pu)"
        )
    if analysis.get("max_loading", 0) > 80:
        violations.append(
            f"branch overload at {analysis.get('max_loading')}%"
        )

    rec_text = "; ".join(
        f"open {r.get('open_switches',[])} close {r.get('close_switches',[])}"
        for r in recommendations[:3]
    ) if recommendations else "no switching actions"

    text = (
        f"Scenario: {label}. "
        f"Key: {scenario_key}. "
        f"Losses: {analysis.get('totPl', 0):.1f} kW. "
        f"Min voltage: {analysis.get('min_V', 0):.4f} pu at bus "
        f"{analysis.get('min_V_bus', '?')}. "
        f"Violations: {'; '.join(violations) if violations else 'none'}. "
        f"Actions: {rec_text}. "
        f"Explanation: {explanation}"
    )

    _cases_col().upsert(
        ids=[query_id],
        documents=[text],
        metadatas=[{
            "query_id":     query_id,
            "label":        label,
            "scenario_key": scenario_key,
            "timestamp":    str(time.time()),
            "min_V":        str(analysis.get("min_V", 0)),
            "total_losses": str(analysis.get("totPl", 0)),
            "explanation":  explanation[:500],
        }],
    )


def retrieve_similar_cases(query_text: str, top_k: int = 5) -> list[dict]:
    """Semantic search over stored cases."""
    col = _cases_col()
    if col.count() == 0:
        return []
    results = col.query(query_texts=[query_text], n_results=min(top_k, col.count()))
    out = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        out.append({
            "query_id":     meta.get("query_id"),
            "label":        meta.get("label"),
            "scenario_key": meta.get("scenario_key"),
            "similarity":   round(1 - dist, 4),
            "min_V":        meta.get("min_V"),
            "total_losses": meta.get("total_losses"),
            "explanation":  meta.get("explanation"),
            "document":     doc,
        })
    return out


def get_all_cases() -> list[dict]:
    col = _cases_col()
    if col.count() == 0:
        return []
    results = col.get(include=["documents", "metadatas"])
    out = []
    for i, doc in enumerate(results["documents"]):
        meta = results["metadatas"][i]
        out.append({"document": doc, **meta})
    return out


# ══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════

SEED_KNOWLEDGE = [
    {
        "id": "voltage_violation",
        "text": (
            "Voltage violation: A bus voltage below 0.95 p.u. indicates insufficient "
            "reactive power support or excessive resistive drop. In radial networks like "
            "IEEE 33-bus, voltage drops accumulate from the substation outward. Buses at "
            "the end of long feeders (bus 18, bus 33) are most vulnerable. "
            "Recommendation: Add capacitor banks, deploy DG, or reconfigure the feeder "
            "to reduce electrical distance."
        ),
        "topic": "voltage",
        "severity": "high",
    },
    {
        "id": "thermal_overload",
        "text": (
            "Thermal overload: Branch loading above 80% risks conductor heating and "
            "accelerated insulation degradation. Branch 1-2 carries all downstream load "
            "and is consistently the most loaded in IEEE 33-bus. "
            "Recommendation: Reconfigure by closing a tie switch to redistribute current, "
            "or upgrade the conductor rating."
        ),
        "topic": "thermal",
        "severity": "high",
    },
    {
        "id": "power_loss",
        "text": (
            "Power losses: Real power losses (I²R) are highest in branches with large "
            "resistance and high current. IEEE 33-bus base loss is ~245 kW (6.73% of "
            "total load). Feeder reconfiguration can reduce losses by 10–30%. "
            "Classical optimal reconfiguration closes tie switches 7-20, 8-14, 11-21, "
            "17-32, 24-28 reducing losses to ~139 kW."
        ),
        "topic": "losses",
        "severity": "medium",
    },
    {
        "id": "dg_injection",
        "text": (
            "DG injection: Injecting real power at a bus raises local voltage and reduces "
            "upstream current flow, decreasing losses. Over-injection can cause reverse "
            "power flow and voltage rise above 1.05 p.u. "
            "Recommendation: Size DG to cover local load plus some upstream load. "
            "Use smart inverter reactive control (Q priority mode) to maintain voltage "
            "within [0.95, 1.05] p.u."
        ),
        "topic": "dg",
        "severity": "low",
    },
    {
        "id": "capacitor_bank",
        "text": (
            "Capacitor bank: Shunt capacitors inject reactive power locally, raising bus "
            "voltage and reducing reactive current flow. Optimal placement targets the "
            "bus with highest dV/dQ sensitivity (bus 18 in IEEE 33-bus). "
            "Standard sizes: 300, 600, 900 kVAR."
        ),
        "topic": "capacitor",
        "severity": "low",
    },
    {
        "id": "bii_index",
        "text": (
            "Bus Influence Index (BII): Row sum of the 33x33 active power influence "
            "matrix M20. BII[i] measures total voltage deviation caused across all buses "
            "when bus i load increases by 20%. High BII buses are disturbance sources. "
            "Buses near branching junctions (bus 2, 3, 6) typically have high BII."
        ),
        "topic": "indices",
        "severity": "medium",
    },
    {
        "id": "vfi_index",
        "text": (
            "Voltage Fragility Index (VFI): Column sum of the influence matrix M20. "
            "VFI[j] measures total voltage perturbation received at bus j from all "
            "possible disturbances. High VFI buses are vulnerable receivers. "
            "Feeder-end buses (bus 18, bus 33) typically have highest VFI."
        ),
        "topic": "indices",
        "severity": "medium",
    },
    {
        "id": "eci_index",
        "text": (
            "Electrical Centrality Index (ECI): Composite index combining closeness "
            "centrality (0.30), inverse electrical distance (0.30), BII (0.25), and "
            "VFI (0.15). High ECI buses are both electrically central and influential. "
            "These buses should be prioritised for protection, sensors, and tie-switch "
            "connection points."
        ),
        "topic": "indices",
        "severity": "medium",
    },
    {
        "id": "fiedler_value",
        "text": (
            "Fiedler value (λ₂): Second smallest eigenvalue of the graph Laplacian. "
            "Measures algebraic connectivity of the network. λ₂ < 0.02 indicates "
            "fragile topology vulnerable to islanding. IEEE 33-bus has low Fiedler "
            "value due to radial structure with few tie switches."
        ),
        "topic": "graph",
        "severity": "medium",
    },
    {
        "id": "zone_vulnerability",
        "text": (
            "Feeder zone vulnerability: Near-slack buses (2,3,4,19) have low BII and "
            "low VFI — disturbances there are absorbed by the substation. Mid-feeder "
            "buses (7,8,11,26,27) have moderate BII. Far-end buses (18,22,25,33) have "
            "highest VFI — most vulnerable to all disturbances. Voltage at bus 18 is "
            "the canonical weak-point indicator for IEEE 33-bus."
        ),
        "topic": "zones",
        "severity": "high",
    },
]


def seed_knowledge_base() -> None:
    """Seed the knowledge base on first run if empty."""
    col = _knowledge_col()
    if col.count() > 0:
        return
    col.upsert(
        ids=[k["id"] for k in SEED_KNOWLEDGE],
        documents=[k["text"] for k in SEED_KNOWLEDGE],
        metadatas=[{"topic": k["topic"], "severity": k["severity"]}
                   for k in SEED_KNOWLEDGE],
    )


def query_knowledge(query_text: str, top_k: int = 3) -> list[dict]:
    """Retrieve relevant knowledge entries for a query."""
    col = _knowledge_col()
    if col.count() == 0:
        seed_knowledge_base()
    results = col.query(query_texts=[query_text],
                        n_results=min(top_k, col.count()))
    out = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        out.append({
            "text":       doc,
            "topic":      meta.get("topic"),
            "severity":   meta.get("severity"),
            "relevance":  round(1 - dist, 4),
        })
    return out


def add_knowledge(text: str, topic: str, severity: str = "medium") -> str:
    """Add a new knowledge entry at runtime."""
    import uuid
    kid = f"custom_{uuid.uuid4().hex[:8]}"
    _knowledge_col().upsert(
        ids=[kid],
        documents=[text],
        metadatas=[{"topic": topic, "severity": severity}],
    )
    return kid