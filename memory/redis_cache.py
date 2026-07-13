"""
Redis Cache Layer
=================
Handles:
  - Power flow result cache (TTL 1hr)
  - S3/S4 background job results (TTL 1hr)
  - Active comparison selection per session (TTL 30min)
  - Query metadata sorted set (permanent via no-TTL ZADD)
"""

from __future__ import annotations
import json
import hashlib
import time
import redis

_client = None


def get_redis():
    """
    Returns a Redis client. Falls back to fakeredis (in-memory) if no
    Redis server is running — so the app works without installing Redis.
    """
    global _client
    if _client is not None:
        return _client

    # Try real Redis first
    try:
        import redis as _redis
        r = _redis.Redis(host="localhost", port=6379,
                         decode_responses=True, socket_connect_timeout=1)
        r.ping()   # will raise if server not running
        _client = r
        return _client
    except Exception:
        pass

    # Fall back to fakeredis (in-process, no server needed)
    try:
        import fakeredis
        _client = fakeredis.FakeRedis(decode_responses=True)
        import logging
        logging.getLogger(__name__).warning(
            "Redis not available — using fakeredis (in-memory). "
            "Query history will not persist across restarts. "
            "Install Redis for persistence."
        )
        return _client
    except ImportError:
        raise RuntimeError(
            "Neither Redis nor fakeredis is available. "
            "Run: pip install fakeredis"
        )


# ── Power flow cache ─────────────────────────────────────────────────────────

def _pf_key(scenario_key: str, changes: dict) -> str:
    payload = json.dumps({"s": scenario_key, "c": changes}, sort_keys=True)
    return "pf:" + hashlib.md5(payload.encode()).hexdigest()


def get_pf_cache(scenario_key: str, changes: dict) -> dict | None:
    raw = get_redis().get(_pf_key(scenario_key, changes))
    return json.loads(raw) if raw else None


def set_pf_cache(scenario_key: str, changes: dict, result: dict) -> None:
    get_redis().setex(_pf_key(scenario_key, changes), 3600,
                      json.dumps(result))


# ── Background S3/S4 job results ─────────────────────────────────────────────

def set_background_result(query_id: str, data: dict) -> None:
    get_redis().setex(f"bg:{query_id}", 3600, json.dumps(data))


def get_background_result(query_id: str) -> dict | None:
    raw = get_redis().get(f"bg:{query_id}")
    return json.loads(raw) if raw else None


def set_background_status(query_id: str, status: str) -> None:
    """status: 'running' | 'done' | 'error'"""
    get_redis().setex(f"bgstatus:{query_id}", 3600, status)


def get_background_status(query_id: str) -> str:
    return get_redis().get(f"bgstatus:{query_id}") or "pending"


# ── Comparison selection ──────────────────────────────────────────────────────

def set_comparison(session_id: str, query_ids: list[str]) -> None:
    get_redis().setex(f"compare:{session_id}", 1800,
                      json.dumps(query_ids))


def get_comparison(session_id: str) -> list[str]:
    raw = get_redis().get(f"compare:{session_id}")
    return json.loads(raw) if raw else []


# ── Query metadata store ──────────────────────────────────────────────────────

def save_query_meta(query_id: str, meta: dict) -> None:
    """
    meta keys: label, scenario_key, timestamp, min_V, total_losses,
               max_loading, buses_below_095, thread_id
    """
    r = get_redis()
    r.hset(f"qmeta:{query_id}", mapping={
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        for k, v in meta.items()
    })
    # Sorted set by timestamp for ordered history
    r.zadd("query:history", {query_id: meta["timestamp"]})


def get_query_meta(query_id: str) -> dict:
    raw = get_redis().hgetall(f"qmeta:{query_id}")
    return raw


def get_query_history(limit: int = 20) -> list[dict]:
    """Return recent queries newest first."""
    r = get_redis()
    ids = r.zrevrange("query:history", 0, limit - 1)
    result = []
    for qid in ids:
        meta = get_query_meta(qid)
        if meta:
            meta["query_id"] = qid
            result.append(meta)
    return result


def delete_query(query_id: str) -> None:
    r = get_redis()
    r.delete(f"qmeta:{query_id}")
    r.zrem("query:history", query_id)