"""
Step 20 (new) — Audit log.

Every query and action the firewall processes is appended here as one JSON
line: what came in, which checkpoint decided what, and the final status. This
is what makes the product auditable — a customer's security team can prove
what was blocked and why. It's also the data source for the admin dashboard.

Deliberately append-only JSONL: cheap, tail-able, survives restarts, and can
be shipped to a real log pipeline (ELK/CloudWatch) later without code change.
"""

import json
import time
from pathlib import Path
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "audit_log.jsonl"
_lock = Lock()


def record(event: dict):
    event = {"ts": time.time(), **event}
    with _lock:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def read_all() -> list:
    if not LOG_PATH.exists():
        return []
    out = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summary() -> dict:
    """Roll the raw log up into the numbers the dashboard shows."""
    events = read_all()
    q = [e for e in events if e.get("kind") == "query"]
    total = len(q)
    by_status = {}
    for e in q:
        by_status[e.get("status", "?")] = by_status.get(e.get("status", "?"), 0) + 1

    blocked_input = by_status.get("blocked_at_input", 0)
    blocked_output = by_status.get("blocked_at_output", 0)
    answered = by_status.get("answered", 0)
    out_of_domain = by_status.get("short_circuited_out_of_domain", 0)
    smalltalk = by_status.get("smalltalk_instant_reply", 0)

    latencies = [e["latency_ms"] for e in q if e.get("latency_ms") is not None]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    return {
        "total_queries": total,
        "answered": answered,
        "blocked_at_input": blocked_input,
        "blocked_at_output": blocked_output,
        "out_of_domain": out_of_domain,
        "smalltalk": smalltalk,
        "threats_blocked": blocked_input + blocked_output,
        "avg_latency_ms": avg_latency,
        "by_status": by_status,
        "recent": list(reversed(q[-15:])),
    }


if __name__ == "__main__":
    record({"kind": "query", "user": "demo", "query": "test", "status": "answered", "latency_ms": 12.3})
    print(summary())
