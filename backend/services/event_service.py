
import json
from pathlib import Path

from backend.config import EVENT_LOG


def read_recent_events(limit: int = 50):
    if not EVENT_LOG.exists():
        return []

    events = []

    with EVENT_LOG.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return events[-limit:][::-1]


def get_event_count():
    if not EVENT_LOG.exists():
        return 0

    with EVENT_LOG.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())
