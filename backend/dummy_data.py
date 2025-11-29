from __future__ import annotations

from datetime import datetime

# Predefined stub for the current state metrics
current_state_stub = {
    "workers_in_safe": 8,
    "workers_in_medium": 5,
    "workers_in_danger": 2,
    "dangerous_situations_count": 1,
    "cleaning_workers_count": 2,
}

# Full list of events in chronological order for the MVP
events_stub = [
    {
        "id": 1,
        "timestamp": "2025-11-29T12:00:01",
        "event_type": "TRAIN_ARRIVAL",
        "worker_id": None,
        "zone_name": None,
        "clothing_type_name": None,
        "train_number": "A-102",
        "activity": None,
        "is_violation": None,
    },
    {
        "id": 2,
        "timestamp": "2025-11-29T12:02:15",
        "event_type": "ENTER_ZONE",
        "worker_id": 7,
        "zone_name": "safe",
        "clothing_type_name": "worker_orange",
        "train_number": None,
        "activity": None,
        "is_violation": False,
    },
    {
        "id": 3,
        "timestamp": "2025-11-29T12:05:10",
        "event_type": "ENTER_ZONE",
        "worker_id": 12,
        "zone_name": "medium",
        "clothing_type_name": "worker_white",
        "train_number": None,
        "activity": "cleaning",
        "is_violation": False,
    },
    {
        "id": 4,
        "timestamp": "2025-11-29T12:06:55",
        "event_type": "DANGEROUS_SITUATION",
        "worker_id": 5,
        "zone_name": "danger",
        "clothing_type_name": "worker_color3",
        "train_number": "A-102",
        "activity": None,
        "is_violation": True,
    },
    {
        "id": 5,
        "timestamp": "2025-11-29T12:07:30",
        "event_type": "EXIT_ZONE",
        "worker_id": 5,
        "zone_name": "danger",
        "clothing_type_name": "worker_color3",
        "train_number": None,
        "activity": None,
        "is_violation": False,
    },
    {
        "id": 6,
        "timestamp": "2025-11-29T12:09:40",
        "event_type": "ENTER_ZONE",
        "worker_id": 9,
        "zone_name": "danger",
        "clothing_type_name": "worker_orange",
        "train_number": None,
        "activity": "cleaning",
        "is_violation": False,
    },
    {
        "id": 7,
        "timestamp": "2025-11-29T12:12:05",
        "event_type": "TRAIN_DEPARTURE",
        "worker_id": None,
        "zone_name": None,
        "clothing_type_name": None,
        "train_number": "A-102",
        "activity": None,
        "is_violation": None,
    },
]

# Derived stubs for specific views
dangerous_situations_stub = [
    event for event in events_stub if event.get("event_type") == "DANGEROUS_SITUATION"
]

cleaning_events_stub = [
    event for event in events_stub if event.get("activity") == "cleaning"
]

# Ensure timestamps are ISO strings; validate at import time (simple guard for accidental typos)
for _event in events_stub:
    # Parse and reformat to guarantee consistent ISO-like format without timezone
    parsed = datetime.fromisoformat(_event["timestamp"])
    _event["timestamp"] = parsed.isoformat(timespec="seconds")
