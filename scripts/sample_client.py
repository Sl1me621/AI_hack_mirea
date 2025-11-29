from __future__ import annotations

import requests

BACKEND_URL = "http://127.0.0.1:8000"


def build_windows():
    return [
        {
            "video_id": "session_1",
            "window_start_sec": 0.0,
            "window_end_sec": 10.0,
            "workers_in_safe": 3,
            "workers_in_medium": 1,
            "workers_in_danger": 2,
            "dangerous_situations_count": 1,
            "violations_count": 1,
            "cleaning_workers_count": 1,
            "train_present": True,
            "train_arrived_in_window": True,
            "train_departed_in_window": False,
            "train_number": "ЭП20 076",
        },
        {
            "video_id": "session_1",
            "window_start_sec": 10.0,
            "window_end_sec": 20.0,
            "workers_in_safe": 4,
            "workers_in_medium": 0,
            "workers_in_danger": 1,
            "dangerous_situations_count": 0,
            "violations_count": 0,
            "cleaning_workers_count": 1,
            "train_present": True,
            "train_arrived_in_window": False,
            "train_departed_in_window": False,
            "train_number": "ЭП20 076",
        },
        {
            "video_id": "session_1",
            "window_start_sec": 20.0,
            "window_end_sec": 30.0,
            "workers_in_safe": 2,
            "workers_in_medium": 2,
            "workers_in_danger": 1,
            "dangerous_situations_count": 0,
            "violations_count": 0,
            "cleaning_workers_count": 2,
            "train_present": True,
            "train_arrived_in_window": False,
            "train_departed_in_window": False,
            "train_number": "ЭП20 076",
        },
        {
            "video_id": "session_1",
            "window_start_sec": 30.0,
            "window_end_sec": 40.0,
            "workers_in_safe": 2,
            "workers_in_medium": 1,
            "workers_in_danger": 2,
            "dangerous_situations_count": 2,
            "violations_count": 2,
            "cleaning_workers_count": 1,
            "train_present": True,
            "train_arrived_in_window": False,
            "train_departed_in_window": False,
            "train_number": "ЭП20 076",
        },
        {
            "video_id": "session_1",
            "window_start_sec": 40.0,
            "window_end_sec": 50.0,
            "workers_in_safe": 3,
            "workers_in_medium": 0,
            "workers_in_danger": 1,
            "dangerous_situations_count": 0,
            "violations_count": 0,
            "cleaning_workers_count": 1,
            "train_present": True,
            "train_arrived_in_window": False,
            "train_departed_in_window": True,
            "train_number": "ЭП20 076",
        },
        {
            "video_id": "session_1",
            "window_start_sec": 50.0,
            "window_end_sec": 60.0,
            "workers_in_safe": 4,
            "workers_in_medium": 0,
            "workers_in_danger": 0,
            "dangerous_situations_count": 0,
            "violations_count": 0,
            "cleaning_workers_count": 0,
            "train_present": False,
            "train_arrived_in_window": False,
            "train_departed_in_window": False,
            "train_number": "ЭП20 076",
        },
    ]


def main():
    payload = {"windows": build_windows()}
    resp = requests.post(f"{BACKEND_URL}/api/windows/batch", json=payload, timeout=15)
    if resp.status_code != 200:
        raise SystemExit(f"Failed to send batch: {resp.status_code} {resp.text}")
    print("Windows uploaded:", len(resp.json()))


if __name__ == "__main__":
    main()
