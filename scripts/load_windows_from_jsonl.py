from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.db import SessionLocal  # noqa: E402
from backend.orm_models import WindowAnalytics  # noqa: E402


def load_windows(
    jsonl_path: Path,
    video_id_override: str | None = None,
    truncate_video: bool = False,
) -> int:
    """Read windows from JSONL and store them in the DB. Returns inserted count."""
    with SessionLocal() as db:
        if truncate_video and video_id_override:
            db.query(WindowAnalytics).filter(WindowAnalytics.video_id == video_id_override).delete(
                synchronize_session=False
            )
            db.commit()

        to_create: List[WindowAnalytics] = []
        for raw in _iter_jsonl(jsonl_path):
            video_id = video_id_override or raw["video_id"]
            window = WindowAnalytics(
                video_id=video_id,
                window_start_sec=_pick(raw, "start_time_sec", "window_start_sec"),
                window_end_sec=_pick(raw, "end_time_sec", "window_end_sec"),
                workers_in_safe=raw["workers_in_safe"],
                workers_in_medium=raw["workers_in_medium"],
                workers_in_danger=raw["workers_in_danger"],
                dangerous_situations_count=raw["dangerous_situations_count"],
                violations_count=raw["violations_count"],
                cleaning_workers_count=raw["cleaning_workers_count"],
                train_present=raw["train_present"],
                train_arrived_in_window=raw["train_arrived_in_window"],
                train_departed_in_window=raw["train_departed_in_window"],
                train_number=raw.get("train_number"),
            )
            to_create.append(window)

        db.add_all(to_create)
        db.commit()
        return len(to_create)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pick(obj: dict, *keys):
    for k in keys:
        if k in obj:
            return obj[k]
    raise KeyError(f"Missing keys {keys} in record: {obj}")


def main():
    parser = argparse.ArgumentParser(
        description="Load window analytics JSONL into the sqlite DB (depot_analytics.db by default)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "logs" / "windows_simple.jsonl",
        help="Path to JSONL with window records.",
    )
    parser.add_argument(
        "--video-id",
        dest="video_id_override",
        default=None,
        help="Override video_id for all records (optional).",
    )
    parser.add_argument(
        "--truncate-video",
        action="store_true",
        help="Delete existing records for the overridden video_id before insert.",
    )
    args = parser.parse_args()

    inserted = load_windows(
        args.path,
        video_id_override=args.video_id_override,
        truncate_video=args.truncate_video,
    )
    print(f"Inserted {inserted} window records from {args.path}")


if __name__ == "__main__":
    main()
