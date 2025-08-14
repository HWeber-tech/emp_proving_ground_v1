#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta

from src.data_foundation.persist.jsonl_writer import write_events_jsonl
from src.data_integration.openbb_integration import fetch_calendar


def parse_args():
    p = argparse.ArgumentParser(description="Fetch macro calendar and write JSONL")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--out", default="data/macro/calendar.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    end = datetime.utcnow().date()
    start = (end - timedelta(days=args.days)).isoformat()
    end_s = end.isoformat()
    events = [e.model_dump() for e in fetch_calendar(start, end_s)]
    write_events_jsonl(events, args.out)
    print(f"Wrote {len(events)} events to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


