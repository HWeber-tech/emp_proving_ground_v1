#!/usr/bin/env python3

import argparse
from src.data_integration.openbb_integration import fetch_yields
from src.data_foundation.persist.jsonl_writer import write_events_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Fetch yields and write JSONL")
    p.add_argument("--curve", default="UST")
    p.add_argument("--out", default="data/macro/yields.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ev = [e.model_dump() for e in fetch_yields(args.curve)]
    write_events_jsonl(ev, args.out)
    print(f"Wrote {len(ev)} yield points to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


