from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def write_events_jsonl(events: List[Dict[str, Any]], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
        return out_path
    except Exception:
        return ""


