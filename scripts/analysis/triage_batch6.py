#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path

TARGETS = [
    # original targets
    "src/trading/strategy_engine",
    "src/trading/performance/reporting",
    "src/trading/performance/dashboards",
    "src/portfolio",
    "src/performance",
    "src/operational/bus",
    "src/operational/container",
    "src/ecosystem/optimization",
    "src/ecosystem/coordination",
    "src/evolution/engine",
    "src/evolution/selection",
    "src/thinking/analysis",
    "src/thinking/prediction",
    "src/sensory/what/patterns",
    "src/sensory/what/features",
    "src/ui/api",
    "src/ui/cli",
    # added breadth to reach 80–120 files while staying low-coupling
    "src/trading/execution",
    "src/trading/integration",
    "src/trading/monitoring",
    "src/trading/performance/analytics",
    "src/trading/portfolio",
    "src/trading/risk",
    "src/trading/risk_management",
    "src/trading/strategies",
    "src/trading/order_management/execution",
    "src/trading/order_management/monitoring",
    "src/trading/order_management/order_book",
    "src/trading/order_management/smart_routing",
    "src/operational/monitoring",
    "src/ecosystem/evolution",
    "src/ecosystem/evaluation",
    "src/ecosystem/species",
    "src/evolution/ambusher",
    "src/evolution/evaluation",
    "src/evolution/meta",
    "src/evolution/mutation",
    "src/thinking/inference",
    "src/thinking/learning",
    "src/thinking/memory",
    "src/thinking/patterns",
    "src/sensory/organs",
    "src/sensory/enhanced",
    "src/sensory/services",
    "src/sensory/models",
    "src/sensory/utils",
    "src/sensory/when",
    "src/sensory/why",
    "src/validation",
    "src/validation/performance",
    "src/core/performance",
    "src/core/evolution",
    "src/core/strategy/templates",
    "src/integration",
    "src/market_intelligence/dimensions",
    "src/system",
    "src/ui/models",
    "src/governance",
    "src/governance/fitness",
    "src/governance/registry",
    "src/governance/vault",
    "src/data_foundation/config",
    "src/data_foundation/ingest",
    "src/data_foundation/persist",
    "src/data_foundation/replay",
    "src/data_sources",
]

MAX_FILES = 120
LOC_LIMIT = 400
INTERNAL_IMPORT_LIMIT = 1


def compute_metrics(path: Path) -> tuple[int, int]:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return (10**9, 10**9)
    loc = txt.count("\n") + 1
    internal = len(re.findall(r"^(?:\s*)(?:from\s+src\.|import\s+src\.)", txt, flags=re.M))
    return loc, internal


def main() -> None:
    candidates: list[tuple[int, int, str]] = []
    for base in TARGETS:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py") or fn == "__init__.py":
                    continue
                file_path = Path(root) / fn
                loc, internal = compute_metrics(file_path)
                if loc <= LOC_LIMIT and internal <= INTERNAL_IMPORT_LIMIT:
                    candidates.append((loc, internal, str(file_path)))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    selected = candidates[:MAX_FILES]

    out_dir = Path("mypy_snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "selected_batch6.txt"
    out_file.write_text(
        "\n".join(path for _, __, path in selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )

    print(f"selected={len(selected)}")
    print("first10=")
    for _, __, path_str in selected[:10]:
        print(path_str)


if __name__ == "__main__":
    main()
