#!/usr/bin/env python3
"""Generate the market data lineage documentation."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_foundation.documentation import (  # noqa: E402
    build_market_data_lineage,
    render_lineage_markdown,
)


OUTPUT_PATH = Path("docs/deployment/data_lineage.md")


def main() -> int:
    document = build_market_data_lineage()
    markdown = render_lineage_markdown(document)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
