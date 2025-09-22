#!/usr/bin/env python3

import argparse
import csv
import os

HTML_TMPL = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>WHY Features Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #f4f4f4; text-align: left; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
  </head>
<body>
  <h1>WHY Features</h1>
  <table>
    <thead>
      <tr>{headers}</tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <h1>Summary</h1>
  <ul>
    <li>Total rows: {rowcount}</li>
  </ul>
  <p>Generated at {now}</p>
  <h1>PnL vs Costs (last 100 rows)</h1>
  <table>
    <thead><tr><th>timestamp</th><th>pnl</th><th>cum_cost</th><th>regime</th></tr></thead>
    <tbody>
      {pnltable}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description="Generate HTML table from why_features.csv")
    p.add_argument("--csv", default="docs/reports/backtests/why_features.csv")
    p.add_argument("--out", default="docs/reports/backtests/WHY_FEATURES.html")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return 1
    headers = []
    rows = []
    with open(args.csv, "r", encoding="utf-8") as fh:
        r = csv.reader(fh)
        for i, row in enumerate(r):
            if i == 0:
                headers = [f"<th>{h}</th>" for h in row]
            else:
                rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    # Build a simple PnL vs cost table using last 100 rows
    pnltable = []
    try:
        with open(args.csv, "r", encoding="utf-8") as fh:
            rr = list(csv.DictReader(fh))[-100:]
            for d in rr:
                pnltable.append(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                        d.get("timestamp", ""),
                        d.get("pnl", ""),
                        d.get("cum_cost", ""),
                        d.get("regime", ""),
                    )
                )
    except Exception:
        pass
    from datetime import datetime

    html = HTML_TMPL.format(
        headers="".join(headers),
        rows="\n".join(rows),
        rowcount=len(rows),
        now=datetime.utcnow().isoformat(),
        pnltable="\n".join(pnltable),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
