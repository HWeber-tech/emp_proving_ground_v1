def test_data_integration_has_no_indicator_code():
    import os
    import re
    forbidden = re.compile(r"\b(RSI|Bollinger|MACD|SMA|EMA|VectorizedIndicators)\b", re.I)
    root = "src/data_integration"
    for base, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(base, f), "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    assert not forbidden.search(content), f"Indicator code found in {os.path.join(base, f)}"

