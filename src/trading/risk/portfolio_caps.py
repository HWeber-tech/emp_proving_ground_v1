from __future__ import annotations


def apply_aggregate_cap(current_total_abs: float, cap: float, desired_abs: float) -> float:
    """Return allowed absolute exposure given current total and cap.

    If current_total_abs + desired_abs <= cap, returns desired_abs.
    Otherwise, returns max(0, cap - current_total_abs).
    """
    if cap <= 0:
        return 0.0
    remaining = max(0.0, cap - max(0.0, current_total_abs))
    return min(desired_abs, remaining)


def usd_beta_sign(symbol: str, exposure: float) -> float:
    """Approximate USD beta contribution for an FX pair symbol and exposure.

    - If USD is base (e.g., USDJPY), USD beta aligns with exposure sign.
    - If USD is quote (e.g., EURUSD), USD beta opposes exposure sign.
    - If USD not present, returns 0.
    """
    s = (symbol or "").upper()
    if len(s) < 6:
        return 0.0
    base = s[:3]
    quote = s[3:6]
    if base == "USD":
        return exposure
    if quote == "USD":
        return -exposure
    return 0.0


