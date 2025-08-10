from __future__ import annotations


def apply_aggregate_cap(current_abs_exposure_sum: float, aggregate_cap: float, desired_abs: float) -> float:
    """
    Given current aggregate absolute exposure and a per-portfolio aggregate cap,
    return the allowed absolute exposure for the next position change.
    If adding desired_abs exceeds the cap, clamp to the remaining capacity.
    """
    current = max(0.0, float(current_abs_exposure_sum))
    cap = max(0.0, float(aggregate_cap))
    desired = max(0.0, float(desired_abs))
    remaining = max(0.0, cap - current)
    return min(desired, remaining)


def usd_beta_sign(symbol: str, exposure: float) -> float:
    """
    Approximate USD beta sign for a FX symbol: positive if USD quote currency (e.g., EURUSD)
    and exposure is long base vs USD, negative if USD is base (e.g., USDJPY) for long exposure.
    This is a very simplified proxy sufficient for tests.
    """
    s = (symbol or "").upper()
    if len(s) >= 6:
        base = s[:3]
        quote = s[3:6]
    else:
        base = s[:3]
        quote = "USD" if base != "USD" else "XXX"
    e = float(exposure)
    if quote == "USD":
        # Long base vs USD -> positive beta with USD
        return e
    if base == "USD":
        # Long USD vs other -> negative beta vs USD when long
        return -e
    # Cross not involving USD: negligible USD beta
    return 0.0