"""Legacy portfolio evolution engine removed.

The intelligence facade now exposes a fallback stub for portfolio evolution and
orchestrates canonical ecosystem tooling under ``src.thinking.ecosystem``.  Use
those surfaces instead of importing this module directly.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.intelligence.portfolio_evolution was removed. Use the intelligence facade "
    "or the ecosystem optimizers under src.thinking.ecosystem instead."
)
