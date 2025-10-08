"""Pytest package marker to avoid module-name collisions during collection."""

# Ensure guardrail-critical modules are loaded before lightweight stubs replace them.
import src.runtime.runtime_builder  # noqa: F401  (import side effect)
