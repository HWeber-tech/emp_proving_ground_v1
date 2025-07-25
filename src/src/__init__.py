"""
Compatibility layer for legacy imports.

This package exists to support modules that import subpackages via the
`src.*` namespace when the `src` directory itself is placed on the
Python path.  Without this layer, imports like `from src.core import ...`
would fail with "attempted relative import beyond top-level package"
because there is no actual `src` package when `src` is added to
``sys.path``.  By defining this package and re-exporting the top-level
modules, both `import core` and `import src.core` will refer to the same
objects.

Whenever a new top-level module is added to the project, add its name to
the ``_EXPOSED_MODULES`` list below so it can be accessed via the
``src.`` namespace.
"""

import sys

# List of top-level modules to expose under the ``src`` namespace.
_EXPOSED_MODULES = [
    'core',
    'risk',
    'pnl',
    'data',
    'sensory',
    'evolution',
    'simulation',
    'trading',
    'validation',
    'data_integration',
    'decision_genome',
    # The ``genome`` package contains encoders, decoders and models.  Some
    # modules import it via ``src.genome``, so expose it here to avoid
    # import errors when ``src`` is on ``sys.path``.
    'genome',

    # Additional top-level packages referenced via ``src.`` in various modules.
    # Exposing these names ensures that imports like ``src.domain.models`` or
    # ``src.governance.token_manager`` resolve correctly when ``src`` is on
    # ``sys.path``.  Without listing them here, attempting to import them
    # through the ``src`` namespace would result in a ModuleNotFoundError.
    'domain',
    'ecosystem',
    'governance',
    'operational',
    'thinking',
]

for _mod_name in _EXPOSED_MODULES:
    try:
        # Import the top-level module.  If it fails, skip exposing it.
        __import__(_mod_name)
        # Register the module under the src namespace
        sys.modules[__name__ + '.' + _mod_name] = sys.modules[_mod_name]
    except ImportError:
        # Some modules may not be importable in certain environments (e.g. optional
        # dependencies).  Suppress the error so that unavailable modules do not
        # break the loading of this compatibility layer.
        pass

__all__ = [name for name in _EXPOSED_MODULES if 'src.' + name in sys.modules]
