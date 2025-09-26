# Type Stub Intake Checklist

Use this checklist whenever you introduce a new runtime dependency or upgrade an existing one. Following these steps keeps mypy visibility high without littering the codebase with broad `type: ignore` directives.

1. **Check for published stubs**
   - Run `python -m pip index versions <package>` and look for `types-<package>`.
   - If stubs exist, add them to `requirements/dev.txt` (or `requirements/type-stubs.txt`) and update `mypy.ini` if custom paths are needed.

2. **Evaluate inline protocol coverage**
   - If the dependency exposes a large surface area, define a minimal Protocol in `stubs/` or `src/.../protocols.py` capturing only the methods we call.
   - Avoid annotating client variables as `Any`; prefer Protocols with concrete return/argument types.

3. **Add project-specific stubs when necessary**
   - Place `.pyi` files under `stubs/<package>/` mirroring the import path.
   - Annotate every public attribute used by the codebase.
   - Include a `# TODO(types):` comment linking to the upstream issue if planning to contribute stubs back.

4. **Update documentation**
   - Note the dependency in the [mypy status log](mypy_status_log.md) if it changes strictness guarantees.
   - Cross-link any new helper utilities or coercion functions in `docs/mypy_conventions.md`.

5. **Verify locally**
   - Run `mypy --config-file mypy.ini src` to ensure no new warnings or unused ignores appear.
   - Add targeted tests exercising the typed interfaces when practical.

Keeping this checklist in version control ensures each new dependency enters the codebase with predictable typing guarantees.
