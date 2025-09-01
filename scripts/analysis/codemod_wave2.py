from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Tuple

FUTURE_LINE = "from __future__ import annotations"
SHEBANG_RE = re.compile(r"^#!.*\n")
TRIPLE_START = re.compile(r"^[ruRU]*([\'\"]{3})")

# Conservative, semantics-preserving text substitutions for built-in generics
GENERIC_MAP: dict[str, str] = {
    "List[": "list[",
    "Dict[": "dict[",
    "Tuple[": "tuple[",
    "Set[": "set[",
    "FrozenSet[": "frozenset[",
    # Note: We intentionally do NOT rewrite abstract/typing constructs like
    # Sequence/Mapping/Iterable/etc. to avoid changing runtime semantics.
}


def insert_future(text: str) -> Tuple[str, bool]:
    """
    Insert 'from __future__ import annotations' after any shebang and
    top-level module docstring if not already present. Returns (new_text, changed).
    Also normalizes spacing to ensure exactly one blank line after the future import,
    satisfying Black/PEP 8.
    """
    # Normalize if FUTURE_LINE already present near top: ensure exactly one blank line after it
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines[:10]):
        if line.strip() == FUTURE_LINE:
            # Ensure next line exists and is a single blank line
            # Target: FUTURE_LINE + "\n" and ensure following line is blank (one empty line)
            # Current line already endswith newline by splitlines(keepends=True) semantics; if not, handle robustly.
            changed = False
            # Ensure FUTURE_LINE line ends with a newline
            if not line.endswith("\n"):
                lines[i] = line + "\n"
                changed = True
            # Ensure exactly one blank line after
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip() != "":
                    # Insert a blank line
                    lines.insert(i + 1, "\n")
                    changed = True
                else:
                    # Next line is blank; ensure there's not an extra blank after it
                    # If there is a second consecutive blank, remove extras
                    j = i + 2
                    while j < len(lines) and lines[j].strip() == "":
                        del lines[j]
                        changed = True
            else:
                # Future import is last line; append a blank line
                lines.append("\n")
                changed = True
            if changed:
                return "".join(lines), True
            return text, False
    # If not present, insert it in the correct place
    idx = 0
    m = SHEBANG_RE.match(text)
    if m:
        idx = m.end()

    # Find insertion position after module docstring if present
    rest = text[idx:]
    insert_pos = idx
    # Simple docstring detection at file start
    if rest.startswith(("'", '"')) or TRIPLE_START.match(rest):
        q3 = rest[:3]
        if q3 in ("'''", '"""'):
            end = rest.find(q3, 3)
            if end != -1:
                insert_pos = idx + end + 3

    prefix, suffix = text[:insert_pos], text[insert_pos:]
    # No blank line before future import (must be immediately after docstring if any)
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    # Ensure exactly one blank line after the future import
    if suffix.startswith("\n"):
        future_block = f"{FUTURE_LINE}\n"
    else:
        future_block = f"{FUTURE_LINE}\n\n"
    new_text = f"{prefix}{future_block}{suffix}"
    return new_text, True


def convert_generics(text: str) -> Tuple[str, bool]:
    """
    Replace legacy typing generics (List/Dict/Tuple/Set/FrozenSet) with PEP 585 built-in generics.
    Does not touch abstract containers like Sequence/Mapping/etc.
    """
    changed = False
    new_text = text
    for old, newv in GENERIC_MAP.items():
        if old in new_text:
            new_text = new_text.replace(old, newv)
            changed = True
    return new_text, changed


def main(paths_file: str) -> int:
    paths = [l.strip() for l in Path(paths_file).read_text().splitlines() if l.strip()]
    changed_files: list[str] = []

    for rel in paths:
        p = Path(rel)
        if not p.exists() or not p.is_file():
            continue
        try:
            src = p.read_text()
        except Exception:
            continue

        new_src, did_future = insert_future(src)
        new_src2, did_generics = convert_generics(new_src)

        if did_future or did_generics:
            p.write_text(new_src2)
            changed_files.append(rel)

    Path("changed_files_batch6_wave2.txt").write_text(
        "\n".join(changed_files) + ("\n" if changed_files else "")
    )

    print(f"Codemod-modified {len(changed_files)} files")
    for cf in changed_files:
        print(cf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))