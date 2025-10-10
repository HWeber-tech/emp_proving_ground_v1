#!/usr/bin/env python3
"""Lightweight Markdown structure check for docs/ tree."""

from __future__ import annotations

import pathlib
import sys

errors = []
for path in pathlib.Path('docs').rglob('*.md'):
    if any(part in {'development', 'reports'} for part in path.parts):
        continue
    text = path.read_text(encoding='utf-8')
    if not text.strip():
        errors.append(f"{path}: empty file")
    if not text.lstrip().startswith('#'):
        errors.append(f"{path}: missing top-level heading")

if errors:
    print('\n'.join(errors))
    sys.exit(1)

print('Markdown structure check passed.')
