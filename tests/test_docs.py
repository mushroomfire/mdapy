# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.
"""Every Python example in the documentation must run as shown.

Code blocks of one page run in order and share variables, in a temporary
working directory, so examples may write files. A block preceded by the line
``<!-- skip-test -->`` is an illustration and is not run. A page with a line
``<!-- needs: name -->`` is skipped where the module ``name`` cannot be
imported.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(
    r"(?P<before>[^\n]*)\n```python[^\n]*\n(?P<code>.*?)\n```", re.DOTALL
)
SKIP = "<!-- skip-test -->"
NEEDS = re.compile(r"<!-- needs: ([\w.]+) -->")


def python_blocks(page: Path) -> list[str]:
    blocks = []
    for match in BLOCK.finditer(page.read_text(encoding="utf-8")):
        if match["before"].strip() != SKIP:
            blocks.append(match["code"])
    return blocks


PAGES = sorted(p for p in DOCS.rglob("*.md") if python_blocks(p))


def test_pages_with_examples_are_found():
    assert DOCS.is_dir()
    assert any(p.name == "box.md" for p in PAGES)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_examples_run(page, tmp_path, monkeypatch):
    for module in NEEDS.findall(page.read_text(encoding="utf-8")):
        pytest.importorskip(module)
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for i, code in enumerate(python_blocks(page)):
        try:
            exec(compile(code, f"{page.name}[block {i}]", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - only reached on failure
            pytest.fail(f"{page.relative_to(DOCS)} block {i} failed: {exc!r}\n{code}")
