# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
"""Every Python example in the documentation must run as shown.

Code blocks of one page run in order and share variables, in a temporary
working directory, so examples may write files. A block preceded by the line
``<!-- skip-test -->`` is an illustration and is not run. A block preceded by
``<!-- script: name.py -->`` is saved under that name and run by itself, as a
reader runs a script that starts processes. A page with a line
``<!-- needs: name -->`` is skipped where the module ``name`` cannot be
imported. The files of a folder, given relative to the page by a line
``<!-- files: folder -->``, are copied into the working directory first,
as readers download them. The working directory has an empty folder
``images``, where examples save the pictures shown on their page.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(
    r"(?P<before>[^\n]*)\n```python[^\n]*\n(?P<code>.*?)\n```", re.DOTALL
)
SKIP = "<!-- skip-test -->"
SCRIPT = re.compile(r"<!-- script: ([\w.]+) -->")
NEEDS = re.compile(r"<!-- needs: ([\w.]+) -->")
FILES = re.compile(r"<!-- files: ([\w./-]+) -->")


def python_blocks(page: Path) -> list[tuple[str, str | None]]:
    """The code to run of each block, and the name of the file if it is a script."""
    blocks = []
    for match in BLOCK.finditer(page.read_text(encoding="utf-8")):
        if match["before"].strip() != SKIP:
            script = SCRIPT.fullmatch(match["before"].strip())
            blocks.append((match["code"], script[1] if script else None))
    return blocks


PAGES = sorted(p for p in DOCS.rglob("*.md") if python_blocks(p))
README = DOCS.parent / "README.md"
# the README is also the page of the package on PyPI, where only full URLs work
IN_REPOSITORY = re.compile(
    r"https://(?:raw\.githubusercontent\.com/mushroomfire/mdapy"
    r"|github\.com/mushroomfire/mdapy/blob)/(?:v2|master)/([\w./-]+)"
)


def test_pages_with_examples_are_found():
    assert DOCS.is_dir()
    assert any(p.name == "box.md" for p in PAGES)


def test_the_example_of_the_readme_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for code, _ in python_blocks(README):
        exec(compile(code, "README.md", "exec"), namespace)
    assert namespace["structures"].counts()["fcc"] == 10796
    assert Path("grains.png").stat().st_size > 0


def test_the_files_the_readme_points_to_exist():
    paths = IN_REPOSITORY.findall(README.read_text(encoding="utf-8"))
    assert paths
    assert not [path for path in paths if not (README.parent / path).exists()]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_examples_run(page, tmp_path, monkeypatch):
    for module in NEEDS.findall(page.read_text(encoding="utf-8")):
        pytest.importorskip(module)
    for folder in FILES.findall(page.read_text(encoding="utf-8")):
        shutil.copytree(page.parent / folder, tmp_path, dirs_exist_ok=True)
    (tmp_path / "images").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    try:
        for i, (code, script) in enumerate(python_blocks(page)):
            if script is not None:
                Path(script).write_text(code, encoding="utf-8")
                done = subprocess.run(
                    [sys.executable, script], capture_output=True, text=True
                )
                assert done.returncode == 0, f"{script} failed:\n{done.stderr}"
                continue
            try:
                exec(compile(code, f"{page.name}[block {i}]", "exec"), namespace)
            except Exception as exc:  # pragma: no cover - only reached on failure
                pytest.fail(
                    f"{page.relative_to(DOCS)} block {i} failed: {exc!r}\n{code}"
                )
    finally:
        # the figures a page leaves open, which matplotlib warns about
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")
