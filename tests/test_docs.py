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

import importlib
import inspect
import re
import shutil
import subprocess
import sys
from pathlib import Path

import conftest
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


def test_the_page_for_contributors_names_what_the_tests_need():
    page = (DOCS / "about" / "contributing.md").read_text(encoding="utf-8")
    needed = set()
    for test in (DOCS.parent / "tests").rglob("*.py"):
        needed |= set(
            re.findall(r'importorskip\(\s*"(\w+)', test.read_text(encoding="utf-8"))
        )
    assert not {name for name in needed if name not in page.lower()}
    workflow = (DOCS.parent / ".github" / "workflows" / "tests.yml").read_text()
    packages = re.search(r"missing = \{(.*?)\}", workflow)[1]
    for package in re.findall(r"'([\w-]+)'", packages):
        assert f"`{package}`" in page and f"PKG_{package}=on" in page


def test_the_python_api_lists_every_public_name():
    pages = (DOCS / "reference" / "api").glob("*.md")
    listed = set()
    for page in pages:
        listed |= set(re.findall(r"^::: (\S+)", page.read_text("utf-8"), re.M))
    missing = []
    for name in (
        "mdapy",
        "mdapy.io",
        "mdapy.neighbor",
        "mdapy.identify",
        "mdapy.analysis",
        "mdapy.build",
        "mdapy.transform",
        "mdapy.calculators",
        "mdapy.simulation",
        "mdapy.lattice_dynamics",
        "mdapy.mlip",
        "mdapy.visualization",
        "mdapy.interop",
    ):
        module = importlib.import_module(name)
        for item in module.__all__:
            skipped = item.isupper() or item.startswith("_")
            skipped = skipped or inspect.ismodule(getattr(module, item))
            if not skipped and not any(n.endswith(f".{item}") for n in listed):
                missing.append(f"{name}.{item}")
    assert not missing


def test_docstrings_have_no_roles_of_sphinx():
    # the API pages are Markdown: :meth:`sort` would show as it is written
    role = re.compile(r":(?:meth|func|class|data|attr|mod|exc|obj|ref):`")
    for path in sorted((DOCS.parent / "src" / "mdapy").rglob("*.py")):
        assert not role.search(path.read_text()), path


def test_tables_and_docstrings_give_exact_types():
    bare = re.compile(r"\b(ints?|floats?|f64|numpy\.ndarray)\b|\) array\b")
    found = []
    for path in sorted(DOCS.rglob("*.md")):
        if path.name == "ackland-jones.md":  # its column Type is the structure found
            continue
        column = None
        for line in path.read_text().splitlines():
            if not line.startswith("|"):
                column = None
                continue
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip("| "))]
            if "Type" in cells:
                column = cells.index("Type")
            elif (
                column is not None
                and len(cells) > column
                and bare.search(cells[column])
            ):
                found.append(f"{path.name}: {line[:70]}")
    parameter = re.compile(r"^\s+[\w*]+(?:, [\w*]+)* : .*\b(?:ints?|floats?)\b")
    for path in sorted((DOCS.parent / "src" / "mdapy").rglob("*.py")):
        for line in path.read_text().splitlines():
            if parameter.match(line):
                found.append(f"{path.name}: {line.strip()[:70]}")
    assert not found, "\n".join(found)


def test_a_lammps_without_a_package_is_recognized():
    error = Exception(
        "Unrecognized compute style 'adf' is part of the EXTRA-COMPUTE package "
        "which is not enabled in this LAMMPS binary."
    )
    assert conftest.lammps_package_missing(error) == "EXTRA-COMPUTE"
    assert conftest.lammps_package_missing(Exception("an atom is lost")) is None


def test_pages_with_examples_are_found():
    assert DOCS.is_dir()
    assert any(p.name == "box.md" for p in PAGES)


def test_the_examples_of_the_readme_run(tmp_path, monkeypatch):
    from mdapy import visualization

    monkeypatch.chdir(tmp_path)
    # the first picture of the movie is enough, and FFmpeg is optional
    monkeypatch.setattr(
        visualization, "write_movie", lambda path, pictures, **_: next(iter(pictures))
    )
    namespace: dict = {}
    for code, _ in python_blocks(README):
        exec(compile(code, "README.md", "exec"), namespace)
    assert namespace["structures"].counts()["fcc"] == 60682
    for name in ("grains", "boundaries", "phonons", "frames"):
        assert Path(f"{name}.png").stat().st_size > 0


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
