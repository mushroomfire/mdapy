import os
import sys
import re
import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

version_file = Path(__file__).resolve().parent.parent / "src/mdapy/__init__.py"

with open(version_file, encoding="utf-8") as f:
    content = f.read()

match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
if match:
    release = match.group(1)
else:
    release = "unknown"

project = "mdapy"
year = datetime.date.today().year
copyright = f"2022-{year} Yongchao Wu"
author = "Yongchao Wu"

templates_path = ["_templates"]
exclude_patterns = ["_build", "_esbonio", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",  # for jupyter
]


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": False,
    "exclude-members": "__weakref__",
    "member-order": "bysource",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_param = True

html_theme = "furo"
html_title = f"{project} {release}"
html_favicon = "_static/mdapy-icon.svg"
html_theme_options = {
    "light_logo": "mdapy-logo.svg",
    "dark_logo": "mdapy-logo.svg",
    "announcement": (
        "This is MDAPY 1.x. We recommend upgrading to MDAPY 2.x: "
        '<a href="https://mushroomfire.github.io/mdapy/">its documentation</a>, and '
        '<a href="https://mushroomfire.github.io/mdapy/getting-started/from-1x/">'
        "the 2.0 way of each 1.x call</a>."
    ),
}

# Suppress ambiguous cross-reference warnings for short attribute names (e.g. "N")
# that appear in multiple classes on the same documentation page.
suppress_warnings = ["ref.python"]
