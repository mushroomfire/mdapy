# for using Read the Docs theme
import sphinx_rtd_theme
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "mdapy"
copyright = "2022, mushroomfire"
author = "mushroomfire"
release = "0.7.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]

# html_theme = 'sphinxdoc'
html_theme = "sphinx_rtd_theme"

# html_theme_path = []
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
