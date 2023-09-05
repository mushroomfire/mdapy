# for using Read the Docs theme
import sphinx_rtd_theme
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "mdapy"
copyright = "2022, Yong-Chao Wu & Jian-Li Shao Group in Beijing Institute of Technology"
author = "Yong-Chao Wu & Jian-Li Shao Group in Beijing Institute of Technology"
version = "0.8.9"
release = "0.8.9"

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
    "nbsphinx",
]


templates_path = ["_templates"]

# html_theme = 'sphinxdoc'
html_theme = "sphinx_rtd_theme"

html_logo = "images/mdapy_logo_white.png"
html_favicon = "images/mdapy_favico_white.ico"

# html_theme_path = []
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
