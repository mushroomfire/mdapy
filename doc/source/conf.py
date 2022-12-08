# for using Read the Docs theme
import sphinx_rtd_theme

project = "mdapy"
copyright = "2022, mushroomfire"
author = "mushroomfire"
release = "0.7.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# html_theme = 'sphinxdoc'
html_theme = "sphinx_rtd_theme"

# html_theme_path = []
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
