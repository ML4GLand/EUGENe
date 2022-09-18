import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, os.path.abspath(".."))
import eugene  # noqa

on_rtd = os.environ.get("READTHEDOCS") == "True"

# -- General configuration ---------------------------------------------------

nitpicky = True  # Warn about broken links. This is here for a reason: Do not change.
needs_sphinx = "2.0"  # Nicer param docs
suppress_warnings = [
    "ref.citation",
    "myst.header",  # https://github.com/executablebooks/MyST-Parser/issues/262
]
project = "EUGENe"
copyright = "2022, Adam Klie, Hayden Stites"
author = "Adam Klie, Hayden Stites"
release = "0.0.0"

# default settings
templates_path = ["_templates"]
master_doc = "index"
default_role = "literal"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
api_dir = HERE / "api"  # function_images

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_sphinx = False
