import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------

needs_sphinx = "2.0"  # Nicer param docs
suppress_warnings = [
    "ref.citation",
    "myst.header",  # https://github.com/executablebooks/MyST-Parser/issues/262
]
project = "EUGENe"
copyright = "2022, Adam Klie, Hayden Stites"
author = "Adam Klie, Hayden Stites"
release = "0.0.4"

# default settings
templates_path = ["_templates"]
master_doc = "index"
default_role = "literal"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "sphinx_gallery.load_style",
]

autosummary_generate = True
# autodoc_inherit_docstrings = False
# autodoc_mock_imports = ["pytorch_lightning", "torch"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "EUGENe"
html_static_path = ["_static"]
html_show_sphinx = False

# Thumbnail selection for nbsphinx gallery
nbsphinx_thumbnails = {
    "tutorials/single_task_regression_tutorial": "_static/thumbnails/single_task_regression_thumbnail.png",
    "tutorials/binary_classification_tutorial": "_static/thumbnails/single_task_regression_thumbnail.png",
}

# -- Options for extensions -------------------------------------------------------------------------------
nbsphinx_execute = "never"
