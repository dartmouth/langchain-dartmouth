import os
import sys

sys.path.insert(0, os.path.abspath("../src/langchain_dartmouth/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "langchain_dartmouth"
copyright = (
    "by Simon Stone for Dartmouth College under Creative Commons CC BY-NC 4.0 License"
)
author = "Simon Stone"
release = "0.2.12"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinxcontrib.autodoc_pydantic",
]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_model_undoc_members = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = html_static_path[0] + "/img/d-pine.png"
html_logo = html_static_path[0] + "/img/langchain_dartmouth-logo-light.png"
html_css_files = [
    "css/custom.css",
]
