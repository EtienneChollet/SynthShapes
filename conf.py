# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, '/Users/etiennechollet/Desktop/coding/SynthShapes')  # Adjust as needed


exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'conf.rst', 'setup.rst', 'setup.py']
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SynthShapes'
copyright = '2024, Etienne Chollet'
author = 'Etienne Chollet'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For NumPy and Google-style docstrings
    "sphinx.ext.viewcode",  # To link to source code
    "sphinx.ext.autosummary",  # To generate summary tables
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']



napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True  # Include the `__init__` method docstring
napoleon_include_private_with_doc = False  # Skip private methods
napoleon_include_special_with_doc = True  # Include special methods like __call__
napoleon_use_param = True  # Use `Parameters` for function arguments
napoleon_use_rtype = True  # Use `Returns` for return type
autosummary_generate = True  # Automatically generate summary .rst files
