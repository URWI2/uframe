# Configuration file for the Sphinx documentation builder.


import os
import sys
sys.path.insert(0, os.path.abspath('../../src/uframe'))

print(os.path.abspath('../../src/uframe'))

project = 'uframe'
copyright = '2024, Christian Amesoeder, Michael Hagn'
author = 'Christian Amesoeder, Michael Hagn'
release = '0.0.25'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # automatically generate documentation for modules
    "sphinx.ext.napoleon",  # to read Google-style or Numpy-style docstrings
    "sphinx.ext.viewcode", # to allow vieing the source code in the web page
    'sphinx.ext.autodoc',
    "autodocsumm",  # to generate tables of functions, attributes, methods, etc.
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# don't include docstrings from the parent class
autodoc_inherit_docstrings = False
# Show types only in descriptions, not in signatures
autodoc_typehints = "description"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


