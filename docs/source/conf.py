# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quivr'
copyright = '2023, Spencer Nelson'
author = 'Spencer Nelson'

import quivr
version = quivr.__version__
release = quivr.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_toolbox.more_autodoc.typehints",
    "sphinx_toolbox.more_autodoc.typevars",
    'sphinx_toolbox.more_autodoc.overloads'
]

# From sphinx_toolbox.more_autodoc.typehints
hide_none_rtype = True

templates_path = ['_templates']
exclude_patterns = []

autodoc_type_aliases = {
    "MetadataDict": "MetadataDict",
    "quivr.MetadataDict": "MetadataDict",
    "quivr.columns.MetadataDict": "MetadataDict",
}

autodoc_member_order = 'bysource'
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

nitpicky = True
nitpick_ignore = {
    ("py:mod", "quivr.columns"),
    ("py:mod", "quivr.tables"),
    # see: https://github.com/apache/arrow/issues/35413, should be fixed in 13.0.0
    ("py:class", "pyarrow.FloatArray"),
    ("py:class", "pyarrow.HalfFloatArray"),
    ("py:class", "pyarrow.DoubleArray"),

    ("py:mod", "pyarrow.compute"),
}
