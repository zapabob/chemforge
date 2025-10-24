# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'ChemForge'
copyright = '2025, ChemForge Development Team'
author = 'ChemForge Development Team'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'sphinx.ext.imgmath',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'myst_parser',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True

# -- Options for napoleon extension -----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx extension --------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'rdkit': ('https://www.rdkit.org/docs/', None),
    'streamlit': ('https://docs.streamlit.io/', None),
    'dash': ('https://dash.plotly.com/', None),
}

# -- Options for todo extension ---------------------------------------------
todo_include_todos = True

# -- Options for coverage extension -----------------------------------------
coverage_show_missing_items = True

# -- Options for math extension ---------------------------------------------
mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

# -- Options for imgmath extension -----------------------------------------
imgmath_latex = 'latex'
imgmath_latex_args = ['-interaction=nonstopmode', '-halt-on-error']
imgmath_use_preview = True

# -- Options for graphviz extension -----------------------------------------
graphviz_output_format = 'png'

# -- Options for inheritance_diagram extension -------------------------------
inheritance_graph_attrs = dict(rankdir="TB", size='"6.0, 8.0"', fontsize=14, ratio='compress')

# -- Options for MyST parser ------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------
html_theme_options = {
    'analytics_id': '',  # Provided by Google Analytics
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    'preamble': '',
    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'ChemForge.tex', 'ChemForge Documentation',
     'ChemForge Development Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ('index', 'chemforge', 'ChemForge Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    ('index', 'ChemForge', 'ChemForge Documentation',
     author, 'ChemForge', 'Advanced CNS drug discovery platform with PWA+PET Transformer',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The basename for the epub file. It defaults to the project name.
epub_basename = project

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Options for linkcheck extension ----------------------------------------
linkcheck_ignore = [
    r'http://localhost:\d+',
    r'http://127.0.0.1:\d+',
    r'https://github.com/.*#.*',
]

# -- Options for doctest extension ------------------------------------------
doctest_global_setup = '''
import numpy as np
import pandas as pd
import torch
'''

# -- Options for viewcode extension -----------------------------------------
viewcode_import = True

# -- Options for githubpages extension --------------------------------------
html_baseurl = 'https://chemforge.readthedocs.io/'

# -- Options for sphinx_rtd_theme ------------------------------------------
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}