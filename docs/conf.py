#!/usr/bin/env python
#
# gym_gridverse documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), '../VERSION.txt')
with open(version_file, 'r') as file_handler:
    __version__ = file_handler.read().strip()


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined), e.g. for
# py:function directives. Default is True.
add_module_names = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'gym-gridverse'
copyright = "2020, Andrea Baisero"
author = "Andrea Baisero"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------

html_logo = '../images/logo.svg'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme  # isort:skip pylint: disable=unused-import,wrong-import-position

# html_theme = 'classic'
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'gym_gridversedoc'


# -- Options for LaTeX output ------------------------------------------

latex_elements = {  # type: ignore
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        'gym_gridverse.tex',
        'gym-gridverse Documentation',
        'Andrea Baisero',
        'manual',
    ),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'gym_gridverse', 'gym-gridverse Documentation', [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        'gym_gridverse',
        'gym-gridverse Documentation',
        author,
        'gym_gridverse',
        'One line description of project.',
        'Miscellaneous',
    ),
]

# -- Configurations ----------------------------------------------------

napoleon_include_init_with_doc = True

# dev options
todo_emit_warnings = True
todo_include_todos = True

nitpicky = True
nitpick_ignore = [
    ('py:class', 'PositionOrTuple'),
    ('py:class', 'Ray'),
    ('py:class', 'TextIO'),
    ('py:class', 'enum.Enum'),
    ('py:class', 'gym.core.Env'),
    ('py:class', 'gym.envs.classic_control.rendering.Geom'),
    ('py:class', 'gym.spaces.space.Space'),
    ('py:class', 'gym_minigrid.wrappers.FullyObsWrapper'),
    ('py:class', 'list<bigint>'),
    ('py:class', 'np.ndarray'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'numpy.random.Generator'),
    ('py:class', 'numpy.random._generator.Generator'),
    ('py:class', 'rnd.Generator'),
    ('py:class', 'typing.Protocol'),
    ('py:class', 'typing_extensions.Protocol'),
    ('py:data', 'rng'),
    ('py:func', 'functools.partial'),
    ('py:func', 'gym.make'),
    ('py:meth', '__call__'),
    ('py:mod', 'gym_minigrid'),
    ('py:mod', 'typing'),
]

# recognizes custom types
autodoc_type_aliases = {
    'RecordingElement': 'gym_gridverse.recording.RecordingElement',
}
