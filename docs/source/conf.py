import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/Numpy docstring対応
]

html_theme = 'sphinx_rtd_theme'
project = 'mlsampler'
author = 'Jugai O'
release = metadata.version('mlsampler')
version = release