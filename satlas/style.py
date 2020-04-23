"""
Collection of style setting functions using both style sheets in the subfolder 'styles' and user-supplied adjustments.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be
"""
import matplotlib as mpl
import contextlib
import os
import glob

STYLE_PATH = os.path.dirname(os.path.realpath(__file__)) + '\\styles\\'
STYLES = [s.split('\\')[-1].split('.')[0] for s in glob.glob(STYLE_PATH + '*.mplstyle')]
STYLE_MAPPING = {s: os.path.dirname(os.path.realpath(__file__)) + '\\styles\\' + s + '.mplstyle' for s in STYLES}

__all__ = ['set', 'get_available_styles', 'context', 'set_font']

def set(style='standard'):
    """Sets the current style to a SATLAS style. For
    a list of available styles, use :func:`.get_available_styles()`.

    Parameters
    ----------
    style: str or list
        Style sheets specification. Valid options are:

        +------+----------------------------------------------------+
        | str  | The name of a SATLAS style.                        |
        +------+----------------------------------------------------+
        | list | A list of style specifiers (str) applied from first|
        |      | to last in the list.                               |
        +------+----------------------------------------------------+"""
    if mpl.cbook.is_string_like(style):
        style = [style]
    style = [STYLE_MAPPING[s] for s in style]
    mpl.style.use(style)

def get_available_styles():
    """Returns the available stylesheets in the subfolder 'styles'."""
    return STYLES

@contextlib.contextmanager
def context(style, after_reset=False):
    """Context manager for using style settings temporarily.

    Parameters
    ----------
    style: str or list
        Style sheets specification. Valid options are:

        +------+----------------------------------------------------+
        | str  | The name of a SATLAS style.                        |
        +------+----------------------------------------------------+
        | list | A list of style specifiers (str) applied from first|
        |      | to last in the list.                               |
        +------+----------------------------------------------------+

    after_reset : bool
        If True, apply style after resetting settings to their defaults;
        otherwise, apply style on top of the current settings.
    """
    if mpl.cbook.is_string_like(style):
        style = [style]
    style = [STYLE_MAPPING[s] for s in style]
    mpl.style.context(style, after_rest=after_reset)

def set_font(font='Palatino Linotype', family='serif'):
    """Sets the font to the chosen family and the family options
    to the supplied fonts:

    Parameters
    ----------
    font: str or list
        Name or list of names of fonts.
    family: str
        Family name of the fonts that will be set and used."""
    d = {'font.family': family, 'font.' + family: font}
    mpl.rcParams.update(d)
