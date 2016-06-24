import matplotlib as mpl
import contextlib
import os
import glob
import sys

STYLE_PATH = os.path.dirname(os.path.realpath(__file__)) + '\\styles\\'
STYLES = [s.split('\\')[-1].split('.')[0] for s in glob.glob(STYLE_PATH + '*.mplstyle')]
STYLE_MAPPING = {s: os.path.dirname(os.path.realpath(__file__)) + '\\styles\\' + s + '.mplstyle' for s in STYLES}

__all__ = ['set', 'get_available_styles', 'context', 'set_font']

def set(style='standard'):
    if mpl.cbook.is_string_like(style):
        style = [style]
    style = [STYLE_MAPPING[s] for s in style]
    mpl.style.use(style)

def get_available_styles():
    return STYLES

@contextlib.contextmanager
def context(style, after_reset=False):
    if mpl.cbook.is_string_like(style):
        style = [style]
    style = [STYLE_MAPPING[s] for s in style]
    mpl.style.context(style, after_rest=after_reset)

def set_font(font='Palatino Linotype', family='serif'):
    d = {'font.family': family, 'font.' + family: font}
    mpl.rcParams.update(d)
