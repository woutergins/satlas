import pickle
from .analysis import Analysis
from .unit import DataUnit
from .singlespectrum import SingleSpectrum
from .combinedspectrum import CombinedSpectrum
from .isomerspectrum import IsomerSpectrum
from .spectrum import Spectrum

def save(analysis, filename, include_data = False):
    if not include_data:
        try:
            del analysis._x
        except AttributeError:
            pass
        try:
            del analysis._y
        except AttributeError:
            pass


        analysis.data_loaded = False

    if not '.analysis' in filename:
        filename += '.analysis'

    with open(filename,'wb') as f:
        pickle.dump(analysis, f)

def load(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)