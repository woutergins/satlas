import pickle
from analysis import Analysis
from singlespectrum import SingleSpectrum
from combinedspectrum import CombinedSpectrum
from isomerspectrum import IsomerSpectrum

def save(analysis, filename):
    with open(filename + '.analysis','wb') as f:
        pickle.dump(analysis, f)

def load(filename):
    with open('test.analysis','rb') as f:
        return pickle.load(f)