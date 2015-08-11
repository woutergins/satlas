import pickle
from satlas.analysis import Analysis
from satlas.singlespectrum import SingleSpectrum
from satlas.combinedspectrum import CombinedSpectrum
from satlas.isomerspectrum import IsomerSpectrum

def save(analysis, filename):
    with open(filename + '.analysis','wb') as f:
        pickle.dump(analysis, f)

def load(filename):
    with open('test.analysis','rb') as f:
        return pickle.load(f)