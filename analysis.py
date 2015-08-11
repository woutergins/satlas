from satlas.singlespectrum import SingleSpectrum
from satlas.combinedspectrum import CombinedSpectrum
from copy import deepcopy
import pickle
import numpy as np
import os

class Analysis(dict):
    """
    Class that contains references to (several) spectra that can 
    be tied to data for later analysis. Notes can also be added
    at will.

    parameters: 
    None

    Attributes:
    _dataPath: string
        Location of the data
    data_loaded: boolean
        Signifies if the data is loaded
    notes: list (of lists)
        List of notes for each spectrum - an empty
        list is appended to this list whenever a new
        spectrum is added
    """
    def __init__(self,paths):
        super(Analysis,self).__init__()
        self._dataPaths = ''

        self.data_loaded = False
        self.notes = {}

        self.dataPaths = paths

    ### Properties and setters
    @property
    def dataPaths(self):
        return self._dataPaths

    @dataPaths.setter
    def dataPaths(self,paths):
        self._dataPaths = paths
        self.loadData()

    ### Methods
    def loadData(self):
        ## load from paths
        try:
            self._x = []
            self._y = []
            for path in self.dataPaths:
                data = np.loadtxt(path)
                self._x.append(data.T[0])
                self._y.append(data.T[1])

            self.data_loaded = True
        except FileNotFoundError:
            print('One or more of the files was not found...')

    def analyse(self,name=None):
        chisq = self[name].chisquare_spectroscopic_fit(x=self._x,y=self._y)
        self[name].likelihood_fit(x=self._x,y=self._y,
                           walking=True,  # Perform the walk
                           walkers=200,  # Number of walkers,
                                        # see the emcee documentation for
                                        # more on this
                           nsteps=20,  # Number of steps for each walker
                           burnin=10.0,  # Defines the percentage of burnin
                           )
        return chisq

    def plot_spectrum(self,name=None):
        if not self.data_loaded:
            self.loadData()
        self[name].plot_spectroscopic(self._x,self._y)

    def __str__(self):
        ret = '' 
        ret += 'Data path:\n'
        ret += '\t' + str(self.dataPaths) + '\n'
        ret += 'Analysis history:\n'
        for name,spec in self.items():
            ret += '\t {}\n'.format(name)
            for n,par in spec.params_from_var().items():
                ret += '\t\t{}:{}+-{}\n'.format(n,par.value,par.stderr)
        return ret

    def __setitem__(self,key,val):
        # self.notes[key] = []
        super(Analysis,self).__setitem__(key,val)
