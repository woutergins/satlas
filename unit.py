from .singlespectrum import SingleSpectrum
from .combinedspectrum import CombinedSpectrum
from copy import deepcopy
import pickle
import numpy as np
import os

class DataUnit(object):
    def __init__(self,name,files):
        super(DataUnit,self).__init__()
        self.name = name
        self.files = files
        self.data_loaded = False

    def loadData(self,load_again = False):
        ## load from paths
        if not load_again and self.data_loaded:
            return 

        self._x = []
        self._y = []
        for path in self.files:
            data = np.loadtxt(path)
            self._x.append(data.T[0])
            self._y.append(data.T[1])

        self.data_loaded = True

    def analyse_chisq(self):
        self.loadData()
        
        chisq = self.spectrum.chisquare_spectroscopic_fit(x=self._x,y=self._y)
        return chisq

    def analyse_mle(self,**kwargs):
        self.loadData()

        ## rough for now of course
        self.spectrum.likelihood_fit(x=self._x,y=self._y,
                           walking=True,  # Perform the walk
                           walkers=200,  # Number of walkers,
                                        # see the emcee documentation for
                                        # more on this
                           nsteps=20,  # Number of steps for each walker
                           burnin=10.0,  # Defines the percentage of burnin
                           )

    def plot(self,name=None, **kwargs):
        self.loadData()

        return self.spectrum.plot_spectroscopic(self._x,self._y, **kwargs)