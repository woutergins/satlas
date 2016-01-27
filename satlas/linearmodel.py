"""
Implementation of a class for the analysis of linear data.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .summodel import SumModel
from .basemodel import BaseModel
from .loglikelihood import poisson_llh

__all__ = ['LinearModel']


class LinearModel(BaseModel):

    r"""Constructs a linear response."""


    def __init__(self, slope=1, intercept=0):
        super(LinearModel, self).__init__()
        self.lnprior_mapping = {}
        self._populate_params(slope, intercept)

    def lnprior(self):
        # Check if the parameter values are within the acceptable range.
        for key in self._params.keys():
            par = self._params[key]
            if par.vary:
                try:
                    leftbound, rightbound = (par.priormin,
                                             par.priormax)
                except:
                    leftbound, rightbound = par.min, par.max
                leftbound = -np.inf if leftbound is None else leftbound
                rightbound = np.inf if rightbound is None else rightbound
                if not leftbound < par.value < rightbound:
                    return -np.inf
        # If defined, calculate the lnprior for each seperate parameter
        try:
            return_value = 1.0
            for key in self.params.keys():
                if key in self.lnprior_mapping.keys():
                    return_value += self.lnprior_mapping[key](self.params[key].value)
            return return_value
        except:
            pass
        return 1.0

    @property
    def params(self):
        """Instance of lmfit.Parameters object characterizing the
        shape of the HFS."""
        self._params = self._check_variation(self._params)
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    ####################################
    #      INITIALIZATION METHODS      #
    ####################################

    def _populate_params(self, slope, intercept):
        # Prepares the params attribute with the initial values
        par = lm.Parameters()
        par.add('Slope', value=slope, vary=True)
        par.add('Intercept', value=intercept, vary=True)

        self.params = self._check_variation(par)

    def _check_variation(self, par):
        # Make sure the variations in the params are set correctly.
        for key in self._vary.keys():
            if key in par.keys():
                par[key].vary = self._vary[key]

        for key in self._constraints.keys():
            for bound in self._constraints[key]:
                if bound.lower() == 'min':
                    par[key].min = self._constraints[key][bound]
                elif bound.lower() == 'max':
                    par[key].max = self._constraints[key][bound]
                else:
                    pass
        return par

    #######################################
    #      METHODS CALLED BY FITTING      #
    #######################################

    def _sanitize_input(self, x, y, yerr=None):
        return x, y, yerr

    def seperate_response(self, x):
        """Wraps the output of the :meth:`__call__` in a list, for
        ease of coding in the fitting routines."""
        return [self(x)]

    ###########################
    #      MAGIC METHODS      #
    ###########################

    def __add__(self, other):
        """Add two models together to get an :class:`.SumModel`.

        Parameters
        ----------
        other: HFSModel
            Other spectrum to add.

        Returns
        -------
        SumModel
            A SumModel combining both spectra."""
        if isinstance(other, LinearModel):
            l = [self, other]
        elif isinstance(other, SumModel):
            l = [self] + other.models
        return SumModel(l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __call__(self, x):
        s = self._params['Slope'].value * x + self._params['Intercept'].value
        return s
