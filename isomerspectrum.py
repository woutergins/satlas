"""
.. module:: CombinedSpectrum
    :platform: Windows
    :synopsis: Implementation of classes for the analysis of hyperfine
     structure spectra, including simultaneous fitting, various fitting
     routines and isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import abc
import emcee as mcmc
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import satlas.loglikelihood as llh
import satlas.profiles as p
import satlas.utilities as utils
from satlas.wigner import wigner_6j as W6J
from satlas.combinedspectrum import CombinedSpectrum

class IsomerSpectrum(CombinedSpectrum):

    """Create a spectrum containing the information of multiple hyperfine
    structures. Most common use will be to fit a spectrum containing an isomer,
    hence the name of the class.

    Parameters
    ----------
    spectra: list of :class:`SingleSpectrum` instances
        A list containing the base spectra"""

    def __init__(self, spectra):
        super(IsomerSpectrum, self).__init__(spectra)
        self.shared = []

    def sanitize_input(self, x, y, yerr=None):
        """Doesn't do anything yet."""
        x, y = np.array(x), np.array(y)
        if yerr is not None:
            yerr = np.array(yerr)
        return x, y, yerr

    def params_from_var(self):
        """Combine the parameters from the subspectra into one Parameters
        instance.

        Returns
        -------
        params: Parameters instance describing the spectrum"""
        params = super(IsomerSpectrum, self).params_from_var()
        for i, s in enumerate(self.spectra):
            if i == 0:
                continue
            else:
                new_key = 's' + str(i) + '_Background'
                params[new_key].value = 0
                params[new_key].vary = False
                params[new_key].expr = None
        return params

    def seperate_response(self, x):
        """Get the response for each seperate spectrum for the values x,
        without background.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz.

        Returns
        -------
        list of floats or NumPy arrays
            Seperate responses of spectra to the input :attr:`x`."""
        return [s(x) - s.background for s in self.spectra]

    def __add__(self, other):
        if isinstance(other, IsomerSpectrum):
            spectra = self.spectra + other.spectra
        elif isinstance(other, SingleSpectrum):
            spectra = self.spectra
            spectra.append(other)
        else:
            raise TypeError('unsupported operand type(s)')
        return IsomerSpectrum(spectra)

    def __call__(self, x):
        return np.sum([s(x) for s in self.spectra], axis=0)