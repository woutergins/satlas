"""
Implementation of a class for the analysis of linear data.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""
from . import lmfit as lm
from .basemodel import BaseModel
from .loglikelihood import poisson_llh
from .summodel import SumModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['PolynomialModel', 'MiscModel']


class PolynomialModel(BaseModel):

    r"""Constructs a polynomial response."""

    def __init__(self, args):
        """:class:`.PolynomialModel` creates a general polynomial
        of the order given by *len(args)-1*. The given coefficients
        are ordered lowest to highest order.

        Parameters
        ----------
        args: iterable of values
            Iterable containing all the values for the
            coefficients. Polynomial order is determined
            by the length. args[0] is the coefficient
            of order 0, etc..."""
        super(PolynomialModel, self).__init__()
        self.lnprior_mapping = {}
        self._populate_params(args)

    @property
    def params(self):
        self._params = self._check_variation(self._params)
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    ####################################
    #      INITIALIZATION METHODS      #
    ####################################

    def _populate_params(self, args):
        # Prepares the params attribute with the initial values
        par = lm.Parameters()
        for i, val in reversed(list(enumerate(args))):
            par.add('Order' + str(i) + 'Coeff', value=val, vary=True)

        self.params = self._check_variation(par)

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
        if isinstance(other, PolynomialModel):
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
        return np.polyval([self.params[p].value for p in self.params.keys()], x)

class MiscModel(BaseModel):

    r"""Constructs a response from a supplied function.
    Call signature is

    def func(x, par):

        a = par[0]

        b = par[1]

        ...

        return y"""

    def __init__(self, func, args, name_list=None):
        """The :class:`.MiscModel` takes a supplied function *func* and list of starting
        argument parameters *args* to contruct an object that responds with the
        given function for the parameter values. A list of names can also
        be supplied to customize the parameter names.

        Parameters
        ----------
        func: callable
            A callable function with call signature *func(x, args)*.
        args: list of values
            List of starting values for the parameters. The number of parameters is based
            on the length of the list of arguments.
        name_list: list of strings, optional
            List of names to be supplied to the parameters. The order of the names
            and the order of the parameters is the same, so *name_list[0]* corresponds
            to *args[0]*."""
        super(MiscModel, self).__init__()
        self.func = func
        self.lnprior_mapping = {}
        self._populate_params(*args, name_list=name_list)

    @property
    def params(self):
        self._params = self._check_variation(self._params)
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    ####################################
    #      INITIALIZATION METHODS      #
    ####################################

    def _populate_params(self, *args, name_list=None):
        # Prepares the params attribute with the initial values
        par = lm.Parameters()
        if name_list is None:
            for i, val in enumerate(args):
                par.add('Param' + str(i + 1), value=val, vary=True)
        else:
            for name, val in zip(name_list, args):
                par.add(name, value=val, vary=True)

        self.params = self._check_variation(par)

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
        other: BaseModel
            Other spectrum to add.

        Returns
        -------
        SumModel
            A SumModel combining both spectra."""
        if isinstance(other, MiscModel):
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
        return self.func(x, [self.params[p].value for p in self.params.keys()])
