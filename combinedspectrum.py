"""
.. module:: CombinedSpectrum
    :platform: Windows
    :synopsis: Implementation of class for the simultaneous fitting of hyperfine
     structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import copy
from .spectrum import Spectrum
__all__ = ['CombinedSpectrum']


class CombinedSpectrum(Spectrum):

    """A class for combining different spectra (:class:`CombinedSpectrum`) or
    combining isomers/isotopes (:class:`IsomerSpectrum`, child class).

    Parameters
    ----------
    spectra: list of :class:`IsomerSpectrum` or :class:`SingleSpectrum` objects
        A list defining the different spectra."""

    def __init__(self, spectra):
        super(CombinedSpectrum, self).__init__()
        self.spectra = spectra
        self.shared = ['Al',
                       'Au',
                       'Bl',
                       'Bu',
                       'Cl',
                       'Cu',
                       'Offset']

    def sanitize_input(self, x, y, yerr=None):
        # Take the :attr:`x`, :attr:`y`, and :attr:`yerr` inputs, and sanitize
        # them for the fit, meaning it should convert :attr:`y`/:attr:`yerr` to
        # the output format of the class, and :attr:`x` to the input format of
        # the class.
        if isinstance(y, list):
            y = np.hstack(y)
        if yerr is not None:
            if isinstance(yerr, list):
                yerr = np.hstack(yerr)
        return x, y, yerr

    @property
    def params(self):
        params = lm.Parameters()
        for i, s in enumerate(self.spectra):
            p = copy.deepcopy(s.params)
            keys = list(p.keys())
            for old_key in keys:
                new_key = 's' + str(i) + '_' + old_key
                p[new_key] = p.pop(old_key)
                # If an expression is defined, replace the old names with the new ones
                if p[new_key].expr is not None:
                    for o_key in keys:
                        n_key = 's' + str(i) + '_' + o_key
                        p[new_key].expr = p[new_key].expr.replace(o_key, n_key)
                # Link the shared parameters to the first subspectrum
                if any([shared in old_key for shared in self.shared]) and i > 0:
                    p[new_key].expr = 's0_' + old_key
                    p[new_key].vary = False
            params += p
        return params

    @params.setter
    def params(self, params):
        for i, spec in enumerate(self.spectra):
            par = lm.Parameters()
            for key in params:
                if key.startswith('s'+str(i)+'_'):
                    new_key = key[len('s'+str(i)+'_'):]
                    expr = params[key].expr
                    if expr is not None:
                        for k in params:
                            nk = k[len('s'+str(i)+'_'):]
                            expr = expr.replace(k, nk)
                    par.add(new_key,
                            value=params[key].value,
                            vary=params[key].vary,
                            min=params[key].min,
                            max=params[key].max,
                            expr=expr)
            spec.params = par

    #################################
    #      CONVENIENCE METHODS      #
    #################################

    def seperate_response(self, x):
        """Generates the response for each subspectrum.

        Parameters
        ----------
        x: list of arrays
            A list equal in length to the number of subspectra,
            contains arrays for which the subspectra have to be
            evaluated.

        Returns
        -------
        evaluated: ndarray
            The output array, of the same shape as the input
            list of arrays, containing the response values."""
        return np.squeeze([s.seperate_response(X)
                           for s, X in zip(self.spectra, x)])

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None,
             no_of_points=10**4, ax=None, show=True, legend=None,
             data_legend=None, xlabel='Frequency (MHz)', ylabel='Counts'):
        """Routine that plots the hfs, possibly on top of experimental data.

        Parameters
        ----------
        x: list of arrays
            Experimental x-data. If None, a suitable region around
            the peaks is chosen to plot the hfs.
        y: list of arrays
            Experimental y-data.
        yerr: list of arrays or dict('high': array, 'low': array)
            Experimental errors on y.
        no_of_points: int
            Number of points to use for the plot of the hfs if
            experimental data is given.
        ax: matplotlib axes object
            If provided, plots on this axis.
        show: boolean
            If True, the plot will be shown at the end.
        legend: string, optional
            If given, an entry in the legend will be made for the spectrum.
        data_legend: string, optional
            If given, an entry in the legend will be made for the experimental
            data.

        Returns
        -------
        fig, ax: matplotlib figure and axis
            Figure and axis used for the plotting."""
        if ax is None:
            fig, ax = plt.subplots(len(self.spectra), 1, sharex=True)
        else:
            fig = ax[0].get_figure()
        toReturn = fig, ax

        if x is None:
            x = [None] * len(self.spectra)
        if y is None:
            y = [None] * len(self.spectra)
        if yerr is None:
            yerr = [None] * len(self.spectra)

        selected = int(len(self.spectra)/2)
        for i, (X, Y, YERR, spec) in enumerate(zip(x, y, yerr,
                                                   self.spectra)):
            if i == selected:
                spec.plot(x=X, y=Y, yerr=YERR, no_of_points=no_of_points, ax=ax[i], show=False,
                          data_legend=data_legend, legend=legend, xlabel='', ylabel=ylabel)
            elif i == len(self.spectra) - 1:
                spec.plot(x=X, y=Y, yerr=YERR, no_of_points=no_of_points, ax=ax[i], show=False, ylabel='',
                          xlabel=xlabel)
            else:
                spec.plot(x=X, y=Y, yerr=YERR, no_of_points=no_of_points, ax=ax[i], show=False, xlabel='', ylabel='')

        plt.tight_layout()
        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, x=None, y=None,
                           no_of_points=10**4, ax=None, show=True):
        """Routine that plots the hfs of all the spectra, possibly on
        top of experimental data. It assumes that the y data is drawn from
        a Poisson distribution (e.g. counting data).

        Parameters
        ----------
        x: list of arrays
            Experimental x-data. If list of Nones, a suitable region around
            the peaks is chosen to plot the hfs.
        y: list of arrays
            Experimental y-data.
        yerr: list of arrays
            Experimental errors on y.
        no_of_points: int
            Number of points to use for the plot of the hfs.
        ax: matplotlib axes object
            If provided, plots on this axis
        show: Boolean
            if True, the plot will be shown at the end.

        Returns
        -------
        None"""

        if y is not None and not any([Y is None for Y in y]):
            yerr = [np.sqrt(Y) for Y in y]
            for i in range(len(yerr)):
                yerr[i] = np.where(yerr[i] == 0, 0, yerr[i])
        else:
            yerr = [None for i in self.spectra]
        return self.plot(x=x, y=y, yerr=yerr, no_of_points=no_of_points, ax=ax, show=show)

    ###########################
    #      MAGIC METHODS      #
    ###########################

    def __add__(self, other):
        if isinstance(other, CombinedSpectrum):
            return_object = CombinedSpectrum(self.spectra.extend(other.spectra))
        elif isinstance(other, Spectrum):
            return_object = CombinedSpectrum(self.spectra.append(other))
        return return_object

    def __call__(self, x):
        return np.hstack([s(X) for s, X in zip(self.spectra, x)])
