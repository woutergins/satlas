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

    def plot(self, xs=None, ys=None, yerrs=None,
             no_of_points=10**4, ax=None, show=True,
             ylabel='Counts', xlabel='Frequency (MHz)'):
        """Routine that plots the hfs of all the spectra,
        possibly on top of experimental data.

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
        ax: list of matplotlib axes object
            If provided, plots on these axes.
        show: Boolean
            if True, the plot will be shown at the end.

        Returns
        -------
        fig, ax: tuple
            Returns a tuple containing the figure and axes which were
            used for the plotting.
        """
        if ax is None:
            fig, ax = plt.subplots(len(self.spectra), 1, sharex=True)
        else:
            fig = ax[0].get_figure()
        toReturn = fig, ax

        if xs is None:
            xs = [None] * len(self.spectra)
        if ys is None:
            ys = [None] * len(self.spectra)
        if yerrs is None:
            yerrs = [None] * len(self.spectra)

        for i, (x, y, yerr, spec) in enumerate(zip(xs, ys, yerrs,
                                                   self.spectra)):
            if x is not None and y is not None:
                ax[i].errorbar(x, y, yerr, fmt='o')
            spec.plot(x, y, yerr, no_of_points, ax[i], show=False)
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

        ax[len(xs)-1].set_xlabel('Frequency (MHz)')
        ax[0].set_ylabel('Counts')

        plt.tight_layout()
        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, xs=None, ys=None,
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

        if ys is not None and not any([y is None for y in ys]):
            yerrs = [np.sqrt(y) for y in ys]
            for i in range(len(yerrs)):
                yerrs[i] = np.where(yerrs[i] == 0, 0, yerrs[i])
        else:
            yerrs = [None for i in self.spectra]
        return self.plot(xs, ys, yerrs, no_of_points, ax, show)

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
