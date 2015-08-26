"""
.. module:: CombinedSpectrum
    :platform: Windows
    :synopsis: Implementation of classes for the analysis of hyperfine
     structure spectra with isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import numpy as np
import matplotlib.pyplot as plt
from .combinedspectrum import CombinedSpectrum
import lmfit
import copy

__all__ = ['IsomerSpectrum']


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

    @property
    def params(self):
        params = lmfit.Parameters()
        for i, s in enumerate(self.spectra):
            p = copy.deepcopy(s.params)
            keys = list(p.keys())
            for old_key in keys:
                new_key = 's' + str(i) + '_' + old_key
                p[new_key] = p.pop(old_key)
                for o_key in keys:
                    if p[new_key].expr is not None:
                        n_key = 's' + str(i) + '_' + o_key
                        p[new_key].expr = p[new_key].expr.replace(o_key, n_key)
                if any([shared in old_key for shared in self.shared]) and i > 0:
                    p[new_key].expr = 's0_' + old_key
                    p[new_key].vary = False
                if i > 0 and 'Background' in new_key:
                    p[new_key].value = 0
                    p[new_key].vary = False
                    p[new_key].expr = None
            params += p
        return params

    @params.setter
    def params(self, params):
        for i, spec in enumerate(self.spectra):
            par = lmfit.Parameters()
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
        return [s(x) - s.params['Background'].value for s in self.spectra]

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerrs=None,
             no_of_points=10**4, ax=None, label=False,
             show=True):
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
        ax: matplotlib axes object
            If provided, plots on this axis
        show: Boolean
            if True, the plot will be shown at the end.
        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            toReturn = fig, ax
        else:
            toReturn = None

        if x is None:
            ranges = []

            fwhm = max([s.fwhm for s in self.spectra])

            for pos in [positions for spectrum in self.spectra for positions in spectrum.mu]:
                r = np.linspace(pos - 4 * fwhm,
                                pos + 4 * fwhm,
                                2 * 10**2)
                ranges.append(r)
            superx = np.sort(np.concatenate(ranges))

        else:
            superx = np.linspace(x.min(), x.max(), no_of_points)

        if x is not None and y is not None:
            ax.errorbar(x, y, yerrs, fmt='o', markersize=3)
        resp = self.seperate_response(superx)
        for i, r in enumerate(resp):
            ax.plot(superx, r, lw=3.0, label='I=' + str(self.spectra[i].I))
        ax.plot(superx, self(superx), lw=3.0, label='Total')

        if label:
            ax.set_xlabel('Frequency (MHz)', fontsize=16)
            ax.set_ylabel('Counts', fontsize=16)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            return toReturn

    def plot_spectroscopic(self, xs=None, ys=None,
                           no_of_points=10**4, ax=None,show = True):
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

        if ys is not None:
            yerrs = np.sqrt(ys + 1)
        else:
            yerrs = None
        self.plot(xs, ys, yerrs, no_of_points, ax)

    def __add__(self, other):
        if isinstance(other, IsomerSpectrum):
            spectra = self.spectra + other.spectra
            return IsomerSpectrum(spectra)
        else:
            try:
                return other.__add__(self)
            except:
                raise TypeError('unsupported operand type(s)')

    def __call__(self, x):
        return np.sum([s(x) for s in self.spectra], axis=0)
