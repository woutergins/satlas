"""
Implementation of a class for the analysis of hyperfine structure spectra with isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import numpy as np
import matplotlib.pyplot as plt
from .combinedspectrum import CombinedSpectrum
from .utilities import poisson_interval
import lmfit
import copy

__all__ = ['IsomerSpectrum']


class IsomerSpectrum(CombinedSpectrum):

    """Create a spectrum containing the information of multiple hyperfine
    structures."""

    def __init__(self, spectra):
        """Initializes the HFS by providing a list of :class:`.SingleSpectrum`
        objects.

        Parameters
        ----------
        spectra: list of :class:`.SingleSpectrum` instances
            A list containing the base spectra."""
        super(IsomerSpectrum, self).__init__(spectra)
        self.shared = []

    def _sanitize_input(self, x, y, yerr=None):
        x, y = np.array(x), np.array(y)
        if yerr is not None:
            yerr = np.array(yerr)
        return x, y, yerr

    @property
    def shared(self):
        """Contains all parameters which share the same value among all spectra."""
        return self._shared

    @shared.setter
    def shared(self, value):
        self._shared = value

    @property
    def params(self):
        """Instance of lmfit.Parameters object characterizing the
        shape of the HFS."""
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

    def seperate_response(self, x, background=False):
        """Get the response for each seperate spectrum for the values *x*,
        without background.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz.

        Other parameters
        ----------------
        background: boolean
            If True, each spectrum has the same background. If False,
            the background of each spectrum is assumed to be 0.

        Returns
        -------
        list of floats or NumPy arrays
            Seperate responses of spectra to the input *x*."""
        back = self.spectra[0].params['Background'].value if background else 0
        return [s(x) - s.params['Background'].value + back  for s in self.spectra]

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None,
             no_of_points=10**4, ax=None,
             show=True, xlabel='Frequency (MHz)',
             ylabel='Counts', data_legend='Data',
             indicate=False):
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
        show: boolean
            If True, the plot will be shown at the end.
        xlabel: string
            String to display on the x-axis.
        ylabel: string
            String to display on the y-axis.
        data_legend: string
            String to use as the legend for the data.
        indicate: boolean
            If True, the peaks will be marked with
            the transition.

        Returns
        -------
        fig, ax: matplotlib figure and axes"""
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            toReturn = fig, ax
        else:
            toReturn = None

        if x is None:
            ranges = []

            fwhm = max([p.fwhm for s in self.spectra for p in s.parts])

            for pos in [l for spectrum in self.spectra for l in spectrum.locations]:
                r = np.linspace(pos - 4 * fwhm,
                                pos + 4 * fwhm,
                                2 * 10**2)
                ranges.append(r)
            superx = np.sort(np.concatenate(ranges))

        else:
            superx = np.linspace(x.min(), x.max(), no_of_points)

        if x is not None and y is not None:
            try:
                ax.errorbar(x, y, yerr=[y - yerr['low'], yerr['high'] - y], fmt='o', label=data_legend)
            except:
                ax.errorbar(x, y, yerr=yerr, fmt='o', label=data_legend)
        resp = self.seperate_response(superx)

        for i, r in enumerate(resp):
            line, = ax.plot(superx, r, label='I=' + str(self.spectra[i].I))
            if indicate:
                for l, lab in zip(self.spectra[i].locations, self.spectra[i].ftof):
                    lab = lab.split('__')
                    lab = lab[0] + '$\\rightarrow$' + lab[1]
                    ax.annotate(lab, xy=(l, self.spectra[i](l)), rotation=90, color=line.get_color(),
                                weight='bold', size=14, ha='center')
        ax.plot(superx, self(superx), label='Total')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend(loc=0)

        plt.tight_layout()
        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, **kwargs):
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
        fig, ax: matplotlib figure and axes"""

        y = kwargs.get('y', None)
        if y is not None:
            ylow, yhigh = poisson_interval(y)
            yerr = {'low': ylow, 'high': yhigh}
        else:
            yerr = None
        kwargs['yerr'] = yerr
        return self.plot(**kwargs)

    def __add__(self, other):
        """Adding an IsomerSpectrum results in a new IsomerSpectrum
        with the new spectrum added.

        Returns
        -------
        IsomerSpectrum"""
        if isinstance(other, IsomerSpectrum):
            spectra = self.spectra + other.spectra
            return IsomerSpectrum(spectra)
        else:
            try:
                return other.__add__(self)
            except:
                raise TypeError('unsupported operand type(s)')

    def __call__(self, x):
        """Get the response for frequency *x* (in MHz) of the spectrum.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz

        Returns
        -------
        float or NumPy array
            Response of the spectrum for each value of *x*."""
        return np.sum([s(x) for s in self.spectra], axis=0)
