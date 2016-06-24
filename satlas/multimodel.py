"""
Implementation of a class for the analysis of hyperfine structure spectra with isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@kuleuven.be>
"""
import copy
import warnings

from . import lmfit as lm
from .basemodel import BaseModel
from .utilities import poisson_interval
import matplotlib.pyplot as plt
import numpy as np


__all__ = ['MultiModel']
warn_msg = """Use of the class MultiModel has been deprecated and will be removed in further updates. Please use the SumModel class in the future."""


class MultiModel(BaseModel):

    """Create a spectrum containing the information of multiple hyperfine
    structures."""

    def __init__(self, models):
        """Initializes the HFS by providing a list of :class:`.HFSModel`
        objects.

        Parameters
        ----------
        models: list of :class:`.HFSModel` instances
            A list containing the models."""
        warnings.warn(warn_msg)
        super(MultiModel, self).__init__()
        self.models = models
        self.shared = []

    def get_chisquare_mapping(self):
        return np.hstack([f.get_chisquare_mapping() for f in self.models])

    def get_lnprior_mapping(self):
        return sum([f.get_lnprior_mapping() for f in self.models])

    @property
    def shared(self):
        """Contains all parameters which share the same value among all models."""
        return self._shared

    @shared.setter
    def shared(self, value):
        self._shared = value

    @property
    def params(self):
        """Instance of lmfit.Parameters object characterizing the
        shape of the HFS."""
        params = lm.Parameters()
        for i, s in enumerate(self.models):
            p = copy.deepcopy(s.params)
            keys = list(p.keys())
            for old_key in keys:
                new_key = 's' + str(i) + '_' + old_key
                p[new_key] = p.pop(old_key)
                if p[new_key].expr is not None:
                    for o_key in keys:
                        if o_key in p[new_key].expr:
                            n_key = 's' + str(i) + '_' + o_key
                            p[new_key].expr = p[new_key].expr.replace(o_key, n_key)
                if any([shared in old_key for shared in self.shared]) and i > 0:
                    p[new_key].expr = 's0_' + old_key
                    p[new_key].vary = False
                if i > 0 and 'Background' in new_key:
                    p[new_key].value = 0
                    p[new_key].vary = False
                    p[new_key].expr = None
                if new_key in self._expr.keys():
                    p[new_key].expr = self._expr[new_key]
            params += p
        return params

    @params.setter
    def params(self, params):
        for i, spec in enumerate(self.models):
            par = lm.Parameters()
            for key in params:
                if key.startswith('s'+str(i)+'_'):
                    new_key = key[len('s'+str(i)+'_'):]
                    expr = params[key].expr
                    if expr is not None:
                        for k in params:
                            nk = k[len('s'+str(i)+'_'):]
                            expr = expr.replace(k, nk)
                    par[new_key] = lm.Parameter(new_key,
                                                value=params[key].value,
                                                min=params[key].min,
                                                max=params[key].max,
                                                vary=params[key].vary,
                                                expr=expr)
                    par[new_key].stderr = params[key].stderr
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
            Seperate responses of models to the input *x*."""
        background_vals = [np.polyval([s.params[par_name].value for par_name in s.params if par_name.startswith('Background')], x) for s in self.models]
        back = self.models[0].params['Background'].value if background else 0
        return [s(x) - b + back for s, b in zip(self.models, background_vals)]

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None, ax=None,
             plot_seperate=True, no_of_points=10**3, show=True,
             legend=None, data_legend=None, xlabel='Frequency (MHz)', ylabel='Counts',
             indicate=False, model=False, colormap='bone_r',
             normalized=False, distance=4):
        """Routine that plots the hfs of all the models,
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
        plot_seperate: boolean, optional
            Controls if the underlying models are drawn as well, or only
            the sum. Defaults to False.
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
        xlabel: string, optional
            If given, sets the xlabel to this string. Defaults to 'Frequency (MHz)'.
        ylabel: string, optional
            If given, sets the ylabel to this string. Defaults to 'Counts'.
        indicate: boolean, optional
            If set to True, dashed lines are drawn to indicate the location of the
            transitions, and the labels are attached. Defaults to False.
        model: boolean, optional
            If given, the region around the fitted line will be shaded, with
            the luminosity indicating the pmf of the Poisson
            distribution characterized by the value of the fit. Note that
            the argument *yerr* is ignored if *model* is True.
        normalized: Boolean
            If True, the data and fit are plotted normalized such that the highest
            data point is one.
        distance: float, optional
            Controls how many FWHM deviations are used to generate the plot.
            Defaults to 4.

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

            fwhm = max([p.fwhm for s in self.models for p in s.parts])

            for pos in [l for spectrum in self.models for l in spectrum.locations]:
                r = np.linspace(pos - distance * fwhm,
                                pos + distance * fwhm,
                                2 * 10**2)
                ranges.append(r)
            superx = np.sort(np.concatenate(ranges))

        else:
            superx = np.linspace(x.min(), x.max(), no_of_points)

        if normalized:
            norm = np.max(y)
        else:
            norm = 1

        if x is not None and y is not None:
            if not model:
                try:
                    ax.errorbar(x, y, yerr=[y - yerr['low'], yerr['high'] - y], fmt='o', label=data_legend)
                except:
                    ax.errorbar(x, y, yerr=yerr, fmt='o', label=data_legend)
            else:
                ax.plot(x, y, 'o')
        resp = self.seperate_response(superx)

        if plot_seperate:
            for i, r in enumerate(resp):
                line, = ax.plot(superx, r, label='I=' + str(self.models[i].I))
                if indicate:
                    for l, lab in zip(self.models[i].locations, self.models[i].ftof):
                        lab = lab.split('__')
                        lab = lab[0] + '$\\rightarrow$' + lab[1]
                        ax.annotate(lab, xy=(l, self.models[i](l)/norm), rotation=90, color=line.get_color(),
                                    weight='bold', size=14, ha='center')
        if model:
            range = (self.locations.min(), self.locations.max())
            max_counts = np.ceil(-optimize.brute(lambda x: -self(x), (range,), full_output=True, Ns=1000, finish=optimize.fmin)[1])
            min_counts = [self._params[par_name].value for par_name in self._params if par_name.startswith('Background')][-1]
            min_counts = np.floor(max(0, min_counts - 3 * min_counts ** 0.5))
            y = np.arange(min_counts, max_counts + 3 * max_counts ** 0.5 + 1)
            x, y = np.meshgrid(superx, y)
            from scipy import stats
            z = stats.poisson(self(x) / norm).pmf(y)

            z = z / z.sum(axis=0)
            ax.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), cmap=plt.get_cmap(colormap))
            line, = ax.plot(superx, self(superx) / norm, label=legend, lw=0.5)
        else:
            line, = ax.plot(superx, self(superx)/norm, label=legend)
        if indicate:
            for (p, l) in zip(self.locations, self.ftof):
                height = self(p)
                lab = l.split('__')
                lableft = '/'.join(lab[0].split('_'))
                labright = '/'.join(lab[1].split('_'))
                lab = '$' + lableft + '\\rightarrow' + labright + '$'
                ax.annotate(lab, xy=(p, height), rotation=90, color=line.get_color(),
                            weight='bold', size=14, ha='center', va='bottom')
                ax.axvline(p, linewidth=0.5, linestyle='--')
        ax.set_xlim(superx.min(), superx.max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, **kwargs):
        """Routine that plots the hfs of all the models, possibly on
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
        """Adding an MultiModel results in a new MultiModel
        with the new spectrum added.

        Returns
        -------
        MultiModel"""
        if isinstance(other, MultiModel):
            models = self.models + other.models
            return MultiModel(models)
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
        return np.sum([s(x) for s in self.models], axis=0)
