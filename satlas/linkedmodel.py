"""
Implementation of a class for the simultaneous fitting of hyperfine structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import copy

from . import lmfit as lm
from .basemodel import BaseModel
from .utilities import poisson_interval
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['LinkedModel']


class LinkedModel(BaseModel):

    """Links different models for simultaneous fitting."""

    def __init__(self, models):
        """Initializes the class for simultaneous fitting of different models.

        Parameters
        ----------
        models: list of :class:`.BaseModel` or :class:`.SingleSpectrum` objects
            A list defining the different models."""
        super(LinkedModel, self).__init__()
        self.models = models
        self.shared = ['Al',
                       'Au',
                       'Bl',
                       'Bu',
                       'Cl',
                       'Cu',
                       'Offset']

    def get_chisquare_mapping(self):
        return np.hstack([f.get_chisquare_mapping() for f in self.models])

    def get_lnprior_mapping(self, params):
        return sum([f.get_lnprior_mapping(params) for f in self.models])

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
                # If an expression is defined, replace the old names with the new ones
                if p[new_key].expr is not None:
                    for o_key in keys:
                        if o_key in p[new_key].expr:
                            n_key = 's' + str(i) + '_' + o_key
                            p[new_key].expr = p[new_key].expr.replace(o_key, n_key)
                # Link the shared parameters to the first subspectrum
                if any([shared in old_key for shared in self.shared]) and i > 0:
                    p[new_key].expr = 's0_' + old_key
                    p[new_key].vary = False
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

    def seperate_response(self, x):
        """Generates the response for each subspectrum.

        Parameters
        ----------
        x: list of arrays
            A list equal in length to the number of submodels,
            contains arrays for which the submodels have to be
            evaluated.

        Returns
        -------
        evaluated: ndarray
            The output array, of the same shape as the input
            list of arrays, containing the response values."""
        return np.squeeze([s.seperate_response(X)
                           for s, X in zip(self.models, x)])

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None,
             ax=None, show=True, plot_kws={}):
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
        ax: matplotlib axes object
            If provided, plots on this axis.
        show: boolean
            If True, the plot will be shown at the end.
        plot_kws: dictionary
            Dictionary containing the additional keyword arguments for the *plot*
            method of the underlying models. Note that the keyword *ax*
            is passed along correctly.

        Returns
        -------
        fig, ax: matplotlib figure and axis
            Figure and axis used for the plotting."""
        if ax is None:
            fig, ax = plt.subplots(len(self.models), 1)
            height = fig.get_figheight()
            width = fig.get_figwidth()
            fig.set_size_inches(width, len(self.models) * height, forward=True)
        else:
            fig = ax[0].get_figure()
        toReturn = fig, ax

        if x is None:
            x = [None] * len(self.models)
        if y is None:
            y = [None] * len(self.models)
        if yerr is None:
            yerr = [None] * len(self.models)

        selected = int(np.floor(len(self.models)/2 - 1))

        plot_kws_no_xlabel = copy.deepcopy(plot_kws)
        plot_kws_no_xlabel['xlabel'] = ''

        plot_kws_no_ylabel = copy.deepcopy(plot_kws)
        plot_kws_no_ylabel['ylabel'] = ''

        plot_kws_no_xlabel_no_ylabel = copy.deepcopy(plot_kws)
        plot_kws_no_xlabel_no_ylabel['xlabel'] = ''
        plot_kws_no_xlabel_no_ylabel['ylabel'] = ''

        for i, (X, Y, YERR, spec) in enumerate(zip(x, y, yerr,
                                                   self.models)):
            if i == selected:
                try:
                    spec.plot(x=X, y=Y, yerr=[YERR['low'], YERR['high']], ax=ax[i], show=False, plot_kws=plot_kws_no_xlabel)
                except:
                    spec.plot(x=X, y=Y, yerr=YERR, ax=ax[i], show=False, plot_kws=plot_kws_no_xlabel)
            elif i == len(self.models) - 1:
                try:
                    spec.plot(x=X, y=Y, yerr=[YERR['low'], YERR['high']], ax=ax[i], show=False, plot_kws=plot_kws_no_ylabel)
                except:
                    spec.plot(x=X, y=Y, yerr=YERR, ax=ax[i], show=False, plot_kws=plot_kws_no_ylabel)
            else:
                try:
                    spec.plot(x=X, y=Y, yerr=[YERR['low'], YERR['high']], ax=ax[i], show=False, plot_kws=plot_kws_no_xlabel_no_ylabel)
                except:
                    spec.plot(x=X, y=Y, yerr=YERR, ax=ax[i], show=False, plot_kws=plot_kws_no_xlabel_no_ylabel)

        # plt.tight_layout()
        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, x=None, y=None, plot_kws={}, **kwargs):
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
        plot_kws: dictionary
            Dictionary with keys to be passed on to :meth:`.plot`.

        Returns
        -------
        fig, ax: matplotlib figure and axis
            Figure and axis used for the plotting."""

        yerr = []
        if y is not None:
            for Y in y:
                ylow, yhigh = poisson_interval(Y)
                yerr.append({'low': Y - ylow, 'high': yhigh - Y})
        else:
            yerr = None
        return self.plot(x=x, y=y, yerr=yerr, plot_kws=plot_kws, **kwargs)

    ###########################
    #      MAGIC METHODS      #
    ###########################

    def __add__(self, other):
        """Adding another LinkedModel adds the models therein
        to the list of models, adding an IsomerSpectrum or SingleSpectrum
        adds that one spectrum to the list.

        Returns
        -------
        LinkedModel"""
        if isinstance(other, LinkedModel):
            return_object = LinkedModel(self.models.extend(other.models))
        else:
            return_object = super(LinkedModel, self).__add__(other)
        return return_object

    def __call__(self, x):
        """Pass the seperate frequency arrays to the submodels,
        and return their response values as a list of arrays.

        Parameters
        ----------
        x : list of floats or array_likes
            Frequency in MHz

        Returns
        -------
        list of floats or NumPy arrays
            Response of each spectrum for each seperate value in *x*."""
        return np.hstack([s(X) for s, X in zip(self.models, x)])
