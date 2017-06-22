"""
Implementation of a class for the simultaneous fitting of hyperfine structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@kuleuven.be>
"""
import copy

import lmfit as lm
from satlas.models.basemodel import BaseModel
from satlas.utilities import poisson_interval
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['LinkedModel']


class LinkedModel(BaseModel):

    """Links different models for simultaneous fitting."""

    def __init__(self, models):
        """Initializes the class for simultaneous fitting of different models.

        Parameters
        ----------
        models: list of :class:`.BaseModel` children objects
            A list defining the different models."""
        super(LinkedModel, self).__init__()
        self.models = models
        for i, model in enumerate(self.models):
            model._add_prefix('s' + str(i) + '_')
        self._set_params()
        self.shared = []

    def _set_params(self):
        for model in self.models:
            try:
                p.add_many(*model.params.values())
            except:
                p = model.params.copy()
        self.params = p

    def _add_prefix(self, value):
        for model in self.models:
            model._add_prefix(value)
        self._set_params()

    def get_chisquare_mapping(self):
        return np.hstack([f.get_chisquare_mapping() for f in self.models])

    def get_lnprior_mapping(self, params):
        return sum([f.get_lnprior_mapping(f.params) for f in self.models])

    @property
    def shared(self):
        """Contains all parameters which share the same value among all models."""
        return self._shared

    @shared.setter
    def shared(self, value):
        params = self.params.copy()
        self._shared = value
        for name in self._shared:
            selected_list = [p for p in params.keys() if name in p]
            try:
                selected_name = selected_list[0]
                for p in selected_list[1:]:
                    params[p].expr = selected_name
            except IndexError:
                pass
        self.params = params

    @property
    def params(self):
        """Instance of lmfit.Parameters object characterizing the
        shape of the HFS."""
        return self._parameters

    @params.setter
    def params(self, params):
        self._parameters = params.copy()
        for spec in self.models:
            spec.params = self._parameters.copy()
            spec._parameters._prefix = spec._prefix

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
        return np.squeeze([s(X)
                           for s, X in zip(self.models, x)])

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None,
             ax=None, show=True, plot_kws={}, linked=True):
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
            If provided, plots on these axes.
        show: boolean
            If True, the plot will be shown at the end.
        plot_kws: dictionary
            Dictionary containing the additional keyword arguments for the *plot*
            method of the underlying models. Note that the keyword *ax*
            is passed along correctly.
        linked: boolean, optional
            If True, the x-axes of the generated plots will be linked.

        Returns
        -------
        fig, ax: matplotlib figure and axis
            Figure and axis used for the plotting."""
        if ax is None:
            fig, ax = plt.subplots(len(self.models), 1, sharex=linked)
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
        to the list of models.

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
