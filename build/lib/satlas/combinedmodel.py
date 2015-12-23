"""
Implementation of a class for the simultaneous fitting of hyperfine structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import copy
from .basemodel import BaseModel
from .utilities import poisson_interval
__all__ = ['CombinedModel']


class CombinedModel(BaseModel):

    """Combines different models for simultaneous fitting."""

    def __init__(self, models):
        """Initializes the class for simultaneous fitting of different models.

        Parameters
        ----------
        models: list of :class:`.BaseModel` or :class:`.SingleSpectrum` objects
            A list defining the different models."""
        super(CombinedModel, self).__init__()
        self.models = models
        self.shared = ['Al',
                       'Au',
                       'Bl',
                       'Bu',
                       'Cl',
                       'Cu',
                       'Offset']

    def lnprior(self):
        return_value = 0
        for i, spec in enumerate(self.models):
            return_value += spec.lnprior()
        return return_value

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
                    par[new_key] = lm.Parameter(new_key,value=params[key].value,
                                             min=params[key].min,
                                             max=params[key].max)
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
            fig, ax = plt.subplots(len(self.models), 1, sharex=True)
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
        for i, (X, Y, YERR, spec) in enumerate(zip(x, y, yerr,
                                                   self.models)):
            if i == selected:
                try:
                    spec.plot(x=X, y=Y, yerr=[YERR['low'], YERR['high']], no_of_points=no_of_points, ax=ax[i], show=False,
                              data_legend=data_legend, legend=legend, xlabel='')
                except:
                    spec.plot(x=X, y=Y, yerr=YERR, no_of_points=no_of_points, ax=ax[i], show=False,
                              data_legend=data_legend, legend=legend, xlabel='')
            elif i == len(self.models) - 1:
                try:
                    spec.plot(x=X, y=Y, yerr=[YERR['low'], YERR['high']], no_of_points=no_of_points, ax=ax[i], show=False, ylabel='')
                except:
                    spec.plot(x=X, y=Y, yerr=YERR, no_of_points=no_of_points, ax=ax[i], show=False, ylabel='')
            else:
                try:
                    spec.plot(x=X, y=Y, yerr=[YERR['low'], YERR['high']], no_of_points=no_of_points, ax=ax[i], show=False, xlabel='', ylabel='')
                except:
                    spec.plot(x=X, y=Y, yerr=YERR, no_of_points=no_of_points, ax=ax[i], show=False, xlabel='', ylabel='')

        plt.tight_layout()
        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, x, y, plot_kws={}):
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
        for Y in y:
            ylow, yhigh = poisson_interval(Y)
            yerr.append({'low': Y - ylow, 'high': yhigh - Y})
        plot_kws['x'] = x
        plot_kws['y'] = y
        plot_kws['yerr'] = yerr
        return self.plot(**plot_kws)

    ###########################
    #      MAGIC METHODS      #
    ###########################

    def __add__(self, other):
        """Adding another CombinedModel adds the models therein
        to the list of models, adding an IsomerSpectrum or SingleSpectrum
        adds that one spectrum to the list.

        Returns
        -------
        CombinedModel"""
        if isinstance(other, CombinedModel):
            return_object = CombinedModel(self.models.extend(other.models))
        elif isinstance(other, Spectrum):
            return_object = CombinedModel(self.models.append(other))
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
        return np.array([s(X) for s, X in zip(self.models, x)])
