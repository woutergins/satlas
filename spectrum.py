"""
.. module:: spectrum
    :platform: Windows
    :synopsis: Implementation of base class for the analysis of hyperfine
     structure spectra, including simultaneous fitting, various fitting
     routines and isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import emcee as mcmc
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import progressbar
except:
    pass
from . import loglikelihood as llh


__all__ = []

def model(params, spectrum, x, y, yerr, pearson):
    spectrum.params = params
    model = spectrum(x)
    if pearson:
        yerr = np.sqrt(model)
    return (y - model) / yerr


class Spectrum(object):

    """Abstract baseclass for all spectra, such as :class:`SingleSpectrum`,
    :class:`CombinedSpectrum` and :class:`IsomerSpectrum`. For input, see these
    classes.

    Attributes
    ----------
    selected: list of strings
        When a walk is performed and a triangle plot is created from the data,
        the parameters with one of these strings in their name will be
        displayed. Defaults to the hyperfine parameters and the centroid.
    loglikelifunc: {'Poisson', 'Gaussian'}
        Indicates if the Maximum Likelihood Estimation uses Poissonian or
        Gaussian formulas for the loglikelihood. Defaults to 'Poisson'.
    """

    def __init__(self):
        super(Spectrum, self).__init__()
        self.selected = ['Al', 'Au', 'Bl', 'Bu', 'Cl', 'Cu', 'Centroid']

    def sanitize_input(self, x, y, yerr=None):
        raise NotImplemented

    def generate_walks(self):
        """If the result of walks has been stored, plot them as
        seperate lines for each parameter. Raises a KeyError if no walks
        have been performed yet.

        Returns
        -------
        figure, axes
            Returns a new figure and axes containing the plot of the
            random walks.
        Raises
        ------
        KeyError
            When this function is called without walks being saved."""
        if self.walks is not None:
            var_names, _, _ = self.vars()
            shape = int(np.ceil(np.sqrt(len(var_names))))
            figWalks, axes = plt.subplots(shape, shape, sharex=True)
            axes = axes.flatten()

            for (n, values), a in zip(self.walks):
                a.plot(values, 'k', alpha=0.4)
                a.set_xlim([0, len(values)])
                a.set_ylabel(n)
            return figWalks, axes
        else:
            raise KeyError("No instance of 'walks' found!")

    def generate_likelihood(self, x, y):
        """Given the data x and y, generate approximate likelihood functions
        for all parameters.

        Parameters
        ----------
        x, y: array_like
            The frequency (x) and counts (y) in the data.

        Returns
        -------
        data: dict of dicts
            A dictionary containing the x and y values for the likelihood
            functions for each variable. The first dictionary has all the
            variable names as keys, the second dictionary has 'x' and 'y'
            as keys."""
        params = self.params
        var_names = []
        vars = []
        for key in params.keys():
            if params[key].vary:
                var_names.append(key)
                vars.append(params[key].value)
        x, y, _ = self.sanitize_input(x, y)

        def lnprobList(fvars, x, y, groupParams):
            for val, n in zip(fvars, var_names):
                groupParams[n].value = val
            return self.lnprob(groupParams, x, y)
        groupParams = lm.Parameters()
        for key in params.keys():
            groupParams[key] = PriorParameter(key,
                                              value=params[key].value,
                                              vary=params[key].vary,
                                              expr=params[key].expr,
                                              priormin=params[key].min,
                                              priormax=params[key].max)
        data = {}
        for i, n in enumerate(var_names):
            best = self.mle_fit[n].value
            std = self.mle_fit[n].stderr
            left, right = (best - 5 * std, best + 5 * std)
            xvalues = np.linspace(left, right, 1000)
            dummy = np.array(vars, dtype='float')
            yvalues = np.zeros(xvalues.shape[0])
            for j, value in enumerate(xvalues):
                dummy[i] = value
                yvalues[j] = lnprobList(dummy, x, y, groupParams)
            data[n] = {}
            data[n]['x'] = xvalues
            data[n]['y'] = yvalues
            self.params = self.mle_fit
        return data

    def display_mle_fit(self, **kwargs):
        """Give a readable overview of the result of the MLE fitting routine.

        Warning
        -------
        The uncertainty shown is the largest of the asymmetrical errors! Work
        is being done to incorporate asymmetrical errors in the report; for
        now, rely on the correlation plot."""
        if hasattr(self, 'mle_fit'):
            if 'show_correl' not in kwargs:
                kwargs['show_correl'] = False
            print(lm.fit_report(self.mle_fit, **kwargs))
        else:
            print('Spectrum has not yet been fitted with this method!')

    def display_chisquare_fit(self, **kwargs):
        """Display all relevent info of the least-squares fitting routine,
        if this has been performed.

        Parameters
        ----------
        kwargs: misc
            Keywords passed on to :func:`fit_report` from the LMFit package."""
        if hasattr(self, 'chisq_res_par'):
            print('Scaled errors estimated from covariance matrix.')
            print('NDoF: {:d}, Chisquare: {:.3G}, Reduced Chisquare: {:.3G}'.format(self.ndof, self.chisqr, self.redchi))
            print(lm.fit_report(self.chisq_res_par, **kwargs))
        else:
            print('Spectrum has not yet been fitted with this method!')

    def vars(self, selection='any'):
        """Return the variable names, values and estimated error bars for the
        parameters.

        Parameters
        ----------
        selection: string, optional
            Selects if the chisquare ('chisquare' or 'any') or MLE values are
            used. Defaults to 'any'.

        Returns
        -------
        names, values, uncertainties: tuple of lists
            Returns a 3-tuple of lists containing the names of the parameters,
            the values and the estimated uncertainties."""
        var, var_names, varerr = [], [], []
        if hasattr(self, 'chisq_res_par') and (selection.lower() == 'chisquare'
                                               or selection.lower() == 'any'):
            for key in sorted(self.chisq_res_par.keys()):
                if self.chisq_res_par[key].vary:
                    var.append(self.chisq_res_par[key].value)
                    var_names.append(self.chisq_res_par[key].name)
                    varerr.append(self.chisq_res_par[key].stderr)
        elif hasattr(self, 'mle_fit'):
            for key in sorted(self.mle_fit.params.keys()):
                if self.mle_fit.params[key].vary:
                    var.append(self.mle_fit.params[key].value)
                    var_names.append(self.mle_fit.params[key].name)
                    varerr.append(self.mle_fit.params[key].stderr)
        else:
            params = self.params
            for key in sorted(params.keys()):
                if params[key].vary:
                    var.append(params[key].value)
                    var_names.append(params[key].name)
                    varerr.append(None)
        return var_names, var, varerr

    def get_result_frame(self, method='chisquare',
                         selected=False, bounds=False,
                         vary=False):
        """Returns the data from the fit in a pandas DataFrame.

        Parameters
        ----------
        method: str, optional
            Selects which fitresults have to be loaded. Can be 'chisquare' or
            'mle'. Defaults to 'chisquare'.
        selected: boolean, optional
            Selects if only the parameters in :attr:`selected` have to be
            given or not. Defaults to :attr:`False`.
        bounds: boolean, optional
            Selects if the boundary also has to be given. Defaults to
            :attr:`False`.
        vary: boolean, optional
            Selects if only the parameters that have been varied have to
            be supplied. Defaults to :attr:`False`.

        Returns
        -------
        DataFrame"""
        if method.lower() == 'chisquare':
            values = self.chisq_res_par.values()
        elif method.lower() == 'mle':
            values = self.mle_fit.values()
        else:
            raise KeyError
        if selected:
            values = [v for n in self.selected for v in values if n in v.name]
        if vary:
            values = [v for v in values if v.vary]
        if bounds:
            ind = ['Value', 'Uncertainty', 'Upper Bound', 'Lower Bound']
            data = np.array([[p.value, p.stderr, p.max, p.min] for p in values]).flatten()
            columns = [[p.name for pair in zip(values, values) for p in pair],
                       [x for p in values for x in ind]]
        else:
            data = np.array([[p.value, p.stderr] for p in values]).flatten()
            ind = ['Value', 'Uncertainty']
            columns = [[p.name for pair in zip(values, values) for p in pair],
                       [x for p in values for x in ind]]
        columns = pd.MultiIndex.from_tuples(list(zip(*columns)))
        result = pd.DataFrame(data, index=columns).T
        return result
