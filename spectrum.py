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


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper


def model(params, spectrum, x, y, yerr, pearson):
    spectrum.var_from_params(params)
    model = spectrum(x)
    if pearson:
        yerr = np.sqrt(model)
    return (y - model) / yerr


class PriorParameter(lm.Parameter):

    """Extended the Parameter class from LMFIT to incorporate prior boundaries.
    """

    def __init__(self, name, value=None, vary=True, min=None, max=None,
                 expr=None, priormin=None, priormax=None):
        super(PriorParameter, self).__init__(name, value=value,
                                             vary=vary, min=min,
                                             max=max, expr=expr)
        self.priormin = priormin
        self.priormax = priormax


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
        self.selected = ['Al', 'Au', 'Bl', 'Bu', 'Cl', 'Cu', 'df']
        self.atol = 0.1
        self.loglikelifunc = 'poisson'
        self._theta_array = np.linspace(-3, 3, 1000)

    @property
    def loglikelifunc(self):
        mapping = {llh.Poisson: 'Poisson', llh.Gaussian: 'Gaussian'}
        return mapping[self._loglikelifunc]

    @loglikelifunc.setter
    def loglikelifunc(self, value):
        mapping = {'poisson': llh.Poisson, 'gaussian': llh.Gaussian}
        self._loglikelifunc = mapping.get(value.lower(), llh.Poisson)

        def x_err_calculation(x, y, s):
            x, theta = np.meshgrid(x, self._theta_array)
            y, _ = np.meshgrid(y, self._theta_array)
            p = self._loglikelifunc(y, self(x + theta))
            g = np.exp(-(theta / s)**2 / 2) / s
            return np.log(np.fft.irfft(np.fft.rfft(p) * np.fft.rfft(g))[:, -1])
        self._loglikelifunc_xerr = x_err_calculation

    def sanitize_input(self, x, y, yerr=None):
        raise NotImplemented

    ##########################################
    # MAXIMUM LIKELIHOOD ESTIMATION ROUTINES #
    ##########################################
    def loglikelihood(self, params, x, y):
        """Returns the total loglikelihood for the given parameter
        dictionary 'params'. Uses the function defined in the attribute
        :attr:`loglikelifunc` to calculate the loglikelihood. Defaultly,
        this is set to a Poisson distribution.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters for which the loglikelihood has to be
            determined.
        x: array_like
            Frequencies in MHz.
        y: array_like
            Counts corresponding to :attr:`x`."""
        self.var_from_params(params)
        if any([np.isclose(X.min(), X.max(), atol=self.atol)
                for X in self.seperate_response(x)]) or any(self(x) < 0):
            return -np.inf
        if params['sigma_x'].value > 0:
            # integrate for each datapoint over a range
            s = params['sigma_x'].value

            return_value = self._loglikelifunc_xerr(x, y, s)
        else:
            return_value = self._loglikelifunc(y, self(x))
        return return_value

    def lnprior(self, params):
        """Defines the (uninformative) prior for all parameters.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters with values to be used in the fit/walk

        Returns
        -------
        float
            If any of the parameters are out of bounds, returns
            :data:`-np.inf`, otherwise 1.0 is returned."""
        for key in params.keys():
            try:
                leftbound, rightbound = (params[key].priormin,
                                         params[key].priormax)
            except:
                leftbound, rightbound = params[key].min, params[key].max
            leftbound = -np.inf if leftbound is None else leftbound
            rightbound = np.inf if rightbound is None else rightbound
            if not leftbound <= params[key].value <= rightbound:
                return -np.inf
        return 1.0

    def lnprob(self, params, x, y):
        """Calculates the sum of the loglikelihoods given the parameters
        :attr:`params`, while also checking the prior first. If this prior
        rejects the parameters, the parameters are not set for the spectrum.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters for which the sum loglikelihood has to be
            calculated.
        x: array_like
            Frequencies in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.

        Returns
        -------
        float
            Sum of the loglikelihoods plus the result of the prior."""
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        res = lp + np.sum(self.loglikelihood(params, x, y))
        return res

    def likelihood_fit(self, x, y, xerr=0, vary_sigma=False, walking=True, **kwargs):
        """Fit the spectrum to the spectroscopic data using the Maximum
        Likelihood technique. This is done by minimizing the negative sum of
        the loglikelihoods of the spectrum given the data (given by the method
        :meth:`lnprob`). Prints a statement regarding the success of the
        fitting.

        Parameters
        ----------
        x: array_like
            Frequency of the data, in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.
        walking: Boolean
            Determines if a Monte Carlo walk is performed after the
            minimization to determine the errorbars and distribution of the
            parameters.
        kwargs: misc
            Keyword arguments passed on to the method :meth:`walk`.

        Returns
        -------
        tuple or :class:`None`
            If any kind of plot is requested, a tuple containing these figures
            will be returned (see :meth:`walk` for more details). If no plot
            is requested, returns the value :class:`None`."""

        def negativeloglikelihood(*args, **kwargs):
            return -self.lnprob(*args, **kwargs)

        x, y, _ = self.sanitize_input(x, y)
        params = self.params_from_var()
        params.add('sigma_x', value=xerr, vary=vary_sigma, min=0)
        result = lm.Minimizer(negativeloglikelihood, params, fcn_args=(x, y))
        result.scalar_minimize(method='Nelder-Mead')
        self.var_from_params(result.params)
        self.mle_fit = result.params
        self.mle_result = result.message

        if walking:
            return self.likelihood_walk(x, y, **kwargs)
        else:
            return None

    def likelihood_walk(self, x, y, nsteps=2000, walkers=20, burnin=10.0,
                        verbose=True, store_walks=False):
        """Performs a random walk in the parameter space to determine the
        distribution for the best fit of the parameters.

        A message is printed before and after the walk.

        Warning
        -------
        The errors calculated can be asymmetrical, but only the largest is
        saved as the overal uncertainty. This is a known issue, and work is
        being done to resolve this.

        Parameters
        ----------
        x: array_like
            Frequency of the data, in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.
        nsteps: int, optional
            Number of steps to be taken, defaults to 2000.
        walkers: int, optional
            Number of walkers to be used, defaults to 20.
        burnin: float, optional
            Burn-in to be used for the walk. Expressed in percentage,
            defaults to 10.0.
        verbose: boolean, optional
            Controls printing the status of the sampling to the stdout.
            Defaults to True.
        store_walks: boolean, optional
            For deeper debugging, the data from the walks can be saved and
            viewed later on."""

        params = self.params_from_var()
        self.mle_fit = self.params_from_var()
        var_names = []
        vars = []
        for key in params.keys():
            if params[key].vary:
                var_names.append(key)
                vars.append(params[key].value)
        ndim = len(vars)
        pos = mcmc.utils.sample_ball(vars, [1e-4] * len(vars), size=walkers)
        x, y, _ = self.sanitize_input(x, y)

        if verbose:
            try:
                widgets = ['Walk:', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker=progressbar.RotatingMarker()),
                           ' ', progressbar.AdaptiveETA()]
                pbar = progressbar.ProgressBar(widgets=widgets,
                                               maxval=walkers * nsteps).start()
            except:
                pass

        def lnprobList(fvars, x, y, groupParams, pbar):
            for val, n in zip(fvars, var_names):
                groupParams[n].value = val
            try:
                pbar += 1
            except:
                pass
            return self.lnprob(groupParams, x, y)
        groupParams = lm.Parameters()
        for key in params.keys():
            groupParams[key] = PriorParameter(key,
                                              value=params[key].value,
                                              vary=params[key].vary,
                                              expr=params[key].expr,
                                              priormin=params[key].min,
                                              priormax=params[key].max)
        sampler = mcmc.EnsembleSampler(walkers, ndim, lnprobList,
                                       args=(x, y, groupParams, pbar))
        burn = int(nsteps * burnin / 100)
        sampler.run_mcmc(pos, burn, storechain=False)
        sampler.reset()
        sampler.run_mcmc(pos, nsteps - burn)
        try:
            pbar.finish()
        except:
            pass
        samples = sampler.flatchain
        val = []
        err = []
        q = [16.0, 50.0, 84.0]
        for i, samp in enumerate(samples.T):
            q16, q50, q84 = np.percentile(samp, q)
            val.append(q50)
            err.append(max([q50 - q16, q84 - q50]))

        for n, v, e in zip(var_names, val, err):
            params[n].value = v
            params[n].stderr = e

        self.mle_fit = params
        self.var_from_params(params)

        data = pd.DataFrame(samples, columns=var_names)
        data.sort_index(axis=1, inplace=True)
        self.mle_data = data
        if store_walks:
            self.walks = [(name, sampler.chain[:, :, i].T)
                          for i, name in enumerate(var_names)]
        else:
            self.walks = None

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
        params = self.params_from_var()
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
            self.var_from_params(self.mle_fit)
        return data

    def display_mle_fit(self, **kwargs):
        """Give a readable overview of the result of the MLE fitting routine.

        Warning
        -------
        The uncertainty shown is the largest of the asymmetrical errors! Work
        is being done to incorporate asymmetrical errors in the report; for
        now, rely on the triangle plot.
        """
        if hasattr(self, 'mle_fit'):
            print(lm.fit_report(self.mle_fit, **kwargs))
        else:
            print('Spectrum has not yet been fitted with this method!')

    ###############################
    # CHI SQUARE FITTING ROUTINES #
    ###############################

    def chisquare_spectroscopic_fit(self, x, y, **kwargs):
        """Use the :meth:`FitToData` method, automatically estimating the errors
        on the counts by the square root."""
        x, y, _ = self.sanitize_input(x, y)
        yerr = np.sqrt(y)
        yerr[np.isclose(yerr, 0.0)] = 1.0
        return self.chisquare_fit(x, y, yerr, **kwargs)

    def chisquare_fit(self, x, y, yerr, pearson=True):
        """Use a non-linear least squares minimization (Levenberg-Marquardt)
        algorithm to minimize the chi-square of the fit to data :attr:`x` and
        :attr:`y` with errorbars :attr:`yerr`. Reasonable bounds are used on
        parameters, and the user-supplied :attr:`self._vary` dictionary is
        consulted to see if a parameter should be varied or not.

        Parameters
        ----------
        x: array_like
            Frequency of the data, in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.
        yerr: array_like
            Error bars on :attr:`y`.
        pearson: boolean, optional
            Selects if the normal or Pearson chi-square statistic is used.
            The Pearson chi-square uses the model value to estimate the
            uncertainty. Defaults to :attr:`True`."""

        x, y, yerr = self.sanitize_input(x, y, yerr)

        params = self.params_from_var()
        try:
            params['sigma_x'].vary = False
        except:
            pass

        result = lm.minimize(model, params, args=(self, x, y, yerr, pearson))

        self.chisq_res_par = result.params
        self.chisq_res_report = lm.fit_report(result)
        return result

    def display_chisquare_fit(self, **kwargs):
        """Display all relevent info of the least-squares fitting routine,
        if this has been performed.

        Parameters
        ----------
        kwargs: misc
            Keywords passed on to :func:`fit_report` from the LMFit package."""
        if hasattr(self, 'chisquare_fit'):
            print('Scaled errors estimated from covariance matrix.')
            print(self.chisq_res_report)
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
            params = self.params_from_var()
            for key in sorted(params.keys()):
                if params[key].vary:
                    var.append(params[key].value)
                    var_names.append(params[key].name)
                    varerr.append(None)
        return var_names, var, varerr

    def display_ci(self):
        """If the confidence bounds for the parameters have been calculated
        with the method :meth:`calculate_confidence_intervals`, print the
        results to stdout."""
        lm.report_ci(self.chisquare_ci)

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
            values = [v for n in self.selected for v in values if v.name in n]
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
