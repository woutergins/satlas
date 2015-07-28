"""
.. module:: spectrum
    :platform: Windows
    :synopsis: Implementation of classes for the analysis of hyperfine
     structure spectra, including simultaneous fitting, various fitting
     routines and isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import abc
import emcee as mcmc
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import satlas.loglikelihood as llh
import satlas.profiles as p
import satlas.utilities as utils
from satlas.wigner import wigner_6j as W6J


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


class Spectrum(object, metaclass=abc.ABCMeta):

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

    @property
    def loglikelifunc(self):
        mapping = {llh.Poisson: 'Poisson', llh.Gaussian: 'Gaussian'}
        return mapping[self._loglikelifunc]

    @loglikelifunc.setter
    def loglikelifunc(self, value):
        mapping = {'poisson': llh.Poisson, 'gaussian': llh.Gaussian}
        self._loglikelifunc = mapping.get(value.lower(), llh.Poisson)

    @abc.abstractmethod
    def sanitize_input(self, x, y, yerr=None):
        return

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
        return self._loglikelifunc(y, self(x))

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

    def likelihood_fit(self, x, y, walking=True, **kwargs):
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
        sampler = mcmc.EnsembleSampler(walkers, ndim, lnprobList,
                                       args=(x, y, groupParams))
        burn = int(nsteps * burnin / 100)
        if verbose:
            print('Starting burn-in ({} steps)...'.format(burn))
        sampler.run_mcmc(pos, burn, storechain=False)
        sampler.reset()
        if verbose:
            print('Starting walk ({} steps)...'.format(nsteps - burn))
        sampler.run_mcmc(pos, nsteps - burn)
        if verbose:
            print('Done.')
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

    def chisquare_fit(self, x, y, yerr, pierson=False):
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
        pierson: boolean, optional
            Selects if the normal or Pierson chi-square statistic is used.
            The Pierson chi-square uses the model value to estimate the
            uncertainty. Defaults to :attr:`False`."""

        x, y, yerr = self.sanitize_input(x, y, yerr)

        def Model(params, x, y, yerr, pierson):
            self.var_from_params(params)
            model = self(x)
            if pierson:
                yerr = np.sqrt(model)
                yerr[np.isclose(yerr, 0.0)] = 1.0
            return (y - model) / yerr

        params = self.params_from_var()

        result = lm.minimize(Model, params, args=(x, y, yerr, pierson))

        self.chisquare_result = result

    def display_chisquare_fit(self, **kwargs):
        """Display all relevent info of the least-squares fitting routine,
        if this has been performed.

        Parameters
        ----------
        kwargs: misc
            Keywords passed on to :func:`fit_report` from the LMFit package."""
        if hasattr(self, 'chisquare_fit'):
            print('Scaled errors estimated from covariance matrix.')
            print(lm.fit_report(self.chisquare_result, **kwargs))
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
        if hasattr(self, 'chisquare_result') and (selection.lower() == 'chisquare'
                                               or selection.lower() == 'any'):
            for key in sorted(self.chisquare_result.params.keys()):
                if self.chisquare_result.params[key].vary:
                    var.append(self.chisquare_result.params[key].value)
                    var_names.append(self.chisquare_result.params[key].name)
                    varerr.append(self.chisquare_result.params[key].stderr)
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

    def chisquare_correlation_plot(self, selected=True, **kwargs):
        """If a chisquare fit has been performed, this method creates a figure
        for plotting the correlation maps between parameters.

        Parameters
        ----------
        selected: boolean, optional
            Controls if only the parameters defined in :attr:`selected` are
            used (True) or if all parameters are used (False). Defaults to True
        kwargs: keywords
            Other keywords are passed on to the :func:`conf_interval2d`
            function from lmfit. The exception is the keyword :attr:`limits`,
            which is now a float that indicates how many standard deviations
            have to be traveled.

        Returns
        -------
        figure
            Returns the generated MatPlotLib figure"""
        g = utils.FittingGrid(self.chisquare_result,
                              selected=self.selected if selected else None,
                              **kwargs)
        return g.fig

    def calculate_confidence_intervals(self, selected=True, **kwargs):
        """Calculates the confidence bounds for parameters by making use of
        lmfit's :func:`conf_interval` function. Results are saved in
        :attr:`self.chisquare_ci`

        Parameters
        ----------
        selected: boolean, optional
            Boolean controlling if the used parameters are only the ones
            defined in the attribute :attr:`selected` (True), or if all
            parameters are to be used."""
        names = [p for f in self.selected for p in self.chisquare_result.params
                 if (f in self.chisquare_result.params[p].name and
                     self.chisquare_result.params[p].vary)] if selected else None
        self.chisquare_ci = lm.conf_interval(self.chisquare_result,
                                             p_names=names,
                                             **kwargs)

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
            values = self.chisquare_result.params.values()
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
        """Take the :attr:`x`, :attr:`y`, and :attr:`yerr` inputs, and sanitize
        them for the fit, meaning it should convert :attr:`y`/:attr:`yerr` to
        the output format of the class, and :attr:`x` to the input format of
        the class."""
        if isinstance(y, list):
            y = np.hstack(y)
        if yerr is not None:
            if isinstance(yerr, list):
                yerr = np.hstack(yerr)
        return x, y, yerr

    def params_from_var(self):
        """Combine the parameters from the subspectra into one Parameters
        instance.

        Returns
        -------
        params: Parameters instance describing the spectrum.

        Warning
        -------
        Black magic going on in here, especially in the block of code
        describing the shared parameters."""
        params = lm.Parameters()
        for i, s in enumerate(self.spectra):
            p = s.params_from_var()
            keys = list(p.keys())
            for old_key in keys:
                new_key = 's' + str(i) + '_' + old_key
                p[new_key] = p.pop(old_key)
                for o_key in keys:
                    if p[new_key].expr is not None:
                        n_key = 's' + str(i) + '_' + o_key
                        p[new_key].expr = p[new_key].expr.replace(o_key, n_key)
            params += p

        for i, s in enumerate(self.spectra):
            for key in self.shared:
                if i == 0:
                    continue
                if isinstance(self.spectra[i], IsomerSpectrum):
                    for j, _ in enumerate(self.spectra[i].spectra):
                        first_key = 's0_s' + str(j) + '_' + key
                        new_key = 's' + str(j) + '_' + key
                        for p in params.keys():
                            if new_key in p:
                                if p.startswith('s0_'):
                                    pass
                                else:
                                    params[p].expr = first_key
                                    params[p].vary = False
                else:
                    if isinstance(self.spectra[0], IsomerSpectrum):
                        first_key = 's0_s0_' + key
                    else:
                        first_key = 's0_' + key
                    new_key = 's' + str(i) + '_' + key
                    for p in params.keys():
                        if new_key in p:
                            params[p].expr = first_key
                            params[p].vary = False
        return params

    def var_from_params(self, params):
        """Given a Parameters instance such as returned by the method
        :meth:`params_from_var`, set the parameters of the subspectra to the
        appropriate values.

        Parameters
        ----------
        params: Parameters
            Parameters instance containing the information for the variables.
        """
        for i, s in enumerate(self.spectra):
            p = lm.Parameters()
            if isinstance(s, IsomerSpectrum):
                for j, spec in enumerate(s.spectra):
                    for key in params.keys():
                        k = 's{!s}_s{!s}_'.format(i, j)
                        if key.startswith(k):
                            dinkie = params[key]
                            new_name = key.split('_')
                            new_name = '_'.join(new_name[1:])
                            p.add(new_name, value=dinkie.value,
                                  vary=dinkie.vary, min=dinkie.min,
                                  max=dinkie.max, expr=dinkie.expr)
            else:
                for key in params.keys():
                    if key.startswith('s' + str(i) + '_'):
                        dinkie = params[key]
                        new_name = key.split('_')[-1]
                        p.add(new_name, value=dinkie.value, vary=dinkie.vary,
                              min=dinkie.min, max=dinkie.max, expr=dinkie.expr)
            s.var_from_params(p)

    def split_parameters(self, params):
        """Helper function to split the parameters of the IsomerSpectrum
        instance into a list of parameters suitable for each subspectrum.

        Parameters
        ----------
        params: Parameters
            Parameters of the :class:`IsomerSpectrum` instance.

        Returns
        -------
        p: list of Parameters
            A list of Parameters instances, each entry corresponding to the
            same entry in the attribute :attr:`spectra`."""
        p = []
        for i, _ in enumerate(self.spectra):
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
            p.append(par)
        return p

    def lnprior(self, params):
        """Defines the (uninformative) prior for all parameters.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters with values to be used in the fit/walk

        Returns
        -------
        float
            If any of the parameters are out of bounds, returns :data:`-np.inf`
            , otherwise 1.0 is returned"""
        params = self.split_parameters(params)
        return np.sum([s.lnprior(par) for s, par in zip(self.spectra, params)])

    def seperate_response(self, x):
        return np.squeeze([s.seperate_response(X)
                           for s, X in zip(self.spectra, x)])

    def __call__(self, x):
        return np.hstack([s(X) for s, X in zip(self.spectra, x)])


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
        return [s(x) - s.background for s in self.spectra]

    def __add__(self, other):
        if isinstance(other, IsomerSpectrum):
            spectra = self.spectra + other.spectra
        elif isinstance(other, SingleSpectrum):
            spectra = self.spectra
            spectra.append(other)
        else:
            raise TypeError('unsupported operand type(s)')
        return IsomerSpectrum(spectra)

    def __call__(self, x):
        return np.sum([s(x) for s in self.spectra], axis=0)


class SingleSpectrum(Spectrum):

    r"""Class for the construction of a HFS spectrum, consisting of different
    peaks described by a certain profile. The number of peaks and their
    positions is governed by the atomic HFS.
    Calling an instance of the Spectrum class returns the response value of the
    HFS spectrum for that frequency in MHz.

    Parameters
    ----------
    I: float
        The nuclear spin.
    J: list of 2 floats
        The spins of the fine structure levels.
    ABC: list of 6 floats
        The hyperfine structure constants A, B and C for ground- and excited
        fine level. The list should be given as [A :sub:`lower`,
        A :sub:`upper`, B :sub:`lower`, B :sub:`upper`, C :sub:`upper`,
        C :sub:`lower`].
    df: float
        Center of Gravity of the spectrum.
    fwhm: float or list of 2 floats, optional
        Depending on the used shape, the FWHM is defined by one or two floats.
        Defaults to [50.0, 50.0]
    scale: float, optional
        Sets the strength of the spectrum, defaults to 1.0. Comparable to the
        amplitude of the spectrum.

    Other parameters
    ----------------
    shape : string, optional
        Sets the transition shape. String is converted to lowercase. For
        possible values, see :attr:`Spectrum.__shapes__.keys()`.
        Defaults to Voigt if an incorrect value is supplied.
    racah_int: Boolean, optional
        If True, fixes the relative peak intensities to the Racah intensities.
        Otherwise, gives them equal intensities and allows them to vary during
        fitting.
    shared_fwhm: Boolean, optional
        If True, the same FWHM is used for all peaks. Otherwise, give them all
        the same initial FWHM and let them vary during the fitting.

    Attributes
    ----------
    fwhm : (list of) float or list of 2 floats
        Sets the FWHM for all the transtions. If :attr:`shared_fwhm` is True,
        this attribute is a list of FWHM values for each peak.
    relAmp : list of floats
        Sets the relative intensities of the transitions.
    scale : float
        Sets the amplitude of the global spectrum.
    background : float
        Sets the background of the global spectrum.
    ABC : list of 6 floats
        List of the hyperfine structure constants, organised as
        [A :sub:`lower`, A :sub:`upper`, B :sub:`lower`, B :sub:`upper`,
        C :sub:`upper`, C :sub:`lower`].
    n : integer
        Sets the number of Poisson sidepeaks.
    offset : float
        Sets the offset for the Poisson sidepeaks.
        The sidepeaks are located at :math:`i\cdot \text{offset}`,
        with :math:`i` the number of the sidepeak.
        Note: this means that a negative value indicates a sidepeak
        to the left of the main peak.
    poisson : float
        Sets the Poisson-factor for the Poisson sidepeaks.
        The amplitude of each sidepeak is multiplied by
        :math:`\text{poisson}^i/i!`, with :math:`i` the number of the sidepeak.

    Note
    ----
    The listed attributes are commonly accessed attributes for the end user.
    More are used, and should be looked up in the source code."""

    __shapes__ = {'gaussian': p.Gaussian,
                  'lorentzian': p.Lorentzian,
                  'irrational': p.Irrational,
                  'hyperbolic': p.HyperbolicSquared,
                  'extendedvoigt': p.ExtendedVoigt,
                  'pseudovoigt': p.PseudoVoigt,
                  'voigt': p.Voigt}

    def __init__(self, I, J, ABC, df, fwhm=[50.0, 50.0], scale=1.0,
                 background=0.1, shape='voigt', racah_int=True,
                 shared_fwhm=True):
        super(SingleSpectrum, self).__init__()
        shape = shape.lower()
        if shape not in self.__shapes__:
            print("""Given profile shape not yet supported.
            Defaulting to Voigt lineshape.""")
            shape = 'voigt'
            fwhm = [50.0, 50.0]

        self.I_value = {0.0: ((False, 0), (False, 0), (False, 0),
                              (False, 0), (False, 0), (False, 0)),
                        0.5: ((True, 1), (True, 1),
                              (False, 0), (False, 0), (False, 0), (False, 0)),
                        1.0: ((True, 1), (True, 1),
                              (True, 1), (True, 1),
                              (False, 0), (False, 0))
                        }
        self.J_lower_value = {0.0: ((False, 0), (False, 0), (False, 0)),
                              0.5: ((True, 1),
                                    (False, 0), (False, 0)),
                              1.0: ((True, 1),
                                    (True, 1), (False, 0))
                              }
        self.J_upper_value = {0.0: ((False, 0), (False, 0), (False, 0)),
                              0.5: ((True, 1),
                                    (False, 0), (False, 0)),
                              1.0: ((True, 1),
                                    (True, 1), (False, 0))
                              }
        self.shape = shape
        self._relAmp = []
        self._racah_int = racah_int
        self.shared_fwhm = shared_fwhm
        self.parts = []
        self._I = I
        self._J = J
        self._ABC = ABC
        self.abc_limit = 30000.0
        self.fwhm_limit = 0.1
        self._df = df

        self.scale = scale
        self._background = background

        self._energies = []
        self._mu = []

        self.n = 0
        self.poisson = 0.609
        self.offset = 0

        self._vary = {}
        self.ratio = [None, None, None]

        self.ratioA = (None, 'lower')
        self.ratioB = (None, 'lower')
        self.ratioC = (None, 'lower')

        self.calculateLevels()
        self.relAmp = [f * scale for f in self.relAmp]
        self.calculateLevels()
        self.fwhm = fwhm

    def set_variation(self, varyDict):
        """Sets the variation of the fitparameters as supplied in the
        dictionary.

        Parameters
        ----------
        varydict: dictionary
            A dictionary containing 'key: True/False' mappings

        Note
        ----
        The list of usable keys:

        * :attr:`FWHM` (only for profiles with one float for the FWHM)
        * :attr:`eta`  (only for the Pseudovoigt profile)
        * :attr:`FWHMG` (only for profiles with two floats for the FWHM)
        * :attr:`FWHML` (only for profiles with two floats for the FWHM)
        * :attr:`Al`
        * :attr:`Au`
        * :attr:`Bl`
        * :attr:`Bu`
        * :attr:`Cl`
        * :attr:`Cu`
        * :attr:`df`
        * :attr:`Background`
        * :attr:`Poisson` (only if the attribute *n* is greater than 0)
        * :attr:`Offset` (only if the attribute *n* is greater than 0)"""
        for k in varyDict.keys():
            self._vary[k] = varyDict[k]

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, value):
        self._I = value
        self.calculateLevels()

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, value):
        self._J = value
        self.calculateLevels()

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, value):
        self._background = value

    @property
    def ABC(self):
        return self._ABC

    @ABC.setter
    def ABC(self, value):
        self._ABC = value
        self._calculate_transitions()

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        self._calculate_transitions()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = np.abs(value)

    @property
    def racah_int(self):
        return self._racah_int

    @racah_int.setter
    def racah_int(self, value):
        self._racah_int = value
        self._calculate_intensities()

    @property
    def relAmp(self):
        return self._relAmp

    @relAmp.setter
    def relAmp(self, value):
        if len(value) is len(self.parts):
            value = np.array(value, dtype='float')
            self._relAmp = np.abs(value)
            for prof, val in zip(self.parts, value):
                prof.amp = np.abs(val)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        if self.shared_fwhm:
            self._fwhm = value
            for prof in self.parts:
                prof.fwhm = value
        else:
            if (self.shape in ['extendedvoigt', 'voigt']
                and all(map(lambda x: isinstance(x, float), value))
                and 2 == len(self.parts)) or (not len(value) ==
                                              len(self.parts)):
                self._fwhm = [value for _ in range(len(self.parts))]
            else:
                self._fwhm = value
            for prof, v in zip(self.parts, self.fwhm):
                prof.fwhm = v

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if len(value) is len(self.parts):
            self._mu = value
            for prof, val in zip(self.parts, value):
                prof.mu = val

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = int(value)

    @property
    def poisson(self):
        return self._poisson

    @poisson.setter
    def poisson(self, value):
        self._poisson = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    def fix_ratio(self, value, target='upper', parameter='A'):
        """Fixes the ratio for a given hyperfine parameter to the given value.

        Parameters
        ----------
        value: float
            Value to which the ratio is set
        target: {'upper', 'lower'}
            Sets the target level. If 'upper', the upper parameter is
            calculated as lower * ratio, 'lower' calculates the lower
            parameter as upper * ratio.
        parameter: {'A', 'B', 'C'}
            Selects which hyperfine parameter to set the ratio for."""
        if target.lower() not in ['lower', 'upper']:
            raise KeyError("Target must be 'lower' or 'upper'.")
        if parameter.lower() not in ['a', 'b', 'c']:
            raise KeyError("Parameter must be 'A', 'B' or 'C'.")
        if parameter.lower() == 'a':
            self.ratioA = (value, target)
        if parameter.lower() == 'b':
            self.ratioB = (value, target)
        if parameter.lower() == 'c':
            self.ratioC = (value, target)

    def calculateLevels(self):
        self._F = [np.arange(abs(self._I - self._J[0]),
                             self._I + self._J[0] + 1, 1),
                   np.arange(abs(self._I - self._J[1]),
                             self._I + self._J[1] + 1, 1)]

        self._calculate_transitions()
        self._calculate_intensities()

    def _calculate_transitions(self):
        self._energies = [[self.calculate_F_level_energy(0, F)
                           for F in self._F[0]],
                          [self.calculate_F_level_energy(1, F)
                           for F in self._F[1]]]

        mu = []
        for i, F1 in enumerate(self._F[0]):
            for j, F2 in enumerate(self._F[1]):
                if abs(F2 - F1) <= 1 and not F2 == F1 == 0.0:
                    mu.append(self._energies[1][j] - self._energies[0][i])

        if not len(self.parts) is len(mu):
            self.parts = tuple(
                self.__shapes__[self.shape]() for _ in range(len(mu)))
        self.mu = mu

    def _calculate_intensities(self):
        ampl = []
        if self.I == 0:
            ampl = [1.0]
        else:
            for i, F1 in enumerate(self._F[0]):
                for j, F2 in enumerate(self._F[1]):
                    a = self.calculate_racah_intensity(self._J[0],
                                                       self._J[1],
                                                       F1,
                                                       F2)
                    if a != 0.0:
                        ampl.append(a)
        self.relAmp = ampl

    def calculate_racah_intensity(self, J1, J2, F1, F2, order=1.0):
        return (2 * F1 + 1) * (2 * F2 + 1) * \
            W6J(J2, F2, self._I, F1, J1, order) ** 2

    def calculate_F_level_energy(self, level, F):
        r"""The hyperfine addition to a central frequency (attribute :attr:`df`)
        for a specific level is calculated. The formula comes from
        :cite:`Schwartz1955` and in a simplified form, reads

        .. math::
            C_F &= F(F+1) - I(I+1) - J(J+1)

            D_F &= \frac{3 C_F (C_F + 1) - 4 I (I + 1) J (J + 1)}{2 I (2 I - 1)
            J (2 J - 1)}

            E_F &= \frac{10 (\frac{C_F}{2})^3 + 20(\frac{C_F}{2})^2 + C_F(-3I(I
            + 1)J(J + 1) + I(I + 1) + J(J + 1) + 3) - 5I(I + 1)J(J + 1)}{I(I -
            1)(2I - 1)J(J - 1)(2J - 1)}

            E &= df + \frac{A C_F}{2} + \frac{B D_F}{4} + C E_F

        A, B and C are the dipole, quadrupole and octupole hyperfine
        parameters. Octupole contributions are calculated when both the
        nuclear and electronic spin is greater than 1, quadrupole contributions
        when they are greater than 1/2, and dipole contributions when they are
        greater than 0.

        Parameters
        ----------
        level: int, 0 or 1
            Integer referring to the lower (0) level, or the upper (1) level.
        F: integer or half-integer
            F-quantum number for which the hyperfine-corrected energy has to be
            calculated.

        Returns
        -------
        energy: float
            Energy in MHz."""
        I = self._I
        J = self._J[level]
        A = self._ABC[level]
        B = self._ABC[level + 2]
        C = self._ABC[level + 4]

        if level == 0:
            df = 0
        else:
            df = self._df

        if (I == 0 or J == 0):
            C_F = 0
            D_F = 0
            E_F = 0
        elif (I == .5 or J == .5):
            C_F = F * (F + 1) - I * (I + 1) - J * (J + 1)
            D_F = 0
            E_F = 0
        elif (I == 1. or J == 1.):
            C_F = F * (F + 1) - I * (I + 1) - J * (J + 1)
            D_F = (3 * C_F * (C_F + 1) -
                   4 * I * (I + 1) * J * (J + 1)) /\
                  (2 * I * (2 * I - 1) * J * (2 * J - 1))
            E_F = 0
        else:
            C_F = F * (F + 1) - I * (I + 1) - J * (J + 1)
            D_F = (3 * C_F * (C_F + 1) -
                   4 * I * (I + 1) * J * (J + 1)) /\
                  (2 * I * (2 * I - 1) * J * (2 * J - 1))
            E_F = (10 * (0.5 * C_F) ** 3 + 20 * (0.5 * C_F) ** 2
                   + C_F * (-3 * I * (I + 1) * J * (J + 1) +
                            I * (I + 1) + J * (J + 1) + 3) -
                   5 * I * (I + 1) * J * (J + 1)) /\
                  (I * (I - 1) * (2 * I - 1) * J * (J - 1) * (2 * J - 1))

        return df + 0.5 * A * C_F + 0.25 * B * D_F + C * E_F

    def sanitize_input(self, x, y, yerr=None):
        return x, y, yerr

    def var_from_params(self, params):
        """Given a Parameters instance 'params', the value-fields for all the
        parameters are extracted and used to set the values of the spectrum.
        Will raise a KeyError exception if an unsuitable instance is
        supplied.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters to set all values."""
        if self.shape not in ['extendedvoigt', 'voigt']:
            if self.shared_fwhm:
                self.fwhm = params['FWHM'].value
            else:
                self.fwhm = [params['FWHM'+str(i)].value
                             for i in range(len(self.parts))]
            if self.shape in ['pseudovoigt']:
                for part in self.parts:
                    part.n = params['eta'].value
        else:
            if self.shared_fwhm:
                self.fwhm = [params['FWHMG'].value, params['FWHML'].value]
            else:
                self.fwhm = [[params['FWHMG' + str(i)].value,
                              params['FWHML' + str(i)].value]
                             for i in range(len(self.parts))]

        self.scale = params['scale'].value
        self.relAmp = [params['Amp' + str(i)].value
                       for i in range(len(self.parts))]

        self.ABC = [params['Al'].value, params['Au'].value,
                    params['Bl'].value, params['Bu'].value,
                    params['Cl'].value, params['Cu'].value]

        self.df = params['df'].value

        self.background = params['Background'].value
        self.n = params['N'].value
        if self.n > 0:
            self.Poisson = params['Poisson'].value
            self.Offset = params['Offset'].value

    def params_from_var(self):
        """Goes through all the relevant parameters of the spectrum,
        and returns a Parameters instance containing all the information. User-
        supplied information in the self._vary dictionary is used to set
        the variation of parameters during the fitting, while
        making sure that the A, B and C parameters are not used if the spins
        do not allow it.

        Returns
        -------
        Parameters
            Instance suitable for the method :meth:`var_from_params`."""
        par = lm.Parameters()
        if self.shape not in ['extendedvoigt', 'voigt']:
            if self.shared_fwhm:
                par.add('FWHM', value=self.fwhm, vary=True,
                        min=self.fwhm_limit)
            else:
                for i, val in enumerate(self.fwhm):
                    par.add('FWHM' + str(i), value=val, vary=True,
                            min=self.fwhm_limit)
            if self.shape in ['pseudovoigt']:
                par.add('eta', value=self.parts[0].n, vary=True, min=0, max=1)
        else:
            if self.shared_fwhm:
                par.add('FWHMG', value=self.fwhm[0], vary=True,
                        min=self.fwhm_limit)
                par.add('FWHML', value=self.fwhm[1], vary=True,
                        min=self.fwhm_limit)
                val = 0.5346 * self.fwhm[1] + np.sqrt(0.2166 *
                                                      self.fwhm[1] ** 2
                                                      + self.fwhm[0] ** 2)
                par.add('TotalFWHM', value=val, vary=False,
                        expr='0.5346*FWHML+sqrt(0.2166*FWHML**2+FWHMG**2)')
            else:
                for i, val in enumerate(self.fwhm):
                    par.add('FWHMG' + str(i), value=val[0], vary=True,
                            min=self.fwhm_limit)
                    par.add('FWHML' + str(i), value=val[1], vary=True,
                            min=self.fwhm_limit)
                    val = 0.5346 * val[1] + np.sqrt(0.2166 * val[1] ** 2
                                                    + val[0] ** 2)
                    par.add('TotalFWHM' + str(i), value=val, vary=False,
                            expr='0.5346*FWHML' + str(i) +
                                 '+sqrt(0.2166*FWHML' + str(i) +
                                 '**2+FWHMG' + str(i) + '**2)')

        par.add('scale', value=self.scale, vary=self.racah_int, min=0)
        for i, prof in enumerate(self.parts):
            par.add('Amp' + str(i), value=self._relAmp[i],
                    vary=not self.racah_int, min=0)

        b = (None, None) if self.abc_limit is None else (-self.abc_limit,
                                                         self.abc_limit)
        par.add('Al', value=self._ABC[0], vary=True, min=b[0], max=b[1])
        par.add('Au', value=self._ABC[1], vary=True, min=b[0], max=b[1])
        par.add('Bl', value=self._ABC[2], vary=True, min=b[0], max=b[1])
        par.add('Bu', value=self._ABC[3], vary=True, min=b[0], max=b[1])
        par.add('Cl', value=self._ABC[4], vary=True, min=b[0], max=b[1])
        par.add('Cu', value=self._ABC[5], vary=True, min=b[0], max=b[1])

        ratios = (self.ratioA, self.ratioB, self.ratioC)
        labels = (('Al', 'Au'), ('Bl', 'Bu'), ('Cl', 'Cu'))
        for r, (l, u) in zip(ratios, labels):
            if r[0] is not None:
                if r[1].lower() == 'lower':
                    fixed, free = l, u
                else:
                    fixed, free = u, l
                par[fixed].expr = str(r[0]) + '*' + free
                par[fixed].vary = False


        par.add('df', value=self._df, vary=True)

        par.add('Background', value=self.background, vary=True, min=0)
        par.add('N', value=self._n, vary=False)
        if self._n > 0:
            par.add('Poisson', value=self._poisson, vary=True, min=0)
            par.add('Offset', value=self._offset, vary=True, min=None, max=-0.01)
        for key in self._vary.keys():
            if key in par.keys():
                par[key].vary = self._vary[key]
        par['N'].vary = False

        if self._I in self.I_value:
            Al, Au, Bl, Bu, Cl, Cu = self.I_value[self._I]
            if not Al[0]:
                par['Al'].vary, par['Al'].value = Al
            if not Au[0]:
                par['Au'].vary, par['Au'].value = Au
            if not Bl[0]:
                par['Bl'].vary, par['Bl'].value = Bl
            if not Bu[0]:
                par['Bu'].vary, par['Bu'].value = Bu
            if not Cl[0]:
                par['Cl'].vary, par['Cl'].value = Cl
            if not Cu[0]:
                par['Cu'].vary, par['Cu'].value = Cu
        if self._J[0] in self.J_lower_value:
            Al, Bl, Cl = self.J_lower_value[self._J[0]]
            if not Al[0]:
                par['Al'].vary, par['Al'].value = Al
            if not Bl[0]:
                par['Bl'].vary, par['Bl'].value = Bl
            if not Cl[0]:
                par['Cl'].vary, par['Cl'].value = Cl
        if self._J[1] in self.J_upper_value:
            Au, Bu, Cu = self.J_upper_value[self._J[1]]
            if not Au[0]:
                par['Au'].vary, par['Au'].value = Au
            if not Bu[0]:
                par['Bu'].vary, par['Bu'].value = Bu
            if not Cu[0]:
                par['Cu'].vary, par['Cu'].value = Cu

        return par

    def bootstrap(self, x, y, bootstraps=100, samples=None, selected=True):
        """Given an experimental spectrum of counts, generate a number of
        bootstrapped resampled spectra, fit these, and return a pandas
        DataFrame containing result of fitting these resampled spectra.

        Parameters
        ----------
        x: array_like
            Frequency in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.

        Other Parameters
        ----------------
        bootstraps: integer, optional
            Number of bootstrap samples to generate, defaults to 100.
        samples: integer, optional
            Number of counts in each bootstrapped spectrum, defaults to
            the number of counts in the supplied spectrum.
        selected: boolean, optional
            Selects if only the parameters in :attr:`self.selected` are saved
            in the DataFrame. Defaults to True (saving only the selected).

        Returns
        -------
        DataFrame
            DataFrame containing the results of fitting the bootstrapped
            samples."""
        total = np.cumsum(y)
        dist = total / float(y.sum())
        names, var, varerr = self.vars(selection='chisquare')
        selected = self.selected if selected else names
        v = [name for name in names if name in selected]
        data = pd.DataFrame(index=np.arange(0, bootstraps + 1),
                            columns=v)
        stderrs = pd.DataFrame(index=np.arange(0, bootstraps + 1),
                               columns=v)
        v = [var[i] for i, name in enumerate(names) if name in selected]
        data.loc[0] = v
        v = [varerr[i] for i, name in enumerate(names) if name in selected]
        stderrs.loc[0] = v
        if samples is None:
            samples = y.sum()
        length = len(x)

        for i in range(bootstraps):
            newy = np.bincount(
                    np.searchsorted(
                            dist,
                            np.random.rand(samples)
                            ),
                    minlength=length
                    )
            self.chisquare_spectroscopic_fit(x, newy)
            names, var, varerr = self.vars(selection='chisquare')
            v = [var[i] for i, name in enumerate(names) if name in selected]
            data.loc[i + 1] = v
            v = [varerr[i] for i, name in enumerate(names) if name in selected]
            stderrs.loc[i + 1] = v
        pan = {'data': data, 'stderr': stderrs}
        pan = pd.Panel(pan)
        return pan

    def __add__(self, other):
        """Add two spectra together to get an :class:`IsomerSpectrum`.

        Parameters
        ----------
        other: Spectrum
            Other spectrum to add.

        Returns
        -------
        IsomerSpectrum
            An Isomerspectrum combining both spectra."""
        if isinstance(other, SingleSpectrum):
            l = [self, other]
        return IsomerSpectrum(l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def seperate_response(self, x):
        """Get the response for each seperate spectrum for the values :attr:`x`
        , without background.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz.

        Returns
        -------
        list of floats or NumPy arrays
            Seperate responses of spectra to the input :attr:`x`."""
        return [self(x)]

    def __call__(self, x):
        """Get the response for frequency :attr:`x` (in MHz) of the spectrum.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz

        Returns
        -------
        float or NumPy array
            Response of the spectrum for each value of :attr:`x`."""
        if self._n > 0:
            s = np.zeros(x.shape)
            for i in range(self._n + 1):
                s += (self.poisson ** i) * sum([prof(x + i * self.offset)
                                                for prof in self.parts]) \
                    / np.math.factorial(i)
            s = s * self.scale
        else:
            s = self.scale * sum([prof(x) for prof in self.parts])
        return s + self.background