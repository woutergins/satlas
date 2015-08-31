import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import emcee as mcmc
import copy
try:
    import progressbar
except:
    pass
import pandas as pd
from . import loglikelihood as llh

__all__ = ['chisquare_spectroscopic_fit', 'chisquare_fit', 'likelihood_fit']

###############################
# CHI SQUARE FITTING ROUTINES #
###############################

def model(params, spectrum, x, y, yerr, pearson=False):
    """Model function for chisquare fitting routines as established
    in this module.

    Parameters
    ----------
    params: lmfit.Parameters
        Instance of lmfit.Parameters object, to be assigned to the spectrum object.
    spectrum: satlas.Spectrum
        Instance of a satlas.Spectrum object, to be fitted to the data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    yerr: array_like
        Experimental errorbars on the y-axis.
    pearson: boolean
        If True, use the square root of the fitted value as the
        experimental uncertainty.

    Returns
    -------
    NumPy array
        Array containing the residuals for the given parameters, divided by the
        experimental uncertainty

    Note
    ----
    If a custom function is to be used for the calculation of the residual,
    this function should be overwritten."""
    spectrum.params = params
    model = spectrum(x)
    if pearson:
        yerr = np.sqrt(model)
    return (y - model) / yerr

def chisquare_spectroscopic_fit(spectrum, x, y, **kwargs):
    """Use the :func:`chisquare_fit` function, automatically estimating the errors
    on the counts by the square root.

    Parameters
    ----------
    spectrum: satlas.Spectrum object
        Spectrum object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis."""
    x, y, _ = spectrum.sanitize_input(x, y)
    yerr = np.sqrt(y)
    yerr[np.isclose(yerr, 0.0)] = 1.0
    return chisquare_fit(spectrum, x, y, yerr, **kwargs)

def chisquare_fit(spectrum, x, y, yerr, pearson=True, monitor=True):
    """Use a non-linear least squares minimization (Levenberg-Marquardt)
    algorithm to minimize the chi-square of the fit to data :attr:`x` and
    :attr:`y` with errorbars :attr:`yerr`.

    Parameters
    ----------
    spectrum: satlas.Spectrum object
        Spectrum object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    yerr: array_like
        Error bars on :attr:`y`.
    pearson: boolean, optional
        Selects if the normal or Pearson chi-square statistic is used.
        The Pearson chi-square uses the model value to estimate the
        uncertainty. Defaults to :attr:`True`.
    monitor: boolean, optional
        If True, a plot will be displayed during the fitting which gives the
        reduced chisquare statistic in function of the iteration
        number."""

    x, y, yerr = spectrum.sanitize_input(x, y, yerr)

    params = spectrum.params
    try:
        params['sigma_x'].vary = False
    except:
        pass

    if monitor:
        result = lm.Minimizer(model, params, fcn_args=(spectrum, x, y, yerr, pearson))
        result.prepare_fit(params)
        try:
            X = np.concatenate(x)
            nfree = len(X.flatten()) - result.nvarys
        except:
            nfree = len(x.flatten()) - result.nvarys
        fig, ax = plt.subplots(1, 1)
        line, = ax.plot([], [])
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$\chi^2_{red}$')
        def plot(params, iter, resid, *args, **kwargs):
            nfree = kwargs['nfree']
            line = kwargs['line']
            ax = kwargs['ax']
            redchi = (resid**2).sum()/nfree
            xdata = np.append(line.get_xdata(), iter)
            ydata = np.append(line.get_ydata(), redchi)
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.show(block=False)
        result = lm.minimize(model, params, args=(spectrum, x, y, yerr, pearson), kws={'nfree': nfree, 'line': line, 'ax': ax},
                             iter_cb=plot)
    else:
        result = lm.minimize(model, params, args=(spectrum, x, y, yerr, pearson))

    spectrum.params = copy.deepcopy(result.params)
    spectrum.chisq_res_par = copy.deepcopy(result.params)
    spectrum.ndof = copy.deepcopy(result.nfree)
    spectrum.redchi = copy.deepcopy(result.redchi)
    spectrum.chisqr = copy.deepcopy(result.chisqr)

##########################################
# MAXIMUM LIKELIHOOD ESTIMATION ROUTINES #
##########################################


class PriorParameter(lm.Parameter):

    # Extended the Parameter class from LMFIT to incorporate prior boundaries.

    def __init__(self, name, value=None, vary=True, min=None, max=None,
                 expr=None, priormin=None, priormax=None):
        super(PriorParameter, self).__init__(name, value=value,
                                             vary=vary, min=min,
                                             max=max, expr=expr)
        self.priormin = priormin
        self.priormax = priormax

theta_array = np.linspace(-3, 3, 1024)

def x_err_calculation(spectrum, x, y, s, func):
    """Calculates the loglikelihood for a spectrum given
    x and y values. Incorporates a common given error on
    the x-axis.

    Parameters
    ----------
    spectrum: satlas.Spectrum object
        Spectrum object set to current parameters.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    s: float
        Experimental uncertainty on the x-axis
    func: function
        Function taking (y_data, y_model) as input,
        and returning the loglikelihood that the data is
        drawn from a distribution characterized by the model.

    Returns
    -------
    array_like

    Note
    ----
    This method uses the FFT algorithm to quickly calculate
    a convolution integral. If greater accuracy is required,
    change :attr:`satlas.fitting.theta_array` to a suitable
    range and length."""
    x, theta = np.meshgrid(x, theta_array)
    y, _ = np.meshgrid(y, theta_array)
    p = np.exp(func(y, spectrum(x + theta)))
    g = np.exp(-(theta / s)**2 / 2) / s
    return np.log(np.fft.irfft(np.fft.rfft(p) * np.fft.rfft(g))[:, -1])

def lnprob(params, spectrum, x, y, func):
    """Calculates the logarithm of the probability that the data fits
    the model given the current parameters.

    Parameters
    ----------
    params: lmfit.Parameters object with satlas.PriorParameters
        Group of parameters for which the fit has to be evaluated.
    spectrum: satlas.Spectrum object
        Spectrum object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    func: function
        Function calculating the loglikelihood of y_data being drawn from
        a distribution characterized by y_model.

    Note
    ----
    The prior is first evaluated for the parameters. If this is
    not finite, the values are rejected from consideration by
    immediately returning -np.inf."""
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    res = lp + np.sum(loglikelihood(params, spectrum, x, y, func))
    return res

def lnprior(params):
    """Calculates the logarithm of the prior given the parameter
    values. This is independent of the data to be fitted to.

    Parameters
    ----------
    params: lmfit.Parameters with satlas.PriorParameters
        Collection of satlas.PriorParameters, which
        contain also prior bounds, so the boundary
        calculations are not triggered before this point.
        Allows easy interfacing with the emcee package.

    Returns
    -------
    1.0 or -np.inf
        Calculates a flat prior: returns 1.0 if all
        parameters are inside their boundaries,
        reutnr -np.inf if one of them is not.

    Note
    ----
    In case a custom prior distribution is required,
    this function has to be overwritten."""
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

def loglikelihood(params, spectrum, x, y, func):
    """Given a parameters object, a Spectrum object, experimental data
    and a loglikelihood function, calculates the loglikelihood for
    all data points.

    Parameters
    ----------
    params: lmfit.Parameters object with satlas.PriorParameters
        Group of parameters for which the fit has to be evaluated.
    spectrum: satlas.Spectrum object
        Spectrum object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    func: function
        Function calculating the loglikelihood of y_data being drawn from
        a distribution characterized by y_model.

    Returns
    -------
    array_like
        Array containing the loglikelihood for each seperate datapoint."""
    spectrum.params = params
    # If any value of the evaluated spectrum is below 0,
    # or the difference between the minimum and maximum is too low,
    # reject the parameter values.
    if any([np.isclose(X.min(), X.max(), atol=0.1)
            for X in spectrum.seperate_response(x)]) or any(spectrum(x) < 0):
        return -np.inf
    # If a value is given to the uncertainty on the x-values, use the adapted
    # function.
    if params['sigma_x'].value > 0:
        s = params['sigma_x'].value
        return_value = x_err_calculation(spectrum, x, y, s, func)
    else:
        return_value = func(y, spectrum(x))
    return return_value

def likelihood_fit(spectrum, x, y, xerr=0, vary_sigma=False, func=llh.poisson_llh, walking=True, walk_kws={}):
    """Fits the given spectrum to the given data using the Maximum Likelihood Estimation technique.

    Parameters
    ----------
    spectrum: satlas.Spectrum
        Spectrum to be fitted to the data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    xerr: float, optional
        Estimated value for the uncertainty on the x-values.
        Set to 0 to ignore this uncertainty. Defaults to 0.
    vary_sigma: boolean, optional
        If True, the uncertainty on the x-values will be
        treated as a parameter and will be optimized.
    func: function, optional
        Used to calculate the loglikelihood that the data is drawn
        from a distribution given a model value. Should accept
        input as (y_data, y_model). Defaults to the Poisson
        loglikelihood.
    walking: boolean, optional
        If True, the uncertainty on the parameters is estimated
        by performing a random walk in parameter space and
        evaluating the loglikelihood.
    walk_kws: dictionary
        Contains the keywords for the :func:`likelihood_walk`
        function, used if walking is set to True."""
    def negativeloglikelihood(*args, **kwargs):
        return -lnprob(*args, **kwargs)

    x, y, _ = spectrum.sanitize_input(x, y)
    params = spectrum.params
    params.add('sigma_x', value=xerr, vary=vary_sigma, min=0)
    result = lm.Minimizer(negativeloglikelihood, params, fcn_args=(spectrum, x, y, func))
    result.scalar_minimize(method='Nelder-Mead')
    spectrum.params = result.params
    spectrum.mle_fit = result.params
    spectrum.mle_result = result.message

    if walking:
        likelihood_walk(spectrum, x, y, func=func, **walk_kws)
    return None

def likelihood_walk(spectrum, x, y, func=llh.poisson_llh, nsteps=2000, walkers=20, burnin=10.0,
                    verbose=True, store_walks=False):
    """Calculates the uncertainty on MLE-optimized parameter values
    by performing a random walk through parameter space and comparing
    the resulting loglikelihood values. For more information,
    see the emcee package.

    Parameters
    ----------
    spectrum: satlas.Spectrum
        Spectrum to be fitted to the data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    func: function, optional
        Used to calculate the loglikelihood that the data is drawn
        from a distribution given a model value. Should accept
        input as (y_data, y_model). Defaults to the Poisson
        loglikelihood.
    walkers: integer, optional
        Sets the number of walkers to be used for the random walk.
        The number of walkers should never be less than twice the
        number of parameters. For more information on this, see
        the emcee documentation. Defaults to 20 walkers.
    nsteps: integer, optional
        Determines how many steps each walker should take.
        Defaults to 2000 steps.
    burnin: float < 100.0
        Sets the percentage of the walk to be considered burn-in
        (steps for which the walk has not started exploring the space yet).
        Defaults to 10.0 percent.
    verbose: boolean, optional
        If True, a progressbar is printed and updated every second.
        This progressbar displays the progress of the walk, with a primitive
        estimate of the remaining time in the calculation.
    stor_walks: boolean, optional
        If True, the data from the walks is stored in the spectrum
        object under the attribute :attr:`walks`. Defaults to False."""

    params = spectrum.mle_fit
    var_names = []
    vars = []
    for key in params.keys():
        if params[key].vary:
            var_names.append(key)
            vars.append(params[key].value)
    ndim = len(vars)
    pos = mcmc.utils.sample_ball(vars, [1e-4] * len(vars), size=walkers)
    x, y, _ = spectrum.sanitize_input(x, y)

    if verbose:
        try:
            widgets = ['Walk:', progressbar.Percentage(), ' ',
                       progressbar.Bar(marker=progressbar.RotatingMarker()),
                       ' ', progressbar.AdaptiveETA(num_samples=100)]
            pbar = progressbar.ProgressBar(widgets=widgets,
                                           maxval=walkers * nsteps).start()
        except:
            pbar = 0
    else:
        pbar = 0

    def lnprobList(fvars, spectrum, groupParams, x, y, pbar, func):
        for val, n in zip(fvars, var_names):
            groupParams[n].value = val
        try:
            pbar += 1
        except:
            pass
        return lnprob(groupParams, spectrum, x, y, func)
    groupParams = lm.Parameters()
    for key in params.keys():
        groupParams[key] = PriorParameter(key,
                                          value=params[key].value,
                                          vary=params[key].vary,
                                          expr=params[key].expr,
                                          priormin=params[key].min,
                                          priormax=params[key].max)
    sampler = mcmc.EnsembleSampler(walkers, ndim, lnprobList,
                                   args=(spectrum, groupParams, x, y, pbar, func))
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

    spectrum.mle_fit = params
    spectrum.params = params

    data = pd.DataFrame(samples, columns=var_names)
    data.sort_index(axis=1, inplace=True)
    spectrum.mle_data = data
    if store_walks:
        spectrum.walks = [(name, sampler.chain[:, :, i].T)
                      for i, name in enumerate(var_names)]
