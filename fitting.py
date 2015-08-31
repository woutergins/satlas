import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import emcee as mcmc
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

def model(params, spectrum, x, y, yerr, pearson, **kwargs):
    spectrum.params = params
    model = spectrum(x)
    if pearson:
        yerr = np.sqrt(model)
    return (y - model) / yerr

def chisquare_spectroscopic_fit(spectrum, x, y, **kwargs):
    """Use the :meth:`FitToData` method, automatically estimating the errors
    on the counts by the square root."""
    x, y, _ = spectrum.sanitize_input(x, y)
    yerr = np.sqrt(y)
    yerr[np.isclose(yerr, 0.0)] = 1.0
    return chisquare_fit(spectrum, x, y, yerr, **kwargs)

def chisquare_fit(spectrum, x, y, yerr, pearson=True, monitor=True):
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

    spectrum.params = result.params
    spectrum.chisq_res_par = result.params

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
    x, theta = np.meshgrid(x, theta_array)
    y, _ = np.meshgrid(y, theta_array)
    p = func(y, spectrum(x + theta))
    g = np.exp(-(theta / s)**2 / 2) / s
    return np.log(np.fft.irfft(np.fft.rfft(p) * np.fft.rfft(g))[:, -1])

def lnprob(params, spectrum, x, y, func):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    res = lp + np.sum(loglikelihood(spectrum, params, x, y, func))
    return res

def lnprior(params):
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

def loglikelihood(spectrum, params, x, y, func):
    spectrum.params = params
    if any([np.isclose(X.min(), X.max(), atol=0.1)
            for X in spectrum.seperate_response(x)]) or any(spectrum(x) < 0):
        return -np.inf
    if params['sigma_x'].value > 0:
        s = params['sigma_x'].value
        return_value = x_err_calculation(spectrum, x, y, s, func)
    else:
        return_value = func(y, spectrum(x))
    return return_value

def likelihood_fit(spectrum, x, y, xerr=0, vary_sigma=False, func=llh.poisson_llh, walking=True, **kwargs):
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
        likelihood_walk(spectrum, x, y, func=func, **kwargs)
    return None

def likelihood_walk(spectrum, x, y, func=llh.poisson_llh, nsteps=2000, walkers=20, burnin=10.0,
                    verbose=True, store_walks=False):

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
    else:
        spectrum.walks = None
