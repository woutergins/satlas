"""
Implementation of fitting routines specialised for Spectrum objects.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import emcee as mcmc
from scipy.misc import derivative
import copy
try:
    import progressbar
except:
    pass
import pandas as pd
import h5py
import os
from . import loglikelihood as llh

__all__ = ['chisquare_spectroscopic_fit', 'chisquare_fit',
           'likelihood_fit', 'likelihood_walk']

###############################
# CHI SQUARE FITTING ROUTINES #
###############################

def chisquare_model(params, f, x, y, yerr, xerr=None, func=None):
    r"""Model function for chisquare fitting routines as established
    in this module.

    Parameters
    ----------
    params: lmfit.Parameters
        Instance of lmfit.Parameters object, to be assigned to the model object.
    f: :class:`.BaseModel`
        Callable instance with the correct methods for the fitmethods.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    yerr: array_like
        Experimental errorbars on the y-axis.

    Other parameters
    ----------------
    xerr: array_like, optional
        Given an array with the same size as *x*, the error is taken into
        account by using the method of estimated variance. Defaults to *None*.
    func: function, optional
        Given a function, the errorbars on the y-axis is calculated from
        the fitvalue using this function. Defaults to *None*.

    Returns
    -------
    NumPy array
        Array containing the residuals for the given parameters, divided by the
        uncertainty.

    Note
    ----
    If a custom function is to be used for the calculation of the residual,
    this function should be overwritten.

    The method of estimated variance calculates the chisquare in the following way:

        .. math::

            \sqrt{\chi^2} = \frac{y-f(x)}{\sqrt{\sigma_x^2+f'(x)^2\sigma_x^2}}"""
    f.params = params
    model = np.hstack(f(x))
    if func is not None:
        yerr = func(model)
    if xerr is not None:
        xerr = np.hstack((derivative(f, x, dx=1E-6) * xerr))
        bottom = np.sqrt(yerr * yerr + xerr * xerr)
    else:
        bottom = yerr
    return (y - model) / bottom

def chisquare_spectroscopic_fit(f, x, y, xerr=None, func=None):
    """Use the :func:`chisquare_fit` function, automatically estimating the errors
    on the counts by the square root.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis."""
    y = np.hstack(y)
    yerr = np.sqrt(y)
    yerr[np.isclose(yerr, 0.0)] = 1.0
    return chisquare_fit(f, x, y, yerr, xerr=xerr, func=func)

def chisquare_fit(f, x, y, yerr, xerr=None, func=None):
    """Use a non-linear least squares minimization (Levenberg-Marquardt)
    algorithm to minimize the chi-square of the fit to data *x* and
    *y* with errorbars *yerr*.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    yerr: array_like
        Error bars on *y*.

    Other parameters
    ----------------
    xerr: array_like, optional
        Error bars on *x*.
    func: boolean, optional
        Uses the provided function on the fitvalue to calculate the
        errorbars.

    Return
    ------
    success, message: bool and string
        Boolean indicating the success of the convergence, and the message
        from the optimizer."""

    params = f.params
    try:
        params['sigma_x'].vary = False
    except:
        pass

    result = lm.minimize(chisquare_model, params, args=(f, x, np.hstack(y), np.hstack(yerr), xerr, func))

    f.params = copy.deepcopy(result.params)
    f.chisq_res_par = copy.deepcopy(result.params)
    f.ndof = copy.deepcopy(result.nfree)
    f.redchi = copy.deepcopy(result.redchi)
    f.chisqr = copy.deepcopy(result.chisqr)
    return result.success, result.message

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

theta_array = np.linspace(-5, 5, 2**10)
_x_err_calculation_stored = {}
sqrt2pi = np.sqrt(2*np.pi)

def likelihood_x_err(f, x, y, xerr, func):
    """Calculates the loglikelihood for a model given
    x and y values. Incorporates a common given error on
    the x-axis.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model object set to current parameters.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    xerr: float
        Experimental uncertainty on *x*.
    func: function
        Function taking (*y_data*, *y_model*) as input,
        and returning the loglikelihood that the data is
        drawn from a distribution characterized by the model.

    Returns
    -------
    array_like

    Note
    ----
    This method uses the FFT algorithm to quickly calculate
    a convolution integral. If greater accuracy is required,
    change *satlas.fitting.theta_array* to a suitable
    range and length."""
    # Cache already calculated values:
    # - x_grid
    # - y_grid
    # - FFT of x-uncertainty
    # Note that this works only if the uncertainty remains the same.
    # If a parameter approach is desired, this needs to be changed.
    key = hash(x.data.tobytes()) + hash(y.data.tobytes())
    if key in _x_err_calculation_stored:
        x_grid, y_grid, theta, rfft_g = _x_err_calculation_stored[key]
    else:
        x_grid, theta = np.meshgrid(x, theta_array)
        y_grid, _ = np.meshgrid(y, theta_array)
        g_top = (np.exp(-theta*theta * 0.5)).T
        g = (g_top.T / (sqrt2pi * xerr)).T
        rfft_g = np.fft.rfft(g)
        _x_err_calculation_stored[key] = x_grid, y_grid, theta, rfft_g
    # Calculate the loglikelihoods for the grid of uncertainty.
    # Each column is a new datapoint.
    vals = func(y_grid, np.hstack(f(x_grid + xerr * theta)))
    # To avoid overflows, subtract the maximal values from each column.
    mod = vals.max(axis=0)
    vals_mod = vals - mod
    p = (np.exp(vals_mod)).T
    # Perform the convolution.
    integral_value = np.fft.irfft(np.fft.rfft(p) * rfft_g)[:, -1]
    # After taking the logarithm, add the maximal values again.
    # The subtraction becomes multiplication (with an exponential) after the exponential,
    # shifts through the integral, and becomes an addition (due to the logarithm).
    return np.log(integral_value) + mod

def likelihood_lnprob(params, f, x, y, xerr, func):
    """Calculates the logarithm of the probability that the data fits
    the model given the current parameters.

    Parameters
    ----------
    params: lmfit.Parameters object with satlas.PriorParameters
        Group of parameters for which the fit has to be evaluated.
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    xerr: array_like
        Uncertainty values on *x*.
    func: function
        Function calculating the loglikelihood of y_data being drawn from
        a distribution characterized by y_model.

    Note
    ----
    The prior is first evaluated for the parameters. If this is
    not finite, the values are rejected from consideration by
    immediately returning -np.inf."""
    lp = likelihood_lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    res = lp + np.sum(likelihood_loglikelihood(params, f, x, y, xerr, func))
    return res

def likelihood_lnprior(params):
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
        return -np.inf if one of them is not.

    Note
    ----
    In case a custom prior distribution is required,
    this function has to be overwritten."""
    for key in params.keys():
        if params[key].vary:
            try:
                leftbound, rightbound = (params[key].priormin,
                                         params[key].priormax)
            except:
                leftbound, rightbound = params[key].min, params[key].max
            leftbound = -np.inf if leftbound is None else leftbound
            rightbound = np.inf if rightbound is None else rightbound
            if not leftbound < params[key].value < rightbound:
                return -np.inf
    return 1.0

def likelihood_loglikelihood(params, f, x, y, xerr, func):
    """Given a parameters object, a Model object, experimental data
    and a loglikelihood function, calculates the loglikelihood for
    all data points.

    Parameters
    ----------
    params: lmfit.Parameters object with satlas.PriorParameters
        Group of parameters for which the fit has to be evaluated.
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    xerr: array_like
        Experimental data on *x*.
    func: function
        Function calculating the loglikelihood of y_data being drawn from
        a distribution characterized by y_model.

    Returns
    -------
    array_like
        Array containing the loglikelihood for each seperate datapoint."""
    f.params = params
    # If any value of the evaluated model is below 0,
    # or the difference between the minimum and maximum is too low,
    # reject the parameter values.
    response = np.hstack(f(x))
    if np.isclose(response.min(), response.max(), atol=0.1) or any(response < 0):
        return -np.inf
    # If a value is given to the uncertainty on the x-values, use the adapted
    # function.
    if xerr is None or np.allclose(0, xerr):
        return_value = func(y, response)
    else:
        return_value = likelihood_x_err(f, x, y, xerr, func)
    return return_value

def likelihood_fit(f, x, y, xerr=None, func=llh.poisson_llh, method='L-BFGS-B', method_kws={}, walking=False, walk_kws={}):
    """Fits the given model to the given data using the Maximum Likelihood Estimation technique.
    The given function is used to calculate the loglikelihood. After the fit, the message
    from the optimizer is printed and returned.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model to be fitted to the data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    xerr: array_like, optional
        Estimated value for the uncertainty on the x-values.
        Set to *None* to ignore this uncertainty. Defaults to *None*.
    func: function, optional
        Used to calculate the loglikelihood that the data is drawn
        from a distribution given a model value. Should accept
        input as (y_data, y_model). Defaults to the Poisson
        loglikelihood.
    method: str, optional
        Selects the algorithm to be used by the minimizer used by LMFIT.
        For an overview, see the LMFIT and SciPy documentation.
        Defaults to 'L-BFGS-B'.
    method_kws: dict, optional
        Dictionary containing the keywords to be passed to the
        minimizer.
    walking: boolean, optional
        If True, the uncertainty on the parameters is estimated
        by performing a random walk in parameter space and
        evaluating the loglikelihood. Defaults to False.
    walk_kws: dictionary
        Contains the keywords for the :func:`.likelihood_walk`
        function, used if walking is set to True.

    Returns
    -------
    success, message: boolean and str
        Boolean indicating the success of the optimization and
        the message from the optimizer."""

    def negativeloglikelihood(*args, **kwargs):
        return -likelihood_lnprob(*args, **kwargs)

    y = np.hstack(y)
    params = f.params
    result = lm.Minimizer(negativeloglikelihood, params, fcn_args=(f, x, y, xerr, func))
    success = result.scalar_minimize(method=method, **method_kws)
    f.params = result.params
    f.mle_fit = result.params
    f.mle_result = result.message
    f.mle_likelihood = negativeloglikelihood(params, f, x, y, xerr, func)

    if walking:
        likelihood_walk(f, x, y, xerr=xerr, func=func, **walk_kws)
    return success, result.message

############################
# uncertainty CALCULATIONS #
############################

def calculate_analytical_uncertainty(f, x, y, method='chisquare', filter=None, fit_kws={}):
    """Calculates the analytical errors on the parameters, by changing the value for
    a parameter and finding the point where the chisquare for the refitted parameters
    is one greater. For MLE, an increase of 0.5 is sought. The corresponding series
    of parameters of the model is adjusted with the values found here.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Instance of a model which is to be fitted.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    method: {'chisquare', 'mle'}
        Select for which method the analytical uncertainty has to be calculated.
        Defaults to 'chisquare'.
    filter: list of strings, optional
        Select only a subset of the variable parameters to calculate the uncertainty for.
        Defaults to *None* (all parameters).
    fit_kws: dictionary, optional
        Dictionary of keywords to be passed on to the selected fitting routine.

    Note
    ----
    The function edits the parameters of the given instance. Furthermore,
    it only searches for the uncertainty in the neighbourhood of the starting
    point, which is taken to be the values of the parameters as given in
    the instance. This does not do a full exploration, so the results might be
    from a local minimum!"""
    def fit_new_value(value, f, params, params_name, x, y, orig_value, func):
        try:
            if all(value == orig_value):
                return 0
            for v, n in zip(value, params_name):
                params[n].value = v
                params[n].vary = False
        except:
            if value == orig_value:
                return 0
            params[params_name].value = value
            params[params_name].vary = False
        f.params = params
        success = False
        counter = 0
        while not success:
            success, message = func(f, x, y, **fit_kws)
            counter += 1
            if counter > 10:
                success = True
                print('Fitting did not converge, carrying on...')
        return_value = getattr(f, attr) - orig_value
        return return_value

    # Save the original goodness-of-fit and parameters for later use
    mapping = {'chisquare': (fitting.chisquare_spectroscopic_fit, 'chisqr', 'chisqr_res_par'),
               'mle': (fitting.likelihood_fit, 'mle_likelihood', 'mle_fit')}
    func, attr, save_attr = mapping.pop(method.lower(), (fitting.chisquare_spectroscopic_fit, 'chisqr'))

    func(f, x_data, y_data, **fit_kws)

    orig_value = getattr(f, attr)
    orig_params = copy.deepcopy(f.params)

    ranges = {}

    # Select all variable parameters, generate the figure
    param_names = []
    no_params = 0
    for p in orig_params:
        if orig_params[p].vary and (filter is None or any([f in p for f in filter])):
            no_params += 1
            param_names.append(p)

    for i in range(no_params):
        ranges[param_names[i]] = {}
        # Initialize the progressbar and set the y-ticklabels.
        params = f.params

        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        stderr = stderr if stderr is not None else 0.1 * value
        stderr = stderr if stderr != 0 else 0.1 * value
        # Search for a value to the right which gives an increase greater than 1.
        search_value = value
        while True:
            search_value += 0.5*stderr
            new_value = fit_new_value(search_value, f, params, param_names[i], x_data, y_data, orig_value, func)
            if new_value > 1 - 0.5*(method.lower() == 'mle'):
                ranges[param_names[i]]['right'] = optimize.brentq(lambda *args: fit_new_value(*args) - (1 - 0.5*(method.lower() == 'mle')), value, search_value,
                                                                  args=(f, params, param_names[i], x_data,
                                                                        y_data, orig_value, func))
                break
        search_value = value
        # Do the same for the left
        while True:
            search_value -= 0.5*stderr
            new_value = fit_new_value(search_value, f, params, param_names[i], x_data, y_data, orig_value, func)
            if new_value > 1 - 0.5*(method.lower() == 'mle'):
                ranges[param_names[i]]['left'] = optimize.brentq(lambda *args: fit_new_value(*args) - (1 - 0.5*(method.lower() == 'mle')), search_value, value,
                                                                  args=(f, params, param_names[i], x_data,
                                                                        y_data, orig_value, func))
                break

        right = np.abs(ranges[param_names[i]]['right'] - value)
        left = np.abs(ranges[param_names[i]]['left'] - value)
        ranges[param_names[i]]['uncertainty'] = max(right, left)

        f.params = copy.deepcopy(orig_params)
        func(f, x_data, y_data, **fit_kws)
    # First, clear all uncertainty estimates
    for p in getattr(f, save_attr):
        getattr(f, save_attr)[p].stderr = None
    # Save all MINOS estimates
    for param_name in ranges.keys():
        getattr(f, save_attr)[param_name].stderr = ranges[param_name]['uncertainty']

def likelihood_walk(f, x, y, xerr=None, func=llh.poisson_llh, nsteps=2000, walkers=20,
                    verbose=True, filename=None):
    """Calculates the uncertainty on MLE-optimized parameter values
    by performing a random walk through parameter space and comparing
    the resulting loglikelihood values. For more information,
    see the emcee package.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model to be fitted to the data.
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
    verbose: boolean, optional
        If True, a progressbar is printed and updated every second.
        This progressbar displays the progress of the walk, with a primitive
        estimate of the remaining time in the calculation.
    filename: string, optional
        Filename where the random walk has to be saved. If *None*,
        the current time in seconds since January 1970 is used."""

    params = f.mle_fit
    var_names = []
    vars = []
    for key in params.keys():
        if params[key].vary:
            var_names.append(key)
            vars.append(params[key].value)
    ndim = len(vars)
    pos = mcmc.utils.sample_ball(vars, [1e-4] * len(vars), size=walkers)

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

    def lnprobList(fvars, groupParams, f, x, y, xerr, func, pbar):
        for val, n in zip(fvars, var_names):
            groupParams[n].value = val
        try:
            pbar += 1
        except:
            pass
        return likelihood_lnprob(groupParams, f, x, y, xerr, func)

    groupParams = lm.Parameters()
    for key in params.keys():
        groupParams[key] = PriorParameter(key,
                                          value=params[key].value,
                                          vary=params[key].vary,
                                          expr=params[key].expr,
                                          priormin=params[key].min,
                                          priormax=params[key].max)
    sampler = mcmc.EnsembleSampler(walkers, ndim, lnprobList,
                                   args=(groupParams, f, x, y, xerr, func, pbar))

    if filename is None:
        import time
        filename = '{}.h5'.format(time.time())
    else:
        filename = '.'.join(filename.split('.')[:-1]) + '.h5'

    if os.path.isfile(filename):
        with h5py.File(filename, 'a') as store:
            dset = store['data']
            offset = dset.len()
            pos = dset[-walkers:, :]
            dset.resize(offset + nsteps * walkers, axis=0)

            for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False)):
                result = result[0]
                dset[offset + i * walkers:offset + (i + 1) * walkers, :] = result
    else:
        with h5py.File(filename, 'w') as store:
            dset = store.create_dataset('data', (nsteps * walkers, ndim), dtype='float', chunks=True, compression='gzip', maxshape=(None, ndim))
            dset.attrs['format'] = np.array([f.encode('utf-8') for f in var_names])

            for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False)):
                result = result[0]
                dset[i * walkers:(i + 1) * walkers, :] = result
    try:
        pbar.finish()
    except:
        pass

    f.mle_fit = params
    f.params = params
