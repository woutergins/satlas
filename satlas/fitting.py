"""
Implementation of fitting routines specialised for BaseModel objects. Note that not all functions are loaded into the global satlas namespace.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import copy
import os
import warnings

from . import emcee as mcmc
from . import linkedmodel
from . import lmfit as lm
from . import loglikelihood as llh
from . import tqdm
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.misc import derivative


__all__ = ['chisquare_spectroscopic_fit', 'chisquare_fit', 'calculate_analytical_uncertainty',
           'likelihood_fit', 'likelihood_walk']
chisquare_warning_message = "The supplied dictionary for {} did not contain the necessary keys 'value' and 'uncertainty'."

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
        x = np.array(x)
        xerr = np.hstack((derivative(f, x, dx=1E-6) * xerr))
        bottom = np.sqrt(yerr * yerr + xerr * xerr)
    else:
        bottom = yerr
    return_value = (y - model) / bottom
    appended_values = f.get_chisquare_mapping()
    if appended_values is not None:
        return_value = np.append(return_value, appended_values)
    return return_value

def chisquare_spectroscopic_fit(f, x, y, xerr=None, func=None, verbose=True):
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
        Experimental data for the y-axis.

    Other parameters
    ----------------
    xerr: array_like, optional
        Error bars on *x*.
    func: function, optional
        Uses the provided function on the fitvalue to calculate the
        errorbars.

    Return
    ------
    success, message: bool and string
        Boolean indicating the success of the convergence, and the message
        from the optimizer."""
    y = np.hstack(y)
    yerr = np.sqrt(y)
    yerr[np.isclose(yerr, 0.0)] = 1.0
    return chisquare_fit(f, x, y, yerr=yerr, xerr=xerr, func=func, verbose=verbose)

def chisquare_fit(f, x, y, yerr=None, xerr=None, func=None, verbose=True):
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
    func: function, optional
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

    def iter_cb(params, iter, resid, *args, **kwargs):
        if verbose:
            progress.update(1)
            progress.set_description('Chisquare fitting in progress (' + str((resid**2).sum()) + ')')
        else:
            pass

    if verbose:
        progress = tqdm.tqdm(desc='Chisquare fitting in progress', leave=True)

    result = lm.minimize(chisquare_model, params, args=(f, x, np.hstack(y), np.hstack(yerr), xerr, func), iter_cb=iter_cb)
    f.params = copy.deepcopy(result.params)
    f.chisqr = copy.deepcopy(result.chisqr)

    success = False
    counter = 0
    while not success:
        result = lm.minimize(chisquare_model, params, args=(f, x, np.hstack(y), np.hstack(yerr), xerr, func), iter_cb=iter_cb)
        f.params = copy.deepcopy(result.params)
        success = np.isclose(result.chisqr, f.chisqr)
        f.chisqr = copy.deepcopy(result.chisqr)
        if counter > 10 and not success:
            break
    if verbose:
        progress.set_description('Chisquare fitting done')
        progress.close()

    f.chisq_res_par = copy.deepcopy(result.params)
    f.ndof = copy.deepcopy(result.nfree)
    f.redchi = copy.deepcopy(result.redchi)
    return success, result.message

##########################################
# MAXIMUM LIKELIHOOD ESTIMATION ROUTINES #
##########################################


class PriorParameter(lm.Parameter):

    # Extended the Parameter class from LMFIT to incorporate prior boundaries.

    def __init__(self, name=None, value=None, vary=True, min=None, max=None,
                 expr=None, priormin=None, priormax=None):
        super(PriorParameter, self).__init__(name=name, value=value,
                                             vary=vary, min=min,
                                             max=max, expr=expr)
        self.priormin = priormin
        self.priormax = priormax

    def __getstate__(self):
        return_value = super(PriorParameter, self).__getstate__()
        return_value += (self.priormin, self.priormax)
        return return_value

    def __setstate__(self, state):
        state_pass = state[:-2]
        self.priormin, self.priormax = state[-2:]
        super(PriorParameter, self).__setstate__(state_pass)

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
    x = np.array(x)
    y = np.array(y)
    key = hash(x.data.tobytes()) + hash(y.data.tobytes())
    if key in _x_err_calculation_stored:
        x_grid, y_grid, theta, rfft_g = _x_err_calculation_stored[key]
    else:
        if isinstance(f, linkedmodel.LinkedModel):
            x_grid = []
            y_grid, _ = np.meshgrid(y, theta_array)
            g = []
            for X, Y in zip(x, y):
                X_grid, theta = np.meshgrid(X, theta_array)
                X_grid = X_grid + xerr * theta
                x_grid.append(X_grid)
                g_top = (np.exp(-theta*theta * 0.5)).T
                g.append((g_top.T / (sqrt2pi * xerr)).T)
            g = np.vstack(g)
            rfft_g = np.fft.rfft(g)
        else:
            x_grid, theta = np.meshgrid(x, theta_array)
            y_grid, _ = np.meshgrid(y, theta_array)
            g_top = (np.exp(-theta*theta * 0.5)).T
            g = (g_top.T / (sqrt2pi * xerr)).T
            rfft_g = np.fft.rfft(g)
            x_grid = x_grid + xerr * theta
        _x_err_calculation_stored[key] = x_grid, y_grid, theta, rfft_g
    # Calculate the loglikelihoods for the grid of uncertainty.
    # Each column is a new datapoint.
    vals = func(y_grid, f(x_grid))
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
    try:
        lp = f.get_lnprior_mapping(params)
    except AttributeError:
        lp = f.lnprior()
        f.params = params
    if not np.isfinite(lp):
        return -np.inf
    res = lp + np.sum(likelihood_loglikelihood(f, x, y, xerr, func))
    return res

def likelihood_loglikelihood(f, x, y, xerr, func):
    """Given a parameters object, a Model object, experimental data
    and a loglikelihood function, calculates the loglikelihood for
    all data points.

    Parameters
    ----------
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
    # If a value is given to the uncertainty on the x-values, use the adapted
    # function.
    response = np.hstack(f(x))
    # If a value is given to the uncertainty on the x-values, use the adapted
    # function.
    if xerr is None or np.allclose(0, xerr):
        return_value = func(y, response)
    else:
        return_value = likelihood_x_err(f, x, y, xerr, func)
    return return_value

def likelihood_fit(f, x, y, xerr=None, func=llh.poisson_llh, method='powell', method_kws={}, walking=False, walk_kws={}, verbose=True):
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
        Defaults to 'powell'.
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
        return_val = -likelihood_lnprob(*args, **kwargs)
        return return_val

    def iter_cb(params, iter, resid, *args, **kwargs):
        if verbose:
            progress.update(1)
            progress.set_description('Likelihood fitting in progress (' + str(resid) + ')')
        else:
            pass

    y = np.hstack(y)
    params = f.params
    # Eliminate the estimated uncertainties
    for p in params:
        params[p].stderr = None
    if verbose:
        progress = tqdm.tqdm(leave=True, desc='Likelihood fitting in progress')

    result = lm.Minimizer(negativeloglikelihood, params, fcn_args=(f, x, y, xerr, func), iter_cb=iter_cb)
    result.scalar_minimize(method=method, **method_kws)
    f.params = result.params
    val = negativeloglikelihood(f.params, f, x, y, xerr, func)
    success = False
    counter = 0
    while not success:
        result = lm.Minimizer(negativeloglikelihood, f.params, fcn_args=(f, x, y, xerr, func), iter_cb=iter_cb)
        result.scalar_minimize(method=method, **method_kws)
        counter += 1
        f.params = result.params
        new_val = negativeloglikelihood(f.params, f, x, y, xerr, func)
        success = np.isclose(val, new_val)
        val = new_val
        # if not success and counter > 10:
        #     break
    if verbose:
        progress.set_description('Likelihood fitting done')
        progress.close()
    f.mle_fit = result.params
    f.mle_result = result.message
    f.mle_likelihood = negativeloglikelihood(f.params, f, x, y, xerr, func)

    if walking:
        likelihood_walk(f, x, y, xerr=xerr, func=func, **walk_kws)
    return success, result.message

############################
# UNCERTAINTY CALCULATIONS #
############################

def calculate_analytical_uncertainty(f, x, y, method='chisquare_spectroscopic', filter=None, fit_kws={}):
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
        params = copy.deepcopy(params)
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
        try:
            try:
                params_name = ' '.join(params_name)
            except:
                pass
            pbar.set_description(params_name + ' (' + str(value, return_value) + ')')
            pbar.update()
        except:
            pass
        return return_value

    # Save the original goodness-of-fit and parameters for later use
    mapping = {'chisquare_spectroscopic': (chisquare_spectroscopic_fit, 'chisqr', 'chisq_res_par'),
               'chisquare': (chisquare_fit, 'chisqr', 'chisq_res_par'),
               'mle': (likelihood_fit, 'mle_likelihood', 'mle_fit')}
    func, attr, save_attr = mapping.pop(method.lower(), (chisquare_spectroscopic_fit, 'chisqr', 'chisq_res_par'))
    fit_kws['verbose'] = False

    func(f, x, y, **fit_kws)


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

    params = copy.deepcopy(f.params)
    for i in range(no_params):
        ranges[param_names[i]] = {}
        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        stderr = stderr if stderr is not None else 0.01 * np.abs(value)
        stderr = stderr if stderr != 0 else 0.01 * np.abs(value)
        # Search for a value to the right which gives an increase greater than 1.
        search_value = value
        success = False
        with tqdm.tqdm(leave=True, desc=param_names[i] + ' (searching right)', mininterval=0) as pbar:
            while True:
                search_value += 0.5*stderr
                if search_value > orig_params[param_names[i]].max:
                    pbar.set_description(param_names[i] + ' (right limit reached)')
                    pbar.update(1)
                    search_value = orig_params[param_names[i]].max
                    ranges[param_names[i]]['right'] = search_value
                    break
                new_value = fit_new_value(search_value, f, params, param_names[i], x, y, orig_value, func) - (1 - 0.5*(method.lower() == 'mle'))
                pbar.set_description(param_names[i] + ' (searching right: ' + str(search_value) + ')')
                pbar.update(1)
                if new_value > 0:
                    pbar.set_description(param_names[i] + ' (finding root)')
                    pbar.update(1)
                    result, output = optimize.ridder(lambda *args: fit_new_value(*args) - (1 - 0.5*(method.lower() == 'mle')),
                                                     value, search_value,
                                                     args=(f, params, param_names[i], x, y, orig_value, func),
                                                     full_output=True)
                    pbar.set_description(param_names[i] + ' (root found: ' + str(result) + ')')
                    pbar.update(1)
                    ranges[param_names[i]]['right'] = result
                    success = output.converged
                    break
        search_value = value
        # Do the same for the left
        with tqdm.tqdm(leave=True, desc=param_names[i] + ' (searching left)', mininterval=0) as pbar:
            while True:
                search_value -= 0.5*stderr
                if search_value < orig_params[param_names[i]].min:
                    pbar.set_description(param_names[i] + ' (left limit reached)')
                    pbar.update(1)
                    search_value = orig_params[param_names[i]].min
                    ranges[param_names[i]]['left'] = search_value
                    success = False
                    break
                new_value = fit_new_value(search_value, f, params, param_names[i], x, y, orig_value, func)
                if new_value > 1 - 0.5*(method.lower() == 'mle'):
                    pbar.set_description(param_names[i] + ' (finding root)')
                    pbar.update(1)
                    result, output = optimize.ridder(lambda *args: fit_new_value(*args) - (1 - 0.5*(method.lower() == 'mle')),
                                                     value, search_value,
                                                     args=(f, params, param_names[i], x, y, orig_value, func),
                                                     full_output=True)
                    pbar.set_description(param_names[i] + ' (root found: ' + str(result) + ')')
                    pbar.update(1)
                    ranges[param_names[i]]['left'] = result
                    success = success * output.converged
                    break

        if not success:
            print("Warning: boundary calculation did not fully succeed for " + param_names[i])
        right = np.abs(ranges[param_names[i]]['right'] - value)
        left = np.abs(ranges[param_names[i]]['left'] - value)
        ranges[param_names[i]]['uncertainty'] = max(right, left)
        ranges[param_names[i]]['value'] = orig_params[param_names[i]].value

        f.params = copy.deepcopy(orig_params)
    # First, clear all uncertainty estimates
    for p in orig_params:
        orig_params[p].stderr = None
    # Save all MINOS estimates
    for param_name in ranges.keys():
        orig_params[param_name].stderr = ranges[param_name]['uncertainty']
        orig_params[param_name].value = ranges[param_name]['value']
    setattr(f, save_attr, copy.deepcopy(orig_params))
    f.params = copy.deepcopy(orig_params)

def likelihood_walk(f, x, y, xerr=None, func=llh.poisson_llh, nsteps=2000, walkers=20,
                    filename=None):
    """Calculates the uncertainty on MLE-optimized parameter values
    by performing a random walk through parameter space and comparing
    the resulting loglikelihood values. For more information,
    see the emcee package. The data from the random walk is saved in a
    file, as defined with the *filename*.

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
    filename: string, optional
        Filename where the random walk has to be saved. If *None*,
        the current time in seconds since January 1970 is used.

    Note
    ----
    The parameters associated with the MLE fit are not updated
    with the uncertainty as estimated by this method."""

    params = f.mle_fit
    var_names = []
    vars = []
    for key in params.keys():
        if params[key].vary:
            var_names.append(key)
            vars.append(params[key].value)
    ndim = len(vars)
    pos = mcmc.utils.sample_ball(vars, [1e-4] * len(vars), size=walkers)
    for i in range(pos.shape[1]):
        pos[:, i] = np.where(pos[:, i] < params[var_names[i]].min, params[var_names[i]].min+(1E-5), pos[:, i])
        pos[:, i] = np.where(pos[:, i] > params[var_names[i]].max, params[var_names[i]].max-(1E-5), pos[:, i])

    def lnprobList(fvars, groupParams, f, x, y, xerr, func):
        for val, n in zip(fvars, var_names):
            groupParams[n].value = val
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
                                   args=(groupParams, f, x, y, xerr, func))

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

            with tqdm.tqdm(total=nsteps, desc='Walk', leave=True) as pbar:
                for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False)):
                    result = result[0]
                    dset[offset + i * walkers:offset + (i + 1) * walkers, :] = result
                    pbar.update(1)
    else:
        with h5py.File(filename, 'w') as store:
            dset = store.create_dataset('data', (nsteps * walkers, ndim), dtype='float', chunks=True, compression='gzip', maxshape=(None, ndim))
            dset.attrs['format'] = np.array([f.encode('utf-8') for f in var_names])

            with tqdm.tqdm(total=nsteps, desc='Walk', leave=True) as pbar:
                for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False)):
                    result = result[0]
                    dset[i * walkers:(i + 1) * walkers, :] = result
                    pbar.update(1)

    f.mle_fit = params
    f.params = params
