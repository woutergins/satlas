"""
Implementation of various functions that ease the work, but do not belong in one of the other modules.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""
import emcee as mcmc
import lmfit as lm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from scipy import optimize
import copy
import progressbar

c = 299792458.0
h = 6.62606957 * (10 ** -34)
q = 1.60217657 * (10 ** -19)

cmap = mpl.colors.ListedColormap(['#A6CEE3', '#1F78B4', '#B2DF8A'])
invcmap = mpl.colors.ListedColormap(['#B2DF8A', '#1F78B4', '#A6CEE3'])

__all__ = ['weighted_average',
           'generate_correlation_plot',
           'generate_spectrum',
           'concat_results',
           'poisson_interval']

def weighted_average(x, sigma, axis=None):
    r"""Takes the weighted average of an array of values and the associated
    errors. Calculates the scatter and statistical error, and returns
    the greater of these two values.

    Parameters
    ----------
    x: array_like
        Array-like assortment of measured values, is transformed into a
        1D-array.
    sigma: array_like
        Array-like assortment of errors on the measured values, is transformed
        into a 1D-array.

    Returns
    -------
    tuple
        Returns a tuple (weighted_average, uncertainty), with the uncertainty
        being the greater of the uncertainty calculated from the statistical
        uncertainty and the scattering uncertainty.

    Note
    ----
    The formulas used are

    .. math::

        \left\langle x\right\rangle_{weighted} &= \frac{\sum_{i=1}^N \frac{x_i}
                                                                 {\sigma_i^2}}
                                                      {\sum_{i=1}^N \frac{1}
                                                                {\sigma_i^2}}

        \sigma_{stat}^2 &= \frac{1}{\sum_{i=1}^N \frac{1}{\sigma_i^2}}

        \sigma_{scatter}^2 &= \frac{\sum_{i=1}^N \left(\frac{x_i-\left\langle
                                                    x\right\rangle_{weighted}}
                                                      {\sigma_i}\right)^2}
               {\left(N-1\right)\sum_{i=1}^N \frac{1}{\sigma_i^2}}"""
    # x = np.ravel(x)
    # sigma = np.ravel(sigma)
    Xstat = (1 / sigma**2).sum(axis=axis)
    Xm = (x / sigma**2).sum(axis=axis) / Xstat
    # Xscatt = (((x - Xm) / sigma)**2).sum() / ((1 - 1.0 / len(x)) * Xstat)
    Xscatt = (((x - Xm) / sigma)**2).sum(axis=axis) / ((len(x) - 1) * Xstat)
    Xstat = 1 / Xstat
    return Xm, np.maximum.reduce([Xstat, Xscatt], axis=axis) ** 0.5

def _make_axes_grid(no_variables, padding=2, cbar_size=0.5, axis_padding=0.5, cbar=True):
    """Makes a triangular grid of axes, with a colorbar axis next to it.

    Parameters
    ----------
    no_variables: int
        Number of variables for which to generate a figure.
    padding: float
        Padding around the figure (in cm).
    cbar_size: float
        Width of the colorbar (in cm).
    axis_padding: float
        Padding between axes (in cm).

    Returns
    -------
    fig, axes, cbar: tuple
        Tuple containing the figure, a 2D-array of axes and the colorbar axis."""

    # Convert to inches.
    padding, cbar_size, axis_padding = (padding * 0.393700787,
                                        cbar_size * 0.393700787,
                                        axis_padding * 0.393700787)
    if not cbar:
        cbar_size = 0

    # Generate the figure, convert padding to percentages.
    fig = plt.figure()

    axis_size = 4
    width = axis_size * no_variables + cbar_size + 2 * padding + no_variables * axis_padding
    height = axis_size * no_variables + 2 * padding
    if no_variables > 1:
        height += (no_variables - 1) * axis_padding
    fig.set_size_inches(width, height, forward=True)

    cbar_size = cbar_size / fig.get_figwidth()
    left_padding = padding / fig.get_figwidth()
    left_axis_padding = axis_padding / fig.get_figwidth()
    up_padding = padding / fig.get_figheight()
    up_axis_padding = axis_padding / fig.get_figheight()
    axis_size_left = axis_size / fig.get_figwidth()
    axis_size_up = axis_size / fig.get_figheight()

    # Pre-allocate a 2D-array to hold the axes.
    axes = np.array([[None for _ in range(no_variables)] for _ in range(no_variables)],
                    dtype='object')

    for i, I in zip(range(no_variables), reversed(range(no_variables))):
        for j in reversed(range(no_variables)):
            # Only create axes on the lower triangle.
            if I + j < no_variables:
                # Share the x-axis with the plot on the diagonal,
                # directly above the plot.
                sharex = axes[j, j] if i != j else None
                # Share the y-axis among the 2D maps along one row,
                # but not the plot on the diagonal!
                sharey = axes[i, i-1] if (i != j and i-1 != j) else None
                # Determine the place and size of the axes
                left_edge = j * axis_size_left + left_padding
                bottom_edge = I * axis_size_up + up_padding
                if j > 0:
                    left_edge += j * left_axis_padding
                if I > 0:
                    bottom_edge += I * up_axis_padding

                a = plt.axes([left_edge, bottom_edge, axis_size_left, axis_size_up],
                             sharex=sharex, sharey=sharey)
                plt.setp(a.xaxis.get_majorticklabels(), rotation=45)
                plt.setp(a.yaxis.get_majorticklabels(), rotation=45)
            else:
                a = None
            axes[i, j] = a

    axes = np.array(axes)
    for a in axes[:-1, :].flatten():
        if a is not None:
            plt.setp(a.get_xticklabels(), visible=False)
    for a in axes[:, 1:].flatten():
        if a is not None:
            plt.setp(a.get_yticklabels(), visible=False)
    left_edge = no_variables*(axis_size_left+left_axis_padding)+left_padding
    bottom_edge = up_padding
    width = cbar_size

    height = 1 - 2 * up_padding

    if cbar:
        cbar = plt.axes([left_edge, bottom_edge, width, height])
        plt.setp(cbar.get_xticklabels(), visible=False)
        plt.setp(cbar.get_yticklabels(), visible=False)
    else:
        cbar = None
    return fig, axes, cbar

def generate_correlation_map(spectrum, x_data, y_data, method='chisquare', filter=None, resolution_diag=20, resolution_map=15, fit_kws={}):
    """Generates a correlation map for either the chisquare or the MLE method.
    On the diagonal, the chisquare or loglikelihood is drawn as a function of one fixed parameter.
    Refitting to the data each time gives the points on the line. A dashed line is drawn on these
    plots, with the intersection with the plots giving the correct confidence interval for the
    parameter. In solid lines, the interval estimated by the fitting routine is drawn.
    On the offdiagonal, two parameters are fixed and the spectrum is again fitted to the data.
    The change in chisquare/loglikelihood is mapped to 1, 2 and 3 sigma contourmaps.

    Parameters
    ----------
    spectrum: :class:`.Spectrum`
        Instance of the spectrum for which the contour map has to be generated.
    x_data: array_like or list of array_likes
        Data on the x-axis for the fit. Must be appropriate input for the *spectrum*
        instance.
    y_data: array_like or list of array_likes
        Data on the y-axis for the fit. Must be appropriate input for the *spectrum*
        instance.

    Other parameters
    ----------------
    method: {'chisquare', 'mle'}
        Chooses between generating the map for the chisquare routine or for
        the likelihood routine.
    filter: list of strings
        Only the parameters matching the names given in this list will be used
        to generate the maps.
    resolution_diag: int
        Number of points for the line plot on each diagonal.
    resolution_map: int
        Number of points along each dimension for the meshgrids.
    fit_kws: dictionary
        Dictionary of keywords to pass on to the fitting routine."""
    from . import fitting

    def fit_new_value(value, spectrum, params, params_name, x, y, orig_value, func):
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
        spectrum.params = params
        success = False
        counter = 0
        while not success:
            success, message = func(spectrum, x, y, **fit_kws)
            counter += 1
            if counter > 10:
                success = True
                print('Fitting did not converge, carrying on...')
        return_value = getattr(spectrum, attr) - orig_value
        return return_value

    # Save the original goodness-of-fit and parameters for later use
    mapping = {'chisquare': (fitting.chisquare_spectroscopic_fit, 'chisqr'),
               'mle': (fitting.likelihood_fit, 'mle_likelihood')}
    func, attr = mapping.pop(method.lower(), (fitting.chisquare_spectroscopic_fit, 'chisqr'))
    title = r'{} = ${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'

    func(spectrum, x_data, y_data, **fit_kws)

    orig_value = getattr(spectrum, attr)
    orig_params = copy.deepcopy(spectrum.params)

    ranges = {}

    # Select all variable parameters, generate the figure
    param_names = []
    no_params = 0
    for p in orig_params:
        if orig_params[p].vary and (filter is None or any([f in p for f in filter])):
            no_params += 1
            param_names.append(p)

    fig, axes, cbar = _make_axes_grid(no_params, axis_padding=2.0, cbar=no_params > 1)

    # Make the plots on the diagonal: plot the chisquare/likelihood
    # for the best fitting values while setting one parameter to
    # a fixed value.
    for i in range(no_params):
        ranges[param_names[i]] = {}
        # Initialize the progressbar and set the y-ticklabels.
        widgets = [param_names[i]+': ',
                   progressbar.Percentage(),
                   ' ',
                   progressbar.Bar(marker=progressbar.RotatingMarker()),
                   ' ',
                   progressbar.AdaptiveETA()]
        params = spectrum.params
        ax = axes[i, i]
        ax.set_title(param_names[i])
        plt.setp(ax.get_yticklabels(), visible=True)
        if method.lower() == 'chisquare':
            ax.set_ylabel(r'$\Delta\chi^2$')
        else:
            ax.set_ylabel(r'$\Delta\mathcal{L}$')

        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        stderr = stderr if stderr is not None else 0.1 * value
        stderr = stderr if stderr != 0 else 0.1 * value
        # Search for a value to the right which gives an increase greater than 1.
        search_value = value
        while True:
            search_value += 0.5*stderr
            new_value = fit_new_value(search_value, spectrum, params, param_names[i], x_data, y_data, orig_value, func)
            if new_value > 1 - 0.5*(method.lower() == 'mle'):
                print('Found value on the right: {:.2f}, gives a value of {:.2f}'.format(search_value, new_value))
                ranges[param_names[i]]['right'] = optimize.brentq(lambda *args: fit_new_value(*args) - (1 - 0.5*(method.lower() == 'mle')), value, search_value,
                                                                  args=(spectrum, params, param_names[i], x_data,
                                                                        y_data, orig_value, func))
                print('Found optimal right value: {:.2f}'.format(ranges[param_names[i]]['right']))
                break
        search_value = value
        # Do the same for the left
        while True:
            search_value -= 0.5*stderr
            new_value = fit_new_value(search_value, spectrum, params, param_names[i], x_data, y_data, orig_value, func)
            if new_value > 1 - 0.5*(method.lower() == 'mle'):
                print('Found value on the left: {:.2f}, gives a value of {:.2f}'.format(search_value, new_value))
                ranges[param_names[i]]['left'] = optimize.brentq(lambda *args: fit_new_value(*args) - (1 - 0.5*(method.lower() == 'mle')), search_value, value,
                                                                  args=(spectrum, params, param_names[i], x_data,
                                                                        y_data, orig_value, func))
                print('Found optimal left value: {:.2f}'.format(ranges[param_names[i]]['left']))
                break

        # Keep the parameter fixed, and let it vary (with given number of points)
        # in a deviation of 4 sigma in both directions.
        # middle = (ranges[param_names[i]]['left'] + ranges[param_names[i]]['right']) * 0.5
        right = np.abs(ranges[param_names[i]]['right'] - value)
        left = np.abs(ranges[param_names[i]]['left'] - value)
        params[param_names[i]].vary = False

        value_range = np.linspace(value - 4 * left, value + 4 * right, resolution_diag)
        chisquare = np.zeros(len(value_range))
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(value_range)).start()
        # Calculate the new value, and store it in the array. Update the progressbar.
        for j, v in enumerate(value_range):
            chisquare[j] = fit_new_value(v, spectrum, params, param_names[i],
                                         x_data, y_data, orig_value, func)
            pbar += 1
        pbar.finish()
        # Plot the result
        ax.plot(value_range, chisquare)

        # For chisquare, an increase of 1 corresponds to 1 sigma errorbars.
        # For the loglikelihood, this need to be reduced by a factor of 2.
        ax.axhline(1 - 0.5*(method.lower()=='mle'), ls="dashed")
        # Indicate the used interval.
        ax.axvline(value + right)
        ax.axvline(value - left)
        ax.set_title(title.format(param_names[i], value, left, right))
        # Restore the parameters.
        spectrum.params = copy.deepcopy(orig_params)
        func(spectrum, x_data, y_data, **fit_kws)

    for i, j in zip(*np.tril_indices_from(axes, -1)):
        widgets = [param_names[j]+' ' + param_names[i]+': ', progressbar.Percentage(), ' ',
                   progressbar.Bar(marker=progressbar.RotatingMarker()),
                   ' ', progressbar.AdaptiveETA()]
        params = copy.deepcopy(orig_params)
        ax = axes[i, j]
        x_name = param_names[j]
        y_name = param_names[i]
        if j == 0:
            ax.set_ylabel(y_name)
        if i == no_params - 1:
            ax.set_xlabel(x_name)
        # middle = (ranges[x_name]['left'] + ranges[x_name]['right']) * 0.5
        value = params[x_name].value
        right = np.abs(ranges[x_name]['right'] - value)
        left = np.abs(ranges[x_name]['left'] - value)
        params[x_name].vary = False
        x_range = np.linspace(value - 4 * left, value + 4 * right, resolution_map)

        value = params[y_name].value
        right = np.abs(ranges[y_name]['right'] - value)
        left = np.abs(ranges[y_name]['left'] - value)
        params[y_name].vary = False
        y_range = np.linspace(value - 4 * left, value + 4 * right, resolution_map)

        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros(X.shape)
        i_indices, j_indices = np.indices(Z.shape)
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(Z.flatten())).start()
        for k, l in zip(i_indices.flatten(), j_indices.flatten()):
            x = X[k, l]
            y = Y[k, l]
            Z[k, l] = fit_new_value([x, y], spectrum, params, [x_name, y_name],
                                    x_data, y_data, orig_value, func)
            pbar += 1
        pbar.finish()
        Z = -Z
        bounds = [-9, -6, -2, 0]
        if method.lower() == 'mle':
            bounds = [b * 0.5 for b in bounds]
        norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)
        contourset = ax.contourf(X, Y, Z, bounds, cmap=invcmap, norm=norm)
        spectrum.params = orig_params
        func(spectrum, x_data, y_data, **fit_kws)
    try:
        cbar = plt.colorbar(contourset, cax=cbar, orientation='vertical')
        cbar.ax.yaxis.set_ticks([0, 1/6, 0.5, 5/6])
        cbar.ax.set_yticklabels(['', r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])
    except:
        pass

    return fig, axes, cbar

def generate_correlation_plot(data, filter=None):
    """Given the random walk data, creates a triangle plot: distribution of
    a single parameter on the diagonal axes, 2D contour plots with 1, 2 and
    3 sigma contours on the off-diagonal. The 1-sigma limits based on the
    percentile method are also indicated, as well as added to the title.

    Parameters
    ----------
    data: DataFrame
        DataFrame collecting all the information on the random walk for each
        parameter.
    filter: list of str, optional
        If supplied, only this list of columns is used for the plot.

    Returns
    -------
    figure
        Returns the MatPlotLib figure created."""
    if filter is not None:
        filter = [c for f in filter for c in data.columns.tolist() if f in c]
        data = data[filter]

    fig, axes, cbar = _make_axes_grid(len(data.columns), axis_padding=0)

    for i, val in enumerate(data.columns):
        ax = axes[i, i]
        x = data[val]
        bins = 50
        ax.hist(x.values, bins)
        q = [16.0, 50.0, 84.0]
        q16, q50, q84 = np.percentile(x.values, q)

        title = x.name + r' = ${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'
        ax.set_title(title.format(q50, q50-q16, q84-q50))
        qvalues = [q16, q50, q84]
        for q in qvalues:
            ax.axvline(q, ls="dashed")
        ax.set_yticks([])
        ax.set_yticklabels([])

    for i, j in zip(*np.tril_indices_from(axes, -1)):
        ax = axes[i, j]
        x = data[data.columns[j]].values
        y = data[data.columns[i]].values
        if j == 0:
            ax.set_ylabel(data.columns[i])
        if i == len(data.columns) - 1:
            ax.set_xlabel(data.columns[j])
        X = np.linspace(x.min(), x.max(), bins + 1)
        Y = np.linspace(y.min(), y.max(), bins + 1)
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),
                                 weights=None)
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
        X, Y = X[:-1], Y[:-1]
        H = (H - H.min()) / (H.max() - H.min())

        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]

        bounds = np.concatenate([[H.max()], V])[::-1]
        norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)

        contourset = ax.contourf(X1, Y1, H.T, bounds, cmap=invcmap, norm=norm)
    cbar = plt.colorbar(contourset, cax=cbar, orientation='vertical')
    cbar.ax.yaxis.set_ticks([0, 1/6, 0.5, 5/6])
    cbar.ax.set_yticklabels(['', r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])
    return fig, axes, cbar

def generate_spectrum(spectrum, x, number_of_counts, nwalkers=100):
    """Generates a spectrum by random sampling from the provided hyperfine
    spectrum and range. The total number of counts for the generated spectrum
    is required.

    Parameters
    ----------
    spectrum: SingleSpectrum
        An instance of SingleSpectrum, which gives the probability distribution
        from which the random samples are drawn.
    x: NumPy array
        NumPy array representing the bin centers for the spectrum.
    number_of_counts: int
        Parameter controlling the total number of counts in the spectrum.
    nwalkers: int, optional
        Number of walkers for the random sampling algorithm from emcee.

    Returns
    -------
    y: NumPy array
        Array containing the number of counts corresponding to each value
        in x.
    """
    binsize = x[1] - x[0]  # Need the binsize for accurate lnprob boundaries

    def lnprob(x, left, right):
        if x > right + binsize / 2 or x < left - binsize / 2:
            return -np.inf  # Make sure only to draw from the provided range
        else:
            return np.log(spectrum(x))  # No need to normalize lnprob!
    ndim = 1
    pos = (np.random.rand(nwalkers) * (x.max() - x.min())
           + x.min()).reshape((nwalkers, ndim))
    sampler = mcmc.EnsembleSampler(nwalkers, ndim, lnprob,
                                   args=(x.min(), x.max()))
    # Burn-in
    pos, prob, state = sampler.run_mcmc(pos, 1000)
    sampler.reset()
    # Making sure not to do too much work! Divide requested number of samples
    # by number of walkers, make sure it's a higher integer.
    sampler.run_mcmc(pos, np.ceil(number_of_counts / nwalkers))
    samples = sampler.flatchain[-number_of_counts:]
    # Bin the samples
    bins = x - binsize / 2
    bins = np.append(bins, bins[-1] + binsize)
    y, _ = np.histogram(samples, bins)
    return y

def concat_results(list_of_results, index=None):
    """Given a list of DataFrames, use the supplied index
    to concatenate the DataFrames.

    Parameters
    ----------
    list_of_results: list of pandas Dataframes
        List of DataFrames to be concatenated.
    index: list
        List of keys to use as row-indices.

    Returns
    -------
    concatenated_frames: DataFrame
        Concatenated DataFrame"""
    if index is None:
        index = range(1, len(list_of_results) + 1)
    concatenated_frames = pd.concat(list_of_results, keys=index)
    return concatenated_frames

def poisson_interval(data, alpha=0.32):
    """Calculates the confidence interval
    for the mean of a Poisson distribution.

    Parameters
    ----------
    data: array_like
        Data giving the mean of the Poisson distributions.
    alpha: float
        Significance level of interval. Defaults to
        one sigma (0.32).

    Returns
    -------
    low, high: array_like
        Lower and higher limits for the interval."""
    a = alpha
    low, high = (chi2.ppf(a / 2, 2 * data) / 2,
                 chi2.ppf(1 - a / 2, 2 * data + 2) / 2)
    low = np.nan_to_num(low)
    return low, high
