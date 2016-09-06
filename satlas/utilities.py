"""
Implementation of various functions that ease the work, but do not belong in one of the other modules.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import copy

from . import emcee as mcmc
from . import lmfit as lm
from . import tqdm
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2


c = 299792458.0
h = 6.62606957 * (10 ** -34)
q = 1.60217657 * (10 ** -19)

cmap = mpl.colors.ListedColormap(['#A6CEE3', '#1F78B4', '#B2DF8A'])
cmap.set_over('#A6CEE3')
cmap.set_under('#B2DF8A')
invcmap = mpl.colors.ListedColormap(['#B2DF8A', '#1F78B4', '#A6CEE3'])
invcmap.set_under('#A6CEE3')
invcmap.set_over('#B2DF8A')
inv_color_list = ['#7acfff', '#fff466', '#00c48f', '#ff8626', '#ff9cd3', '#0093e6']
color_list = [c for c in reversed(inv_color_list)]
cmap = mpl.colors.ListedColormap(color_list)
cmap.set_over(color_list[-1])
cmap.set_under(color_list[0])
invcmap = mpl.colors.ListedColormap(inv_color_list)
invcmap.set_over(inv_color_list[-1])
invcmap.set_under(inv_color_list[0])
# cmap = mpl.colors.ListedColormap(['#0072B2', '#CC79A7', '#D55E00', '#009E73', '#F0E442', '#56B4E9'])
# cmap.set_under('#56B4E9')
# cmap.set_over('#0072B2')
# invcmap = mpl.colors.ListedColormap(['#56B4E9', '#F0E442', '#009E73', '#D55E00', '#CC79A7', '#0072B2'])
# invcmap.set_over('#56B4E9')
# invcmap.set_under('#0072B2')

__all__ = ['weighted_average',
           'generate_correlation_map',
           'generate_correlation_plot',
           'generate_spectrum',
           'poisson_interval',
           'load_model',
           'beta',
           'dopplerfactor']

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

def _make_axes_grid(no_variables, padding=0, cbar_size=0.5, axis_padding=0.5, cbar=True):
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
    padding = 1

    axis_size_left = (fig.get_figwidth()-padding - 0*(no_variables + 1) * padding) / no_variables
    axis_size_up = (fig.get_figheight()-padding - 0*(no_variables + 1) * padding) / no_variables

    cbar_size = cbar_size / fig.get_figwidth()
    left_padding = padding * 0.5 / fig.get_figwidth()
    left_axis_padding = axis_padding / fig.get_figwidth()
    up_padding = padding * 0.5 / fig.get_figheight()
    up_axis_padding = 0*axis_padding / fig.get_figheight()
    axis_size_left = axis_size_left / fig.get_figwidth()
    axis_size_up = axis_size_up / fig.get_figheight()

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
            if i == j:
                a.yaxis.tick_right()
                a.yaxis.set_label_position('right')
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

    height = axis_size_up * len(axes) + up_padding * (len(axes) - 1)

    cbar_width = axis_size_left * 0.1
    if cbar:
        cbar = plt.axes([1-cbar_width-padding*0.5/fig.get_figwidth(), padding*0.5/fig.get_figheight()+axis_size_up*1.5, cbar_width, axis_size_up*(no_variables-1)-axis_size_up*0.5])
        plt.setp(cbar.get_xticklabels(), visible=False)
        plt.setp(cbar.get_yticklabels(), visible=False)
    else:
        cbar = None
    return fig, axes, cbar

def generate_correlation_map(f, x_data, y_data, method='chisquare_spectroscopic', filter=None, resolution_diag=20, resolution_map=15, fit_args=tuple(), fit_kws={}, distance=5, npar=1):
    """Generates a correlation map for either the chisquare or the MLE method.
    On the diagonal, the chisquare or loglikelihood is drawn as a function of one fixed parameter.
    Refitting to the data each time gives the points on the line. A dashed line is drawn on these
    plots, with the intersection with the plots giving the correct confidence interval for the
    parameter. In solid lines, the interval estimated by the fitting routine is drawn.
    On the offdiagonal, two parameters are fixed and the model is again fitted to the data.
    The change in chisquare/loglikelihood is mapped to 1, 2 and 3 sigma contourmaps.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Instance of the model for which the contour map has to be generated.
    x_data: array_like or list of array_likes
        Data on the x-axis for the fit. Must be appropriate input for *f*.
    y_data: array_like or list of array_likes
        Data on the y-axis for the fit. Must be appropriate input for *f*.

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
        Dictionary of keywords to pass on to the fitting routine.
    npar: int
        Number of parameters for which simultaneous predictions need to be made.
        Influences the uncertainty estimates from the parabola."""
    from . import fitting

    # Save the original goodness-of-fit and parameters for later use
    mapping = {'chisquare_spectroscopic': (fitting.chisquare_spectroscopic_fit, 'chisqr'),
               'chisquare': (fitting.chisquare_fit, 'chisqr'),
               'mle': (fitting.likelihood_fit, 'mle_likelihood')}
    func, attr = mapping.pop(method.lower(), (fitting.chisquare_spectroscopic_fit, 'chisqr'))
    title = '{}\n${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$'
    fit_kws['verbose'] = False
    fit_kws['hessian'] = False

    to_save = {'mle': ('mle_fit', 'mle_result')}
    to_save = to_save.pop(method.lower(), ('chisq_res_par', 'ndof', 'redchi'))
    saved = [copy.deepcopy(getattr(f, attr)) for attr in to_save]

    func(f, x_data, y_data, *fit_args, **fit_kws)

    orig_value = getattr(f, attr)
    orig_params = copy.deepcopy(f.params)
    state = fitting._get_state(f, method=method.lower())

    ranges = {}

    chifunc = lambda x: chi2.cdf(x, npar) - 0.682689492 # Calculate 1 sigma boundary
    boundary = optimize.root(chifunc, npar).x[0] * 0.5 if method.lower() == 'mle' else optimize.root(chifunc, npar).x[0]
    # Select all variable parameters, generate the figure
    param_names = []
    no_params = 0
    for p in orig_params:
        if orig_params[p].vary and (filter is None or any([f in p for f in filter])):
            no_params += 1
            param_names.append(p)
    fig, axes, cbar = _make_axes_grid(no_params, axis_padding=0, cbar=no_params > 1)

    # Make the plots on the diagonal: plot the chisquare/likelihood
    # for the best fitting values while setting one parameter to
    # a fixed value.
    saved_params = copy.deepcopy(f.params)
    function_kws = {'method': method.lower(), 'func_args': fit_args, 'func_kwargs': fit_kws}
    function_kws['orig_stat'] = orig_value
    for i in range(no_params):
        params = copy.deepcopy(saved_params)
        ranges[param_names[i]] = {}

        # Set the y-ticklabels.
        ax = axes[i, i]
        ax.set_title(param_names[i])
        plt.setp(ax.get_yticklabels(), visible=True)
        if method.lower().startswith('chisquare'):
            ax.set_ylabel(r'$\Delta\chi^2$')
        else:
            ax.set_ylabel(r'$\Delta\mathcal{L}$')

        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        stderr = stderr if stderr is not None else 0.01 * np.abs(value)
        stderr = stderr if stderr != 0 else 0.01 * np.abs(value)

        result_left, success_left = fitting._find_boundary(-stderr, param_names[i], boundary, f, x_data, y_data, function_kwargs=function_kws)
        result_right, success_right = fitting._find_boundary(stderr, param_names[i], boundary, f, x_data, y_data, function_kwargs=function_kws)
        success = success_left * success_right
        ranges[param_names[i]]['left'] = result_left
        ranges[param_names[i]]['right'] = result_right
        print(result_left, result_right)

        if not success:
            print("Warning: boundary calculation did not fully succeed for " + param_names[i])
        right = np.abs(ranges[param_names[i]]['right'] - value)
        left = np.abs(ranges[param_names[i]]['left'] - value)
        params[param_names[i]].vary = False

        left_val, right_val = max(value - distance * left, orig_params[param_names[i]].min), min(value + distance * right, orig_params[param_names[i]].max)
        ranges[param_names[i]]['right_val'] = right_val
        ranges[param_names[i]]['left_val'] = left_val
        value_range = np.linspace(left_val, right_val, resolution_diag)
        value_range = np.sort(np.append(value_range, np.array([value - left, value + right])))
        chisquare = np.zeros(len(value_range))
        # Calculate the new value, and store it in the array. Update the progressbar.
        with tqdm.tqdm(value_range, desc=param_names[i], leave=True) as pbar:
            for j, v in enumerate(value_range):
                chisquare[j] = fitting.calculate_updated_statistic(v, param_names[i], f, x_data, y_data, **function_kws)
                pbar.update(1)
        # Plot the result
        ax.plot(value_range, chisquare)

        c = next(ax._get_lines.prop_cycler)['color']
        # ax.axhline(boundary, ls="dashed", color=color)
        # Indicate the used interval.
        ax.axvline(value + right, ls="dashed", color=c)
        ax.axvline(value - left, ls="dashed", color=c)
        ax.axvline(value, ls="dashed", color=c)
        ax.set_title(title.format(param_names[i], value, left, right))
        # Restore the parameters.
        fitting._set_state(f, state, method=method.lower())

    for i, j in zip(*np.tril_indices_from(axes, -1)):
        params = copy.deepcopy(orig_params)
        ax = axes[i, j]
        x_name = param_names[j]
        y_name = param_names[i]
        if j == 0:
            ax.set_ylabel(y_name)
        if i == no_params - 1:
            ax.set_xlabel(x_name)
        right = ranges[x_name]['right_val']
        left = ranges[x_name]['left_val']
        x_range = np.linspace(left, right, resolution_map)

        right = ranges[y_name]['right_val']
        left = ranges[y_name]['left_val']
        y_range = np.linspace(left, right, resolution_map)

        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros(X.shape)
        i_indices, j_indices = np.indices(Z.shape)
        with tqdm.tqdm(i_indices.flatten(), desc=param_names[j]+' ' + param_names[i], leave=True) as pbar:
            for k, l in zip(i_indices.flatten(), j_indices.flatten()):
                x = X[k, l]
                y = Y[k, l]
                Z[k, l] = fitting.calculate_updated_statistic([x, y], [x_name, y_name], f, x_data, y_data, **function_kws)
                pbar.update(1)
        Z = -Z
        npar = 1
        bounds = []
        for bound in [0.997300204, 0.954499736, 0.682689492]:
            chifunc = lambda x: chi2.cdf(x, npar) - bound # Calculate 1 sigma boundary
            bounds.append(-optimize.root(chifunc, npar).x[0])
        # bounds = sorted([-number*number for number in np.arange(1, 9, .1)])
        bounds.append(1)
        if method.lower() == 'mle':
            bounds = [b * 0.5 for b in bounds]
        norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)
        contourset = ax.contourf(X, Y, Z, bounds, cmap=invcmap, norm=norm)
        # contourset = ax.contourf(X, Y, Z, cmap=invcmap)
        # print(bounds)
        # raise ValueError
        f.params = copy.deepcopy(orig_params)
    if method.lower() == 'mle':
        f.mle_fit = copy.deepcopy(orig_params)
    else:
        f.chisq_res_par
    try:
        cbar = plt.colorbar(contourset, cax=cbar, orientation='vertical')
        cbar.ax.yaxis.set_ticks([0, 1/6, 0.5, 5/6])
        cbar.ax.set_yticklabels(['', r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])
    except:
        pass
    setattr(f, attr, orig_value)
    for attr, value in zip(to_save, saved):
        setattr(f, attr, copy.deepcopy(value))
    for a in axes.flatten():
        if a is not None:
            for label in a.get_xticklabels()[::2]:
                label.set_visible(False)
            for label in a.get_yticklabels()[::2]:
                label.set_visible(False)
    return fig, axes, cbar

def _diaconis_rule(data, minimum, maximum):
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    bin_size = 2 * iqr * data.shape[0]**(-1/3)
    bin_size = 0.01 * (maximum - minimum)
    if bin_size == 0 or bin_size < 0.01 * (maximum - minimum):
        return np.ceil(np.sqrt(maximum - minimum))
    else:
        return np.ceil((maximum - minimum) / bin_size)

def generate_correlation_plot(filename, filter=None, bins=None, selection=(0, 100)):
    """Given the random walk data, creates a triangle plot: distribution of
    a single parameter on the diagonal axes, 2D contour plots with 1, 2 and
    3 sigma contours on the off-diagonal. The 1-sigma limits based on the
    percentile method are also indicated, as well as added to the title.

    Parameters
    ----------
    filename: string
        Filename for the h5 file containing the data from the walk.
    filter: list of str, optional
        If supplied, only this list of columns is used for the plot.
    bins: int or list of int, optional
        If supplied, use this number of bins for the plotting.

    Returns
    -------
    figure
        Returns the MatPlotLib figure created."""
    # cmap = plt.cm.get_cmap()
    # try:
    #     invcmap = plt.cm.get_cmap(name=cmap.name + '_r')
    # except:
    #     invcmap = plt.cm.get_cmap(name=cmap.name.split('_')[0])
    with h5py.File(filename, 'r') as store:
        columns = store['data'].attrs['format']
        columns = [f.decode('utf-8') for f in columns]
        if filter is not None:
            filter = [c for f in filter for c in columns if f in c]
        else:
            filter = columns
        with tqdm.tqdm(total=len(filter)+(len(filter)**2-len(filter))/2, leave=True) as pbar:
            fig, axes, cbar = _make_axes_grid(len(filter), axis_padding=0)

            metadata = {}
            if not isinstance(bins, list):
                bins = [bins for _ in filter]
            dataset_length = store['data'].shape[0]
            first, last = int(np.floor(dataset_length/100*selection[0])), int(np.ceil(dataset_length/100*selection[1]))
            for i, val in enumerate(filter):
                pbar.set_description(val)
                ax = axes[i, i]
                bin_index = i
                i = columns.index(val)
                x = store['data'][first:last, i]
                if bins[bin_index] is None:
                    # When the diaconis rule is properly implemented, this will be used (uses too many bins)
                    bins[bin_index] = _diaconis_rule(x, x.min(), x.max())
                    # bins[bin_index] = 50
                try:
                    ax.hist(x, int(bins[bin_index]), histtype='step')
                except ValueError:
                    bins = 50
                    ax.hist(x, bins, histtype='step')
                metadata[val] = {'bins': bins[bin_index], 'min': x.min(), 'max': x.max()}

                q = [16.0, 50.0, 84.0]
                q16, q50, q84 = np.percentile(x, q)

                title = title = '{}\n${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$'
                ax.set_title(title.format(val, q50, q50-q16, q84-q50))
                qvalues = [q16, q50, q84]
                c = next(ax._get_lines.prop_cycler)['color']
                for q in qvalues:
                    ax.axvline(q, ls="dashed", color=c)
                ax.set_yticks([])
                ax.set_yticklabels([])
                pbar.update(1)

            for i, j in zip(*np.tril_indices_from(axes, -1)):
                x_name = filter[j]
                y_name = filter[i]
                pbar.set_description(x_name + ' ' + y_name)
                ax = axes[i, j]
                if j == 0:
                    ax.set_ylabel(filter[i])
                if i == len(filter) - 1:
                    ax.set_xlabel(filter[j])
                j = columns.index(x_name)
                i = columns.index(y_name)
                x = store['data'][first:last, j]
                y = store['data'][first:last, i]
                x_min, x_max, x_bins = metadata[x_name]['min'], metadata[x_name]['max'], metadata[x_name]['bins']
                y_min, y_max, y_bins = metadata[y_name]['min'], metadata[y_name]['max'], metadata[y_name]['bins']
                X = np.linspace(x_min, x_max, x_bins + 1)
                Y = np.linspace(y_min, y_max, y_bins + 1)
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

                bounds = np.unique(np.concatenate([[H.max()], V])[::-1])
                norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)

                contourset = ax.contourf(X1, Y1, H.T, bounds, cmap=invcmap, norm=norm)
                pbar.update(1)
            cbar = plt.colorbar(contourset, cax=cbar, orientation='vertical')
            cbar.ax.yaxis.set_ticks([0, 1/6, 0.5, 5/6])
            cbar.ax.set_yticklabels(['', r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])
    return fig, axes, cbar

def generate_spectrum(spectrum, x, number_of_counts, nwalkers=100):
    """Generates a model by random sampling from the provided :class:`.HFSModel`
    and range. The total number of counts for the generated spectrum
    is required.

    Parameters
    ----------
    spectrum: :class:`.HFSModel`
        An instance of class:`.HFSModel`, which gives the probability distribution
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

def load_model(path):
    """Loads the saved BaseModel and returns the reconstructed object.

    Parameters
    ----------
    path: string
        Location of the saved model.

    Returns
    -------
    model: BaseModel
        Saved BaseModel/child class instance."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

def beta(mass, V):
    r"""Calculates the beta-factor for a mass in amu
    and applied voltage in Volt. The formula used is

    .. math::

        \beta = \sqrt{1-\frac{m^2c^4}{\left(mc^2+eV\right)^2}}

    Parameters
    ----------
    mass : float
        Mass in amu.
    V : float
        voltage in volt.

    Returns
    -------
    float
        Relativistic beta-factor.
    """
    c = 299792458.0
    q = 1.60217657 * (10 ** (-19))
    AMU2KG = 1.66053892 * 10 ** (-27)
    mass = mass * AMU2KG
    top = mass ** 2 * c ** 4
    bottom = (mass * c ** 2 + q * V) ** 2
    beta = np.sqrt(1 - top / bottom)
    return beta

def dopplerfactor(mass, V):
    r"""Calculates the Doppler shift of the laser frequency for a
    given mass in amu and voltage in V. Transforms from the lab frame
    to the particle frame. The formula used is

    .. math::

        doppler = \sqrt{\frac{1-\beta}{1+\beta}}

    To invert, divide instead of multiply with
    this factor.

    Parameters
    ----------
    mass : float
        Mass in amu.
    V : float
        Voltage in volt.

    Returns
    -------
    float
        Doppler factor.
    """
    betaFactor = beta(mass, V)
    dopplerFactor = np.sqrt((1.0 - betaFactor) / (1.0 + betaFactor))
    return dopplerFactor
