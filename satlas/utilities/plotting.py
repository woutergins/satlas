import copy

from satlas import tqdm
from satlas.stats import fitting
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import uncertainties as u
from scipy import optimize
from scipy.stats import chi2

inv_color_list = ['#7acfff', '#fff466', '#00c48f', '#ff8626', '#ff9cd3', '#0093e6']
color_list = [c for c in reversed(inv_color_list)]
cmap = mpl.colors.ListedColormap(color_list)
cmap.set_over(color_list[-1])
cmap.set_under(color_list[0])
invcmap = mpl.colors.ListedColormap(inv_color_list)
invcmap.set_over(inv_color_list[-1])
invcmap.set_under(inv_color_list[0])

__all__ = ['generate_correlation_map', 'generate_correlation_plot', 'generate_walk_plot']

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
    method: {'chisquare', 'chisquare_spectroscopic', mle'}
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

    # Save the original goodness-of-fit and parameters for later use
    mapping = {'chisquare_spectroscopic': (fitting.chisquare_spectroscopic_fit, 'chisqr_chi'),
               'chisquare': (fitting.chisquare_fit, 'chisqr_chi'),
               'mle': (fitting.likelihood_fit, 'likelihood_mle')}
    func, attr = mapping.pop(method.lower(), (fitting.chisquare_spectroscopic_fit, 'chisqr_chi'))
    title = '{}\n${}_{{-{}}}^{{+{}}}$'
    title_e = '{}\n$({}_{{-{}}}^{{+{}}})e{}$'
    fit_kws['verbose'] = False
    fit_kws['hessian'] = False

    to_save = {'mle': ('fit_mle', 'result_mle')}
    to_save = to_save.pop(method.lower(), ('chisq_res_par', 'ndof_chi', 'redchi_chi'))
    saved = [copy.deepcopy(getattr(f, attr)) for attr in to_save]

    # func(f, x_data, y_data, *fit_args, **fit_kws)

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
        if i == no_params-1:
            if method.lower().startswith('chisquare'):
                ax.set_ylabel(r'$\Delta\chi^2$')
            else:
                ax.set_ylabel(r'$\Delta\mathcal{L}$')

        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        print(stderr)
        stderr = stderr if stderr is not None else 0.01 * np.abs(value)
        stderr = stderr if stderr != 0 else 0.01 * np.abs(value)
        result_left, success_left = fitting._find_boundary(-stderr, param_names[i], boundary, f, x_data, y_data, function_kwargs=function_kws)
        result_right, success_right = fitting._find_boundary(stderr, param_names[i], boundary, f, x_data, y_data, function_kwargs=function_kws)
        success = success_left * success_right
        ranges[param_names[i]]['left'] = result_left
        ranges[param_names[i]]['right'] = result_right

        if not success:
            print("Warning: boundary calculation did not fully succeed for " + param_names[i])
        right = np.abs(ranges[param_names[i]]['right'] - value)
        left = np.abs(ranges[param_names[i]]['left'] - value)
        params[param_names[i]].vary = False

        left_val, right_val = max(value - distance * left, orig_params[param_names[i]].min), min(value + distance * right, orig_params[param_names[i]].max)
        ranges[param_names[i]]['right_val'] = right_val
        ranges[param_names[i]]['left_val'] = left_val
        value_range = np.linspace(left_val, right_val, resolution_diag)
        value_range = np.sort(np.append(value_range, np.array([value - left, value + right, value])))
        chisquare = np.zeros(len(value_range))
        # Calculate the new value, and store it in the array. Update the progressbar.
        with tqdm.tqdm(value_range, desc=param_names[i], leave=True) as pbar:
            for j, v in enumerate(value_range):
                chisquare[j] = fitting.calculate_updated_statistic(v, param_names[i], f, x_data, y_data, **function_kws)
                fitting._set_state(f, state, method=method.lower())
                pbar.update(1)
        # Plot the result
        ax.plot(value_range, chisquare, color='k')

        c = '#0093e6'
        # Indicate the used interval.
        ax.axvline(value + right, ls="dashed", color=c)
        ax.axvline(value - left, ls="dashed", color=c)
        ax.axvline(value, ls="dashed", color=c)
        ax.axhline(boundary, color=c)
        up = '{:.2ug}'.format(u.ufloat(value, right))
        down = '{:.2ug}'.format(u.ufloat(value, left))
        val = up.split('+')[0].split('(')[-1]
        r = up.split('-')[1].split(')')[0]
        l = down.split('-')[1].split(')')[0]
        if 'e' in up or 'e' in down:
            ex = up.split('e')[-1]
            ax.set_title(title_e.format(param_names[i], val, l, r, ex))
        else:
            ax.set_title(title.format(param_names[i], val, l, r))
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
        x_range = np.append(np.linspace(left, right, resolution_map), orig_params[x_name].value)
        x_range = np.sort(x_range)

        right = ranges[y_name]['right_val']
        left = ranges[y_name]['left_val']
        y_range = np.append(np.linspace(left, right, resolution_map), orig_params[y_name].value)
        y_range = np.sort(y_range)

        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros(X.shape)
        i_indices, j_indices = np.indices(Z.shape)
        with tqdm.tqdm(i_indices.flatten(), desc=param_names[j]+' ' + param_names[i], leave=True) as pbar:
            for k, l in zip(i_indices.flatten(), j_indices.flatten()):
                x = X[k, l]
                y = Y[k, l]
                print(x, y, f.params['Background0'].value)
                Z[k, l] = fitting.calculate_updated_statistic([x, y], [x_name, y_name], f, x_data, y_data, **function_kws)
                fitting._set_state(f, state, method=method.lower())

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
        f.params = copy.deepcopy(orig_params)
    if method.lower() == 'mle':
        f.fit_mle = copy.deepcopy(orig_params)
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
                    width = 3.5*np.std(x)/x.size**(1/3) #Scott's rule for binwidth
                    bins[bin_index] = np.arange(x.min(), x.max()+width, width)
                try:
                    n, b, p, = ax.hist(x, int(bins[bin_index]), histtype='step', color='k')
                except TypeError:
                    bins[bin_index] = 50
                    n, b, p, = ax.hist(x, int(bins[bin_index]), histtype='step', color='k')
                center = n.argmax()
                # q50 = (b[center] + b[center+1])/2
                # q16 = np.percentile(x[x<q50], 84)
                # q84 = np.percentile(x[x>q50], 16)
                metadata[val] = {'bins': bins[bin_index], 'min': x.min(), 'max': x.max()}

                q = [50.0, 16.0, 84.0]
                q50, q16, q84 = np.percentile(x, q)

                title = '{}\n${}_{{-{}}}^{{+{}}}$'
                title_e = '{}\n$({}_{{-{}}}^{{+{}}})e{}$'
                up = '{:.2ug}'.format(u.ufloat(q50, np.abs(q84-q50)))
                down = '{:.2ug}'.format(u.ufloat(q50, np.abs(q50-q16)))
                param_val = up.split('+')[0].split('(')[-1]
                r = up.split('+/-')[1].split(')')[0]
                l = down.split('+/-')[1].split(')')[0]
                if 'e' in up or 'e' in down:
                    ex = up.split('e')[-1]
                    ax.set_title(title_e.format(val, param_val, l, r, ex))
                else:
                    ax.set_title(title.format(val, param_val, l, r))

                qvalues = [q16, q50, q84]
                c = '#0093e6'
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
            try:
                cbar = plt.colorbar(contourset, cax=cbar, orientation='vertical')
                cbar.ax.yaxis.set_ticks([0, 1/6, 0.5, 5/6])
                cbar.ax.set_yticklabels(['', r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])
            except:
                cbar = None
    return fig, axes, cbar

def generate_walk_plot(filename, filter=None, selection=(0, 100), walkers=20):
    """Given the random walk data, the random walk for the selected parameters
    is plotted.

    Parameters
    ----------
    filename: string
        Filename for the h5 file containing the data from the walk.
    filter: list of str, optional
        If supplied, only this list of parameters is used for the plot.

    Returns
    -------
    figure
        Returns the MatPlotLib figure created."""

    with h5py.File(filename, 'r') as store:
        columns = store['data'].attrs['format']
        columns = [f.decode('utf-8') for f in columns]
        if filter is not None:
            filter = [c for f in filter for c in columns if f in c]
        else:
            filter = columns
        with tqdm.tqdm(total=len(filter)+(len(filter)**2-len(filter))/2, leave=True) as pbar:
            fig, axes = plt.subplots(len(filter), 1, sharex=True)

            dataset_length = store['data'].shape[0]
            first, last = int(np.floor(dataset_length/100*selection[0])), int(np.ceil(dataset_length/100*selection[1]))
            for i, (val, ax) in enumerate(zip(filter, axes)):
                pbar.set_description(val)
                i = columns.index(val)
                x = store['data'][first:last, i]
                new_x = np.reshape(x, (-1, walkers))
                q50 = np.percentile(x, [50.0])
                ax.plot(range(new_x.shape[0]), new_x, alpha=0.3, color='gray')
                ax.set_ylabel(val)
                ax.axhline(q50, color='k')
                pbar.update(1)
            ax.set_xlabel('Step')
        pbar.close()
    return fig, axes
