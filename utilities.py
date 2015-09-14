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
    if filter is None:
        g = WalkingGrid(data, diag_sharey=False,
                        despine=False)
        returnfig = g.fig
    else:
        filter = [c for f in filter for c in data.columns.tolist() if f in c]
        data = data[filter]
        g = WalkingGrid(data, diag_sharey=False,
                        despine=False)
        returnfig = g.fig
    return returnfig

def contour2d(x, y, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    bins = kwargs.pop("bins", 50)

    try:
        x = x.values
    except:
        x = x.astype(np.float64)
    try:
        y = y.values
    except:
        y = y.astype(np.float64)
    X = np.linspace(x.min(), x.max(), bins + 1)
    Y = np.linspace(y.min(), y.max(), bins + 1)
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),
                             weights=kwargs.get('weights', None))
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

    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_rotation(45)
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_rotation(45)

    return ax, contourset

def removeAxis(x, y, ax=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.set_visible(False)
    ax.set_frame_on(False)
    ax.set_axis_off()
    return ax

def addTitle(x, ax=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    q = [16.0, 50.0, 84.0]
    q16, q50, q84 = np.percentile(x.values, q)

    title = x.name + r' = ${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$'
    ax.set_title(title.format(q50, q50-q16, q84-q50))
    qvalues = [q16, q50, q84]
    for q in qvalues:
        ax.axvline(q, ls="dashed")
    return ax

def addTruths(x, truth=None, ax=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    if truth is None:
        raise ValueError('truth should be a value')
    else:
        ax.axvline(truth)

class WalkingGrid(sns.PairGrid):

    def __init__(self, *args, **kwargs):
        super(WalkingGrid, self).__init__(*args, **kwargs)

        size = kwargs.pop("size", 4)
        aspect = kwargs.pop("aspect", 1)
        despine = kwargs.pop("despine", True)
        # Create the figure and the array of subplots
        figsize = len(self.x_vars) * size * aspect, len(self.y_vars) * size

        plt.close(self.fig)

        fig, axes = plt.subplots(len(self.y_vars), len(self.x_vars),
                                 figsize=figsize,
                                 squeeze=False)

        self.fig = fig
        self.axes = axes

        l, b, r, t = (0.25 * size * aspect / figsize[0],
                      0.4 * size / figsize[1],
                      1 - 0.1 * size * aspect / figsize[0],
                      1 - 0.2 * size * aspect / figsize[1])

        fig.subplots_adjust(left=l, bottom=b, right=r, top=t,
                            wspace=0.02, hspace=0.02)
        for ax in np.diag(self.axes):
            ax.set_yticks([])
        for ax in self.axes[:-1, :].flatten():
            ax.set_xticks([])
        for ax in self.axes[:, 1:].flatten():
            ax.set_yticks([])
        for ax in self.axes.flatten():
            labels = ax.get_xticklabels()
            for label in labels:
                label.set_rotation(45)
            labels = ax.get_yticklabels()
            for label in labels:
                label.set_rotation(45)
        y, x = np.triu_indices_from(self.axes, k=1)
        for i, j in zip(y, x):
            self.axes[i, j].set_visible(False)
            self.axes[i, j].set_frame_on(False)
            self.axes[i, j].set_axis_off()

        # Make the plot look nice
        if despine:
            sns.despine(fig=fig)
        self.map_diag(sns.distplot, kde=False)
        self.map_diag(addTitle)
        self.map_lower_with_colorbar(contour2d)

    def map_diag(self, func, **kwargs):
        """Plot with a univariate function on each diagonal subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take an x array as a positional arguments and draw onto the
            "currently active" matplotlib Axes. There is a special case when
            using a ``hue`` variable and ``plt.hist``; the histogram will be
            plotted with stacked bars.

        """
        # Add special diagonal axes for the univariate plot
        if self.square_grid and self.diag_axes is None:
            # diag_axes = []
            # for ax in np.diag(self.axes):
            #     diag_axes.append(ax)
            # self.diag_axes = np.array(diag_axes, np.object)
            self.diag_axes = np.diag(self.axes)
        else:
            pass
            # self.diag_axes = None

        # Plot on each of the diagonal axes
        for i, var in enumerate(self.x_vars):
            ax = self.diag_axes[i]
            hue_grouped = self.data[var].groupby(self.hue_vals)

            # Special-case plt.hist with stacked bars
            if func is plt.hist:
                plt.sca(ax)
                vals = [v.values for g, v in hue_grouped]
                func(vals, color=self.palette, histtype="barstacked",
                     **kwargs)
            else:
                for k, (label_k, data_k) in enumerate(hue_grouped):
                    plt.sca(ax)
                    func(data_k, label=label_k,
                         color=self.palette[k], **kwargs)

            self._clean_axis(ax)

        self._add_axis_labels()

    def map_lower_with_colorbar(self, func, **kwargs):
        """Plot with a bivariate function on the lower diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.

        """
        kw_color = kwargs.pop("color", None)
        for i, j in zip(*np.tril_indices_from(self.axes, -1)):
            hue_grouped = self.data.groupby(self.hue_vals)
            for k, (label_k, data_k) in enumerate(hue_grouped):

                ax = self.axes[i, j]
                plt.sca(ax)

                x_var = self.x_vars[j]
                y_var = self.y_vars[i]

                # Insert the other hue aesthetics if appropriate
                for kw, val_list in self.hue_kws.items():
                    kwargs[kw] = val_list[k]

                color = self.palette[k] if kw_color is None else kw_color
                ax, c = func(data_k[x_var], data_k[y_var], label=label_k,
                             color=color, **kwargs)

            self._clean_axis(ax)
            self._update_legend_data(ax)

        if kw_color is not None:
            kwargs["color"] = kw_color
        self._add_axis_labels()
        cax, kw = mpl.colorbar.make_axes([a for a in self.axes.flat])
        cbar = plt.colorbar(c, cax=cax)
        cbar.ax.set_yticklabels(['', r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])

    def map_upper(self, func, **kwargs):
        """Plot with a bivariate function on the upper diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.

        """
        kw_color = kwargs.pop("color", None)
        for i, j in zip(*np.triu_indices_from(self.axes, 1)):

            hue_grouped = self.data.groupby(self.hue_vals)
            for k, (label_k, data_k) in enumerate(hue_grouped):

                ax = self.axes[i, j]
                plt.sca(ax)

                x_var = self.x_vars[j]
                y_var = self.y_vars[i]

                # Insert the other hue aesthetics if appropriate
                for kw, val_list in self.hue_kws.items():
                    kwargs[kw] = val_list[k]

                color = self.palette[k] if kw_color is None else kw_color
                func(data_k[x_var], data_k[y_var], label=label_k,
                     color=color, **kwargs)

            self._clean_axis(ax)
            self._update_legend_data(ax)

        if kw_color is not None:
            kwargs["color"] = kw_color

    def map_offdiag(self, func, **kwargs):
        """Plot with a bivariate function on the off-diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.

        """

        self.map_lower(func, **kwargs)
        self.map_upper(func, **kwargs)

    def _add_axis_labels(self):
        """Add labels to the left and bottom Axes."""
        for ax, label in zip(self.axes[-1, :], self.x_vars):
            ax.set_xlabel(label)
        for ax, label in zip(self.axes[:, 0], self.y_vars):
            ax.set_ylabel(label)

    def _find_numeric_cols(self, data):
        """Find which variables in a DataFrame are numeric."""
        # This can't be the best way to do this, but  I do not
        # know what the best way might be, so this seems ok
        numeric_cols = []
        for col in data:
            try:
                data[col].astype(np.float)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                pass
        return numeric_cols

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
