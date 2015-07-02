"""
.. module:: utilities
    :platform: Windows
    :synopsis: Implementation of various functions that ease the work,
     but do not belong in one of the other modules.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""
import emcee as mcmc
import lmfit as lm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

c = 299792458.0
h = 6.62606957 * (10 ** -34)
q = 1.60217657 * (10 ** -19)

cmap = mpl.colors.ListedColormap(['#A6CEE3', '#1F78B4', '#B2DF8A'])
invcmap = mpl.colors.ListedColormap(['#B2DF8A', '#1F78B4', '#A6CEE3'])


def state_number_enumerate(dims, state=None, idx=0):
    """Create the indices for the different entries in
    a multi-dimensional array. Code copied from the QuTiP package.

    Parameters
    ----------
    shape: tuple
        Describes the shape of the multi-dimensional array.

    Returns
    -------
    tuple
        Tuple with each entry being a tuple containing the indices."""
    if state is None:
        state = np.zeros(len(dims))

    if idx == len(dims):
        yield tuple(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, state, idx + 1):
                yield s


# Create a vectorized function for calling an array of callables,
# mixed with non-callables.
def ifCallableCall(ob, arg):
    return ob(arg) if callable(ob) else ob

vFifCallableCall = np.vectorize(ifCallableCall, otypes=[np.float])


def callNDArray(arr, arg):
    """Goes over each subarray in the first dimension,
    and calls the corresponding argument. Returns the values itself
    if the entry is not callable.

    Parameters
    ----------
    arr: NumPy array
        Array containing a mix of callable and not-callable entries.
    arg: misc
        Argument to be passed to each callable entry.

    Returns
    -------
    NumPy array
        Array with the callable entries replaced by the returned value."""
    n = arr.shape[0]
    assert n == len(arg)
    res = np.zeros(arr.shape)
    for i in range(n):
        # Go for the vectorized function. In case of problems,
        # comment the following line and use the try-except
        # block. That is proven to work, but is slower.
        res[i] = vFifCallableCall(arr[i], arg[i])
        # try:
        #     res[i] = np.array(
        #         [[x(arg[i]) if callable(x) else x for x in xarr]
        #         for xarr in arr[i]])
        # except ValueError:
        #     raise ValueError()
    return res


class ReleaseCurve(object):

    r"""Creates a callable object for the standard release curve. Formula
    based on J.P. Ramos et al. :cite:`Ramos2014`. Input parameters are
    initialized to an 35Ar release curve.

    Parameters
    ----------
    amp : float,
        Influences the height of the curve, roughly the maximum of the
        release rate. Is also an attribute. Default: 4E7
    a : float between 0 and 1
        Weighting of the different exponentials in the formula. Is also an
        attribute. Default: 0.9
    tr : float
        Time constant parameter in seconds. The attribute is saved as the
        corresponding l-parameter. Default: 78 ms
    tf : float
        Time constant parameter in seconds. The attribute is saved as the
        corresponding l-parameter. Default: 396 ms
    ts : float
        Time constant parameter in seconds. The attribute is saved as the
        corresponding l-parameter. Default: 1905 ms
    pulses : integer
        Number of pulses seperated by the delay parameter. Has no effect if the
        :attr:`continued` parameter is True. Is also an attribute. Default: 3
    delay : float
        Seconds between pulses. Is also an attribute. Default: 10.0 s
    continued : bool
        Continuously generate pulses seperated by the delay parameter if True,
        else create the number of pulses given in the pulses parameter. Is also
        an attribute. Default: True

    Note
    ----
    The l-parameters are related to the t-parameters through
    :math:`l = \frac{\ln(2)}{t}`. The release curve is modeled as:

    .. math::
        RC\left(t\right) = a\left(1-\exp\left(-l_rt\right)\right)
        \left(a\exp\left(-l_ft\right)+(1-a)\exp\left(-l_st\right)\right)"""

    def __init__(self, amp=4.0 * 10 ** 7, a=0.9,
                 tr=78 * (10 ** -3), tf=396 * (10 ** -3),
                 ts=1905 * (10 ** -3),
                 pulses=3, delay=10.0, continued=True):
        super(ReleaseCurve, self).__init__()
        self.amp = amp
        self.a = a

        self.lr = np.log(2) / tr
        self.lf = np.log(2) / tf
        self.ls = np.log(2) / ts

        self.pulses = pulses
        self.delay = delay
        self.continued = continued

    def fit_to_data(self, t, y, yerr):
        """If a release curve is measured as a function of time, this should
        fit the parameters to the given curve y(t) with errors yerr.

        Parameters
        ----------
        t: array_like
            Timevector of the measurements.
        y: array_like
            Counts corresponding to t.
        yerr: array_like
            Counting errors of y.

        Warning
        -------
        This method has not been tested!"""
        import lmfit as lm
        params = lm.Parameters()
        params.add_many(
            ('Amp', self.amp, True, 0, None, None),
            ('a', self.a, True, 0, 1, None, None),
            ('tr', np.log(2) / self.lr, True, None, None, None),
            ('tf', np.log(2) / self.lf, True, None, None, None),
            ('ts', np.log(2) / self.ls, True, None, None, None))

        def resid(params):
            self.amp = params['Amp']
            self.a = params['a']
            self.lr = np.log(2) / params['tr']
            self.lf = np.log(2) / params['tf']
            self.ls = np.log(2) / params['ts']
            return (y - self.empirical_formula(t)) / yerr

        return lm.minimize(resid, params)

    @property
    def pulses(self):
        return self._pulses

    @pulses.setter
    def pulses(self, value):
        self._pulses = int(value)

    @property
    def continued(self):
        return self._continued

    @continued.setter
    def continued(self, value):
        self._continued = (value == 1)

    def empirical_formula(self, t):
        amp = self.amp
        a = self.a
        lr = self.lr
        lf = self.lf
        ls = self.ls

        val = amp * (1 - np.exp(-lr * t)) * (a * np.exp(-lf * t) +
                                             (1 - a) * np.exp(-ls * t))
        return val

    def __call__(self, t):
        """Return the evaluation of the formula, taking the pulses
        and delays into account.

        Parameters
        ----------
        t: array_like
            Times for which the yield is requested."""
        pulses = self.pulses
        delay = self.delay
        continued = self.continued

        pulses = np.arange(1.0, pulses) * delay
        rc = self.empirical_formula(t)
        if not continued:
            for pulsetime in pulses:
                mask = t > pulsetime
                try:
                    if any(mask):
                        rc[mask] += self.empirical_formula(t[mask] - pulsetime)
                except TypeError:
                    if mask:
                        rc += self.empirical_formula(t - pulsetime)
        else:
            pulsetime = delay
            try:
                number = (t // pulsetime).astype('int')
                for pulses in range(1, max(number) + 1):
                    mask = (number >= pulses)
                    rc[mask] += self.empirical_formula(t[mask] -
                                                       pulses * pulsetime)
            except AttributeError:
                number = int(t // pulsetime)
                if number > 0:
                    for i in range(number):
                        rc += self.empirical_formula(t - (i + 1) * pulsetime)
        return rc


class Level(object):

    """Ease-of-use class for representing a level.

    Parameters
    ----------
    energy : float
        Fine structure energy in eV.
    hyp_par : list of 2 floats
        Hyperfine parameters [A, B] in MHz.
    L, S, J : integer or half-integers
        Spin quantum numbers."""

    def __init__(self, energy, hyp_par, L, S, J):
        super(Level, self).__init__()
        self.energy = energy
        self.A, self.B = hyp_par
        self.L = L
        self.S = S
        self.J = J

    def __str__(self):
        s = '<Level object: E=%f, A=%f, B=%f, L=%f, S=%f, J=%f>' % (
            self.energy, self.A, self.B, self.L, self.S, self.J)
        return s


def invCM2MHz(invCM):
    return invCM * 100.0 * c * 10 ** -6


def MHz2invCM(MHz):
    return MHz * 10 ** 6 / (100.0 * c)


def invCM2eV(invCM):
    return invCM * 100.0 * h * c / q


def eV2invCM(eV):
    return eV * q / (100.0 * h * c)


def invCM2nm(invCM):
    return ((invCM * 100.0) ** -1) * (10 ** 9)


def nm2invCM(nm):
    return ((nm * (10 ** -9)) ** -1) / 100.0


class Energy(object):

    """Ease-of-use class to represent energy and frequencies.
    Uses automatic conversion to a series of units.

    Parameters
    ----------
    value: float
        Value of the energy or frequency to be converted/worked with.
    unit: string, {cm-1, MHz, eV, nm}
        String denoting the unit for the given value. Default value is inverse
        centimeters (cm-1)."""

    __units__ = ['cm-1', 'MHz', 'eV', 'nm']
    __conversion__ = {'MHz': invCM2MHz,
                      'eV': invCM2eV,
                      'nm': invCM2nm}

    def __init__(self, value, unit='cm-1'):
        super(Energy, self).__init__()
        if unit not in self.__units__:
            m = '{} is an unknown unit!'.format(unit)
            raise TypeError(m)
        self.unit = unit
        self.value = value
        convert = {'MHz': MHz2invCM,
                   'eV': eV2invCM,
                   'nm': nm2invCM}
        if self.unit in convert.keys():
            self.value = convert[self.unit](self.value)
            self.unit = 'cm-1'

    def __call__(self, unit):
        """Convert the value to the given unit.

        Parameters
        ----------
        unit: string
            Requested unit, must be 'cm-1', 'MHz', 'eV' or 'nm'.

        Returns
        -------
        float
            Converted value."""
        if unit in self.__conversion__.keys():
            val = self.__conversion__[unit](self.value)
        else:
            val = self.value
        return val


def round2SignifFigs(vals, n):
    """
    Code copied from
    http://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    Goes over the list or array of vals given, and rounds
    them to the number of significant digits (n) given.

    Parameters
    ----------
    vals : array_like
        Values to be rounded.
    n : integer
        Number of significant digits to round to.

    Note
    ----
    Does not accept: inf, nan, complex

    Example
    -------
    >>> m = [0.0, -1.2366e22, 1.2544444e-15, 0.001222]
    >>> round2SignifFigs(m,2)
    array([  0.00e+00,  -1.24e+22,   1.25e-15,   1.22e-03])
    """
    if np.all(np.isfinite(vals)) and np.all(np.isreal((vals))):
        eset = np.seterr(all='ignore')
        mags = 10.0 ** np.floor(np.log10(np.abs(vals)))  # omag's
        vals = np.around(vals / mags, n - 1) * mags  # round(val/omag)*omag
        np.seterr(**eset)
        vals[np.where(np.isnan(vals))] = 0.0  # 0.0 -> nan -> 0.0
    else:
        raise IOError('Input must be real and finite')
    return vals


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


def bootstrap_ci(dataframe, kind='basic'):
    """Generate confidence intervals on the 1-sigma level for bootstrapped data
    given in a DataFrame.

    Parameters
    ----------
    dataframe: DataFrame
        DataFrame with the results of each bootstrap fit on a row. If the
        t-method is to be used, a Panel is required, with the data in
        the panel labeled 'data' and the uncertainties labeled 'stderr'
    kind: str, optional
        Selects which method to use: percentile, basic, or t-method (student).

    Returns
    -------
    DataFrame
        Dataframe containing the left and right limits for each column as rows.
"""
    if isinstance(dataframe, pd.Panel):
        data = dataframe['data']
        stderrs = dataframe['stderr']
        args = (data, stderrs)
    else:
        data = dataframe
        args = (data)

    def percentile(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        left = data.apply(lambda col: np.percentile(col, 15.865), axis=0)
        right = data.apply(lambda col: np.percentile(col, 84.135), axis=0)
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    def basic(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        left = data.apply(lambda col: 2 * col[0] - np.percentile(col[1:],
                                                                 84.135),
                          axis=0)
        right = data.apply(lambda col: 2 * col[0] - np.percentile(col[1:],
                                                                  15.865),
                           axis=0)
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    def student(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        R = (data - data.loc[0]) / stderrs
        left = R.apply(lambda col: np.percentile(col[1:], 84.135), axis=0)
        right = R.apply(lambda col: np.percentile(col[1:], 15.865), axis=0)
        left = data.loc[0] - stderrs.loc[0] * left
        right = data.loc[0] - stderrs.loc[0] * right
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    method = {'basic': basic, 'percentile': percentile, 't': student}
    method = method.pop(kind.lower(), basic)
    return method(*args)


def generate_likelihood_plot(data):
    shape = int(np.ceil(np.sqrt(len(data.keys()))))
    figLikelih, axes = plt.subplots(shape, shape)
    axes = axes.flatten()
    for name, ax in zip(data.keys(), axes):
        x = data[name]['x']
        y = data[name]['y']
        ax.plot(x, y)
        ax.set_xlabel(name)
    return figLikelih, axes


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


class FittingGrid(sns.Grid):

    def __init__(self, minimizer, size=3, aspect=1,
                 despine=False, nx=10, ny=10, selected=None,
                 limits=5, **kwargs):
        super(FittingGrid, self).__init__()
        # Sort out the variables that define the grid
        self.nx, self.ny = nx, ny
        self.minimizer = minimizer
        vars = []
        self.extra_keywords = kwargs
        self.limit = limits
        for key in minimizer.params:
            if minimizer.params[key].vary:
                if selected is None:
                    vars.append(key)
                else:
                    for r in selected:
                        if r in key:
                            vars.append(key)

        self.vars = list(vars)
        self.vars = sorted(self.vars)
        self.x_vars = self.vars[:-1]
        self.y_vars = self.vars[1:]

        # Create the figure and the array of subplots
        figsize = (len(vars) - 1) * size * aspect, (len(vars) - 1) * size

        fig, axes = plt.subplots(len(vars) - 1, len(vars) - 1,
                                 figsize=figsize,
                                 sharex="col", sharey="row",
                                 squeeze=False)

        self.fig = fig
        self.axes = axes
        if len(vars) == 2:
            self.axes = np.array([axes]).reshape((1, 1))
        self._legend_data = {}
        self.remove_upper()

        # Label the axes
        self._add_axis_labels()

        # Make the plot look nice
        if despine:
            sns.despine(fig=self.fig)
        self.make_maps()

    def make_maps(self):
        V = [0, 0.68, 0.95, 0.99]
        bounds = V
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        for x, y in zip(*np.tril_indices(len(self.vars) - 1)):
            ax = self.axes[x, y]
            param1 = self.x_vars[y]
            param2 = self.y_vars[x]
            p1 = self.minimizer.params[param1]
            p2 = self.minimizer.params[param2]
            limits = ((p1.value + self.limit * p1.stderr,
                       p1.value - self.limit * p1.stderr),
                      (p2.value + self.limit * p2.stderr,
                       p2.value - self.limit * p2.stderr))
            X, Y, GR = lm.conf_interval2d(self.minimizer, param1, param2,
                                          nx=self.nx, ny=self.ny,
                                          limits=limits)
            # cf = ax.contourf(X, Y, GR, V, cmap=self.cm)
            cf = ax.contourf(X, Y, GR, V, cmap=cmap, norm=norm)
            labels = ax.get_xticklabels()
            for label in labels:
                label.set_rotation(45)
            labels = ax.get_yticklabels()
            for label in labels:
                label.set_rotation(45)
        cax, kw = mpl.colorbar.make_axes([a for a in self.axes.flat])
        cbar = plt.colorbar(cf, cax=cax)
        cbar.ax.set_yticklabels(['', r'1$\sigma$', r'2$\sigma$', r'3$\sigma$'])

    def remove_upper(self):
        """Plot with a bivariate function on the upper diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.

        """
        for i, j in zip(*np.triu_indices_from(self.axes, 1)):
            ax = self.axes[i, j]
            plt.sca(ax)
            ax.set_visible(False)
            ax.set_frame_on(False)
            ax.set_axis_off()

            self._clean_axis(ax)
            self._update_legend_data(ax)

    def _add_axis_labels(self):
        """Add labels to the left and bottom Axes."""
        for ax, label in zip(self.axes[-1, :], self.x_vars):
            ax.set_xlabel(label)
        for ax, label in zip(self.axes[:, 0], self.y_vars):
            ax.set_ylabel(label)


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
