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

cm_data = [[ 0.26700401,  0.00487433,  0.32941519],
       [ 0.26851048,  0.00960483,  0.33542652],
       [ 0.26994384,  0.01462494,  0.34137895],
       [ 0.27130489,  0.01994186,  0.34726862],
       [ 0.27259384,  0.02556309,  0.35309303],
       [ 0.27380934,  0.03149748,  0.35885256],
       [ 0.27495242,  0.03775181,  0.36454323],
       [ 0.27602238,  0.04416723,  0.37016418],
       [ 0.2770184 ,  0.05034437,  0.37571452],
       [ 0.27794143,  0.05632444,  0.38119074],
       [ 0.27879067,  0.06214536,  0.38659204],
       [ 0.2795655 ,  0.06783587,  0.39191723],
       [ 0.28026658,  0.07341724,  0.39716349],
       [ 0.28089358,  0.07890703,  0.40232944],
       [ 0.28144581,  0.0843197 ,  0.40741404],
       [ 0.28192358,  0.08966622,  0.41241521],
       [ 0.28232739,  0.09495545,  0.41733086],
       [ 0.28265633,  0.10019576,  0.42216032],
       [ 0.28291049,  0.10539345,  0.42690202],
       [ 0.28309095,  0.11055307,  0.43155375],
       [ 0.28319704,  0.11567966,  0.43611482],
       [ 0.28322882,  0.12077701,  0.44058404],
       [ 0.28318684,  0.12584799,  0.44496   ],
       [ 0.283072  ,  0.13089477,  0.44924127],
       [ 0.28288389,  0.13592005,  0.45342734],
       [ 0.28262297,  0.14092556,  0.45751726],
       [ 0.28229037,  0.14591233,  0.46150995],
       [ 0.28188676,  0.15088147,  0.46540474],
       [ 0.28141228,  0.15583425,  0.46920128],
       [ 0.28086773,  0.16077132,  0.47289909],
       [ 0.28025468,  0.16569272,  0.47649762],
       [ 0.27957399,  0.17059884,  0.47999675],
       [ 0.27882618,  0.1754902 ,  0.48339654],
       [ 0.27801236,  0.18036684,  0.48669702],
       [ 0.27713437,  0.18522836,  0.48989831],
       [ 0.27619376,  0.19007447,  0.49300074],
       [ 0.27519116,  0.1949054 ,  0.49600488],
       [ 0.27412802,  0.19972086,  0.49891131],
       [ 0.27300596,  0.20452049,  0.50172076],
       [ 0.27182812,  0.20930306,  0.50443413],
       [ 0.27059473,  0.21406899,  0.50705243],
       [ 0.26930756,  0.21881782,  0.50957678],
       [ 0.26796846,  0.22354911,  0.5120084 ],
       [ 0.26657984,  0.2282621 ,  0.5143487 ],
       [ 0.2651445 ,  0.23295593,  0.5165993 ],
       [ 0.2636632 ,  0.23763078,  0.51876163],
       [ 0.26213801,  0.24228619,  0.52083736],
       [ 0.26057103,  0.2469217 ,  0.52282822],
       [ 0.25896451,  0.25153685,  0.52473609],
       [ 0.25732244,  0.2561304 ,  0.52656332],
       [ 0.25564519,  0.26070284,  0.52831152],
       [ 0.25393498,  0.26525384,  0.52998273],
       [ 0.25219404,  0.26978306,  0.53157905],
       [ 0.25042462,  0.27429024,  0.53310261],
       [ 0.24862899,  0.27877509,  0.53455561],
       [ 0.2468114 ,  0.28323662,  0.53594093],
       [ 0.24497208,  0.28767547,  0.53726018],
       [ 0.24311324,  0.29209154,  0.53851561],
       [ 0.24123708,  0.29648471,  0.53970946],
       [ 0.23934575,  0.30085494,  0.54084398],
       [ 0.23744138,  0.30520222,  0.5419214 ],
       [ 0.23552606,  0.30952657,  0.54294396],
       [ 0.23360277,  0.31382773,  0.54391424],
       [ 0.2316735 ,  0.3181058 ,  0.54483444],
       [ 0.22973926,  0.32236127,  0.54570633],
       [ 0.22780192,  0.32659432,  0.546532  ],
       [ 0.2258633 ,  0.33080515,  0.54731353],
       [ 0.22392515,  0.334994  ,  0.54805291],
       [ 0.22198915,  0.33916114,  0.54875211],
       [ 0.22005691,  0.34330688,  0.54941304],
       [ 0.21812995,  0.34743154,  0.55003755],
       [ 0.21620971,  0.35153548,  0.55062743],
       [ 0.21429757,  0.35561907,  0.5511844 ],
       [ 0.21239477,  0.35968273,  0.55171011],
       [ 0.2105031 ,  0.36372671,  0.55220646],
       [ 0.20862342,  0.36775151,  0.55267486],
       [ 0.20675628,  0.37175775,  0.55311653],
       [ 0.20490257,  0.37574589,  0.55353282],
       [ 0.20306309,  0.37971644,  0.55392505],
       [ 0.20123854,  0.38366989,  0.55429441],
       [ 0.1994295 ,  0.38760678,  0.55464205],
       [ 0.1976365 ,  0.39152762,  0.55496905],
       [ 0.19585993,  0.39543297,  0.55527637],
       [ 0.19410009,  0.39932336,  0.55556494],
       [ 0.19235719,  0.40319934,  0.55583559],
       [ 0.19063135,  0.40706148,  0.55608907],
       [ 0.18892259,  0.41091033,  0.55632606],
       [ 0.18723083,  0.41474645,  0.55654717],
       [ 0.18555593,  0.4185704 ,  0.55675292],
       [ 0.18389763,  0.42238275,  0.55694377],
       [ 0.18225561,  0.42618405,  0.5571201 ],
       [ 0.18062949,  0.42997486,  0.55728221],
       [ 0.17901879,  0.43375572,  0.55743035],
       [ 0.17742298,  0.4375272 ,  0.55756466],
       [ 0.17584148,  0.44128981,  0.55768526],
       [ 0.17427363,  0.4450441 ,  0.55779216],
       [ 0.17271876,  0.4487906 ,  0.55788532],
       [ 0.17117615,  0.4525298 ,  0.55796464],
       [ 0.16964573,  0.45626209,  0.55803034],
       [ 0.16812641,  0.45998802,  0.55808199],
       [ 0.1666171 ,  0.46370813,  0.55811913],
       [ 0.16511703,  0.4674229 ,  0.55814141],
       [ 0.16362543,  0.47113278,  0.55814842],
       [ 0.16214155,  0.47483821,  0.55813967],
       [ 0.16066467,  0.47853961,  0.55811466],
       [ 0.15919413,  0.4822374 ,  0.5580728 ],
       [ 0.15772933,  0.48593197,  0.55801347],
       [ 0.15626973,  0.4896237 ,  0.557936  ],
       [ 0.15481488,  0.49331293,  0.55783967],
       [ 0.15336445,  0.49700003,  0.55772371],
       [ 0.1519182 ,  0.50068529,  0.55758733],
       [ 0.15047605,  0.50436904,  0.55742968],
       [ 0.14903918,  0.50805136,  0.5572505 ],
       [ 0.14760731,  0.51173263,  0.55704861],
       [ 0.14618026,  0.51541316,  0.55682271],
       [ 0.14475863,  0.51909319,  0.55657181],
       [ 0.14334327,  0.52277292,  0.55629491],
       [ 0.14193527,  0.52645254,  0.55599097],
       [ 0.14053599,  0.53013219,  0.55565893],
       [ 0.13914708,  0.53381201,  0.55529773],
       [ 0.13777048,  0.53749213,  0.55490625],
       [ 0.1364085 ,  0.54117264,  0.55448339],
       [ 0.13506561,  0.54485335,  0.55402906],
       [ 0.13374299,  0.54853458,  0.55354108],
       [ 0.13244401,  0.55221637,  0.55301828],
       [ 0.13117249,  0.55589872,  0.55245948],
       [ 0.1299327 ,  0.55958162,  0.55186354],
       [ 0.12872938,  0.56326503,  0.55122927],
       [ 0.12756771,  0.56694891,  0.55055551],
       [ 0.12645338,  0.57063316,  0.5498411 ],
       [ 0.12539383,  0.57431754,  0.54908564],
       [ 0.12439474,  0.57800205,  0.5482874 ],
       [ 0.12346281,  0.58168661,  0.54744498],
       [ 0.12260562,  0.58537105,  0.54655722],
       [ 0.12183122,  0.58905521,  0.54562298],
       [ 0.12114807,  0.59273889,  0.54464114],
       [ 0.12056501,  0.59642187,  0.54361058],
       [ 0.12009154,  0.60010387,  0.54253043],
       [ 0.11973756,  0.60378459,  0.54139999],
       [ 0.11951163,  0.60746388,  0.54021751],
       [ 0.11942341,  0.61114146,  0.53898192],
       [ 0.11948255,  0.61481702,  0.53769219],
       [ 0.11969858,  0.61849025,  0.53634733],
       [ 0.12008079,  0.62216081,  0.53494633],
       [ 0.12063824,  0.62582833,  0.53348834],
       [ 0.12137972,  0.62949242,  0.53197275],
       [ 0.12231244,  0.63315277,  0.53039808],
       [ 0.12344358,  0.63680899,  0.52876343],
       [ 0.12477953,  0.64046069,  0.52706792],
       [ 0.12632581,  0.64410744,  0.52531069],
       [ 0.12808703,  0.64774881,  0.52349092],
       [ 0.13006688,  0.65138436,  0.52160791],
       [ 0.13226797,  0.65501363,  0.51966086],
       [ 0.13469183,  0.65863619,  0.5176488 ],
       [ 0.13733921,  0.66225157,  0.51557101],
       [ 0.14020991,  0.66585927,  0.5134268 ],
       [ 0.14330291,  0.66945881,  0.51121549],
       [ 0.1466164 ,  0.67304968,  0.50893644],
       [ 0.15014782,  0.67663139,  0.5065889 ],
       [ 0.15389405,  0.68020343,  0.50417217],
       [ 0.15785146,  0.68376525,  0.50168574],
       [ 0.16201598,  0.68731632,  0.49912906],
       [ 0.1663832 ,  0.69085611,  0.49650163],
       [ 0.1709484 ,  0.69438405,  0.49380294],
       [ 0.17570671,  0.6978996 ,  0.49103252],
       [ 0.18065314,  0.70140222,  0.48818938],
       [ 0.18578266,  0.70489133,  0.48527326],
       [ 0.19109018,  0.70836635,  0.48228395],
       [ 0.19657063,  0.71182668,  0.47922108],
       [ 0.20221902,  0.71527175,  0.47608431],
       [ 0.20803045,  0.71870095,  0.4728733 ],
       [ 0.21400015,  0.72211371,  0.46958774],
       [ 0.22012381,  0.72550945,  0.46622638],
       [ 0.2263969 ,  0.72888753,  0.46278934],
       [ 0.23281498,  0.73224735,  0.45927675],
       [ 0.2393739 ,  0.73558828,  0.45568838],
       [ 0.24606968,  0.73890972,  0.45202405],
       [ 0.25289851,  0.74221104,  0.44828355],
       [ 0.25985676,  0.74549162,  0.44446673],
       [ 0.26694127,  0.74875084,  0.44057284],
       [ 0.27414922,  0.75198807,  0.4366009 ],
       [ 0.28147681,  0.75520266,  0.43255207],
       [ 0.28892102,  0.75839399,  0.42842626],
       [ 0.29647899,  0.76156142,  0.42422341],
       [ 0.30414796,  0.76470433,  0.41994346],
       [ 0.31192534,  0.76782207,  0.41558638],
       [ 0.3198086 ,  0.77091403,  0.41115215],
       [ 0.3277958 ,  0.77397953,  0.40664011],
       [ 0.33588539,  0.7770179 ,  0.40204917],
       [ 0.34407411,  0.78002855,  0.39738103],
       [ 0.35235985,  0.78301086,  0.39263579],
       [ 0.36074053,  0.78596419,  0.38781353],
       [ 0.3692142 ,  0.78888793,  0.38291438],
       [ 0.37777892,  0.79178146,  0.3779385 ],
       [ 0.38643282,  0.79464415,  0.37288606],
       [ 0.39517408,  0.79747541,  0.36775726],
       [ 0.40400101,  0.80027461,  0.36255223],
       [ 0.4129135 ,  0.80304099,  0.35726893],
       [ 0.42190813,  0.80577412,  0.35191009],
       [ 0.43098317,  0.80847343,  0.34647607],
       [ 0.44013691,  0.81113836,  0.3409673 ],
       [ 0.44936763,  0.81376835,  0.33538426],
       [ 0.45867362,  0.81636288,  0.32972749],
       [ 0.46805314,  0.81892143,  0.32399761],
       [ 0.47750446,  0.82144351,  0.31819529],
       [ 0.4870258 ,  0.82392862,  0.31232133],
       [ 0.49661536,  0.82637633,  0.30637661],
       [ 0.5062713 ,  0.82878621,  0.30036211],
       [ 0.51599182,  0.83115784,  0.29427888],
       [ 0.52577622,  0.83349064,  0.2881265 ],
       [ 0.5356211 ,  0.83578452,  0.28190832],
       [ 0.5455244 ,  0.83803918,  0.27562602],
       [ 0.55548397,  0.84025437,  0.26928147],
       [ 0.5654976 ,  0.8424299 ,  0.26287683],
       [ 0.57556297,  0.84456561,  0.25641457],
       [ 0.58567772,  0.84666139,  0.24989748],
       [ 0.59583934,  0.84871722,  0.24332878],
       [ 0.60604528,  0.8507331 ,  0.23671214],
       [ 0.61629283,  0.85270912,  0.23005179],
       [ 0.62657923,  0.85464543,  0.22335258],
       [ 0.63690157,  0.85654226,  0.21662012],
       [ 0.64725685,  0.85839991,  0.20986086],
       [ 0.65764197,  0.86021878,  0.20308229],
       [ 0.66805369,  0.86199932,  0.19629307],
       [ 0.67848868,  0.86374211,  0.18950326],
       [ 0.68894351,  0.86544779,  0.18272455],
       [ 0.69941463,  0.86711711,  0.17597055],
       [ 0.70989842,  0.86875092,  0.16925712],
       [ 0.72039115,  0.87035015,  0.16260273],
       [ 0.73088902,  0.87191584,  0.15602894],
       [ 0.74138803,  0.87344918,  0.14956101],
       [ 0.75188414,  0.87495143,  0.14322828],
       [ 0.76237342,  0.87642392,  0.13706449],
       [ 0.77285183,  0.87786808,  0.13110864],
       [ 0.78331535,  0.87928545,  0.12540538],
       [ 0.79375994,  0.88067763,  0.12000532],
       [ 0.80418159,  0.88204632,  0.11496505],
       [ 0.81457634,  0.88339329,  0.11034678],
       [ 0.82494028,  0.88472036,  0.10621724],
       [ 0.83526959,  0.88602943,  0.1026459 ],
       [ 0.84556056,  0.88732243,  0.09970219],
       [ 0.8558096 ,  0.88860134,  0.09745186],
       [ 0.86601325,  0.88986815,  0.09595277],
       [ 0.87616824,  0.89112487,  0.09525046],
       [ 0.88627146,  0.89237353,  0.09537439],
       [ 0.89632002,  0.89361614,  0.09633538],
       [ 0.90631121,  0.89485467,  0.09812496],
       [ 0.91624212,  0.89609127,  0.1007168 ],
       [ 0.92610579,  0.89732977,  0.10407067],
       [ 0.93590444,  0.8985704 ,  0.10813094],
       [ 0.94563626,  0.899815  ,  0.11283773],
       [ 0.95529972,  0.90106534,  0.11812832],
       [ 0.96489353,  0.90232311,  0.12394051],
       [ 0.97441665,  0.90358991,  0.13021494],
       [ 0.98386829,  0.90486726,  0.13689671],
       [ 0.99324789,  0.90615657,  0.1439362 ]]

viridis = mpl.colors.LinearSegmentedColormap.from_list(__file__, cm_data)