"""
Experimental implementation of a plotter class -- to be fed data and models as a repository, which then plots everything.

.. moduleauthor:: Wouter gins <wouter.gins@kuleuven.be>
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

__all__ = ['Plotter']

class Plotter(object):
    """Convenience class for plotting, with support for
    data and BaseModel additions.

    Parameters
    ----------
    xdata: list
        List of measured x-data
    ydata: list
        List of measured y-data
    errors: dict
        Dictionary of the form {'x': list, 'y': list}
    models: list
        List of models to plot
    names: list
        List of label names corresponding to the models.
    left: float
        Left-edge that has to be used for sampling the models.
        When also plotting data, lowest left border gets selected.
    right: float
        Right-edge that has to be used for sampling the models.
        When also plotting data, highest right border gets selected.
    fig_kws: dict
        Dictionary of keywords to pass on the the figure upon creation.
    sampling: float
        Number of points to sample the models for.
    band: list of dicts
        List of dictionaries containing the keywords for creating prediction
        or confidence bands around the models. *band[0]* gets used for the
        first model, etc. Leave as *None* for no band creation."""
    def __init__(self, xdata=None, ydata=None, errors={'x': [], 'y': []}, models=None, names=None, left=None, right=None, fig_kws={}, sampling=1000, band=None):
        super(Plotter, self).__init__()
        self.xdata = xdata if xdata is not None else []
        self.ydata = ydata if ydata is not None else []
        self.errors = errors
        self.models = models if models is not None else []
        self.names = names
        self.left = left
        self.right = right
        self.fig_kws = {}
        self.sampling = sampling
        self.band = band
        self.line_cycle = ['-', '--', '-.', ':']

    def add_data(self, xdata, ydata):
        """Add data to the plot.

        Parameters
        ----------
        xdata: list
            List of data, gets added to the previous xdata
        ydata: list
            List of data, gets added to the previous ydata"""
        self.xdata.extend(xdata)
        self.ydata.extend(ydata)

    def add_errors(self, errors):
        """Adds uncertainties to the data point.

        Parameters
        ----------
        errors: dict
            Dictionary of the form {'x': list, 'y': list}."""
        self.errors['x'].extend(errors['x'])
        self.errors['y'].extend(errors['y'])

    def add_model(self, model):
        """Adds a model to the models to be plotted.

        Parameters
        ----------
        model: :class:`.BaseModel`
            Instance of :class:`.BaseModel` that has to be plotted."""
        self.models.extend(copy.deepcopy([model]))

    def set_range(self, ranges):
        """Set the range of the plot.

        Parameters
        ----------
        ranges: 2-tuple of floats
            Tuple of the form (left, right)"""
        self.left, self.right = ranges

    def set_sampling(self, value):
        """Set the sampling number for the model function.

        Parameters
        ----------
        value: integer
            Sets the number of values the models need to be sampled in the range"""
        self.sampling = int(value)

    def plot(self):
        """Create the plot described by the already given parameters."""
        fig = plt.figure(*self.fig_kws)
        ax = fig.add_subplot(1, 1, 1)
        if self.xdata is not None and self.ydata is not None:
            try:
                ax.errorbar(self.xdata, self.ydata, xerr=self.errors['x'] if 'x' in self.errors.keys() else None, yerr=self.errors['y'] if 'y' in self.errors.keys() else None, fmt='o', label='Data')
            except:
                ax.plot(self.xdata, self.ydata, 'o', label='Data')
            limits = list(ax.get_xlim())
            try:
                limits[0] = min(limits[0], self.left)
            except TypeError:
                pass
            try:
                limits[1] = max(limits[1], self.right)
            except TypeError:
                pass
        else:
            limits = [self.left, self.right]
        if limits[0] is None:
            raise ValueError("When not supplying data, sampling limits have to be provided!")
        sampling_x = np.linspace(limits[0], limits[1], self.sampling)
        for i, model in enumerate(self.models):
            sampling_y = model(sampling_x)
            try:
                line, = ax.plot(sampling_x, sampling_y, linestyle=self.line_cycle[i%len(self.line_cycle)], label=self.names[i])
            except:
                line, = ax.plot(sampling_x, sampling_y, linestyle=self.line_cycle[i%len(self.line_cycle)], label='Model {:.0i}'.format(i))
            try:
                if self.band[i] is not None:
                    from .fitting import createBand
                    kwargs = self.band[i]
                    deviation = createBand(model, sampling_x, self.xdata, self.ydata, self.errors['y'], xerr=self.errors['x'], **kwargs)
                    ax.fill_between(sampling_x, sampling_y + deviation, sampling_y - deviation, alpha=0.3, color=line.get_color(), label=r'$1\sigma$ ' + kwargs['kind'].lower() + ' band')
            except:
                pass
        ax.legend(loc=0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
