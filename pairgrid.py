from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


from seaborn import utils
from seaborn import PairGrid


class myPairGrid(PairGrid):
    """Subplot grid for plotting pairwise relationships in a dataset."""

    def __init__(self, *args, **kwargs):
        super(myPairGrid, self).__init__(*args, **kwargs)

        size = kwargs.pop("size", 3)
        aspect = kwargs.pop("aspect", 1)
        despine = kwargs.pop("despine", True)
        # Create the figure and the array of subplots
        figsize = len(self.x_vars) * size * aspect, len(self.y_vars) * size

        plt.close(self.fig)

        fig, axes = plt.subplots(len(self.y_vars), len(self.x_vars),
                                 figsize=figsize,
                                 # sharex="col", sharey="row",
                                 squeeze=False)

        self.fig = fig
        self.axes = axes

        l, b, r, t = 0.25 * size *aspect / figsize[0], 0.4 * size / figsize[1], 1 - 0.1 * size * aspect / figsize[0], 1 - 0.2 * size * aspect / figsize[1]
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
            utils.despine(fig=fig)

    def map(self, func, **kwargs):
        """Plot with the same function in every subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.

        """
        kw_color = kwargs.pop("color", None)
        for i, y_var in enumerate(self.y_vars):
            for j, x_var in enumerate(self.x_vars):
                hue_grouped = self.data.groupby(self.hue_vals)
                for k, (label_k, data_k) in enumerate(hue_grouped):
                    ax = self.axes[i, j]
                    plt.sca(ax)

                    # Insert the other hue aesthetics if appropriate
                    for kw, val_list in self.hue_kws.items():
                        kwargs[kw] = val_list[k]

                    color = self.palette[k] if kw_color is None else kw_color
                    func(data_k[x_var], data_k[y_var],
                         label=label_k, color=color, **kwargs)

                self._clean_axis(ax)
                self._update_legend_data(ax)

        if kw_color is not None:
            kwargs["color"] = kw_color
        self._add_axis_labels()

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

    def map_lower(self, func, **kwargs):
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
                func(data_k[x_var], data_k[y_var], label=label_k,
                     color=color, **kwargs)

            self._clean_axis(ax)
            self._update_legend_data(ax)

        if kw_color is not None:
            kwargs["color"] = kw_color
        self._add_axis_labels()

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
