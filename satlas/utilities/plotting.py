import copy

from satlas import tqdm
from satlas.stats import fitting
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

__all__ = ['plot_line_ids', 'generate_correlation_map', 'generate_correlation_plot']

# Code for 'plot_line_ids' taken from Prasanth Nair

def _convert_to_array(x, size, name):
    """Check length of array or convert scalar to array.

    Check to see is `x` has the given length `size`. If this is true
    then return Numpy array equivalent of `x`. If not then raise
    ValueError, using `name` as an idnetification. If len(x) returns
    TypeError, then assume it is a scalar and create a Numpy array of
    length `size`. Each item of this array will have the value as `x`.
    """
    try:
        l = len(x)
        if l != size:
            raise ValueError(
                "{0} must be scalar or of length {1}".format(
                    name, size))
    except TypeError:
        # Only one item
        xa = np.array([x] * size)  # Each item is a diff. object.
    else:
        xa = np.array(x)

    return xa

def get_line_flux(line_wave, wave, flux, **kwargs):
    """Interpolated flux at a given wavelength (calls np.intrep).
    """
    return np.interp(line_wave, wave, flux, **kwargs)

def unique_labels(line_labels):
    """If a label occurs more than once, add num. as suffix."""
    from collections import defaultdict
    d = defaultdict(int)
    for i in line_labels:
        d[i] += 1
    d = dict((i, k) for i, k in d.items() if k != 1)
    line_labels_u = []
    for lab in reversed(line_labels):
        c = d.get(lab, 0)
        if c >= 1:
            v = lab + "_num_" + str(c)
            d[lab] -= 1
        else:
            v = lab
        line_labels_u.insert(0, v)

    return line_labels_u

def get_box_loc(fig, ax, line_wave, arrow_tip, box_axes_space=0.06):
    """Box loc in data coords, given Fig. coords offset from arrow_tip.

    Parameters
    ----------
    fig: matplotlib Figure artist
        Figure on which the boxes will be placed.
    ax: matplotlib Axes artist
        Axes on which the boxes will be placed.
    arrow_tip: list or array of floats
        Location of tip of arrow, in data coordinates.
    box_axes_space: float
        Vertical space between arrow tip and text box in figure
        coordinates.  Default is 0.06.

    Returns
    -------
    box_loc: list of floats
        Box locations in data coordinates.

    Notes
    -----
    Note that this function is not needed if user provides both arrow
    tip positions and box locations. The use case is when the program
    has to automatically find positions of boxes. In the automated
    plotting case, the arrow tip is set to be the top of the Axes
    (outside this function) and the box locations are determined by
    `box_axes_space`.

    In Matplotlib annotate function, both the arrow tip and the box
    location can be specified. While calculating automatic box
    locations, it is not ideal to use data coordinates to calculate
    box location, since plots will not have a uniform appearance. Given
    locations of arrow tips, and a spacing in figure fraction, this
    function will calculate the box locations in data
    coordinates. Using this boxes can be placed in a uniform manner.

    """
    # Plot boxes in their original x position, at a height given by the
    # key word box_axes_spacing above the arrow tip. The default
    # is set to 0.06. This is in figure fraction so that the spacing
    # doesn't depend on the data y range.
    box_loc = []
    fig_inv_trans = fig.transFigure.inverted()
    for w, a in zip(line_wave, arrow_tip):
        # Convert position of tip of arrow to figure coordinates, add
        # the vertical space between top edge and text box in figure
        # fraction. Convert this text box position back to data
        # coordinates.
        display_coords = ax.transData.transform((w, a))
        figure_coords = fig_inv_trans.transform(display_coords)
        figure_coords[1] += box_axes_space
        display_coords = fig.transFigure.transform(figure_coords)
        ax_coords = ax.transData.inverted().transform(display_coords)
        box_loc.append(ax_coords)

    return box_loc

def adjust_boxes(line_wave, box_widths, left_edge, right_edge, max_iter=1000, adjust_factor=0.35, factor_decrement=3.0, fd_p=0.75):
    """Ajdust given boxes so that they don't overlap.

    Parameters
    ----------
    line_wave: list or array of floats
        Line wave lengths. These are assumed to be the initial y (wave
        length) location of the boxes.
    box_widths: list or array of floats
        Width of box containing labels for each line identification.
    left_edge: float
        Left edge of valid data i.e., wave length minimum.
    right_edge: float
        Right edge of valid data i.e., wave lengths maximum.
    max_iter: int
        Maximum number of iterations to attempt.
    adjust_factor: float
        Gap between boxes are reduced or increased by this factor after
        each iteration.
    factor_decrement: float
        The `adjust_factor` itself if reduced by this factor, after
        certain number of iterations. This is useful for crowded
        regions.
    fd_p: float
        Percentage, given as a fraction between 0 and 1, after which
        adjust_factor must be reduced by a factor of
        `factor_decrement`. Default is set to 0.75.

    Returns
    -------
    wlp, niter, changed: (float, float, float)
        The new y (wave length) location of the text boxes, the number
        of iterations used and a flag to indicated whether any changes to
        the input locations were made or not.

    Notes
    -----
    This is a direct translation of the code in lineid_plot.pro file in
    NASA IDLAstro library.

    Positions are returned either when the boxes no longer overlap or
    when `max_iter` number of iterations are completed. So if there are
    many boxes, there is a possibility that the final box locations
    overlap.

    References
    ----------
    + http://idlastro.gsfc.nasa.gov/ftp/pro/plot/lineid_plot.pro
    + http://idlastro.gsfc.nasa.gov/

    """
    # Adjust positions.
    niter = 0
    changed = True
    nlines = len(line_wave)

    wlp = line_wave[:]
    while changed:
        changed = False
        for i in range(nlines):
            if i > 0:
                diff1 = wlp[i] - wlp[i - 1]
                separation1 = (box_widths[i] + box_widths[i - 1]) / 2.0
            else:
                diff1 = wlp[i] - left_edge + box_widths[i] * 1.01
                separation1 = box_widths[i]
            if i < nlines - 2:
                diff2 = wlp[i + 1] - wlp[i]
                separation2 = (box_widths[i] + box_widths[i + 1]) / 2.0
            else:
                diff2 = right_edge + box_widths[i] * 1.01 - wlp[i]
                separation2 = box_widths[i]

            if diff1 < separation1 or diff2 < separation2:
                if wlp[i] == left_edge: diff1 = 0
                if wlp[i] == right_edge: diff2 = 0
                if diff2 > diff1:
                    wlp[i] = wlp[i] + separation2 * adjust_factor
                    wlp[i] = wlp[i] if wlp[i] < right_edge else \
                        right_edge
                else:
                    wlp[i] = wlp[i] - separation1 * adjust_factor
                    wlp[i] = wlp[i] if wlp[i] > left_edge else \
                        left_edge
                changed = True
            niter += 1
        if niter == max_iter * fd_p: adjust_factor /= factor_decrement
        if niter >= max_iter: break

    return wlp, changed, niter

def prepare_axes(wave, flux, fig=None, ax_lower=(0.1, 0.1), ax_dim=(0.85, 0.65)):
    """Create fig and axes if needed and layout axes in fig."""
    # Axes location in figure.
    if not fig:
        fig = plt.figure()
    ax = fig.add_axes([ax_lower[0], ax_lower[1], ax_dim[0], ax_dim[1]])
    ax.plot(wave, flux)
    return fig, ax

def plot_line_ids(wave, flux, line_wave, line_label1, label1_size=None, extend=True, **kwargs):
    """Label features with automatic layout of labels.

    Parameters
    ----------
    wave: list or array of floats
        Wave lengths of data.
    flux: list or array of floats
        Flux at each wavelength.
    line_wave: list or array of floats
        Wave length of features to be labelled.
    line_label1: list of strings
        Label text for each line.
    label1_size: list of floats
        Font size in points. If not given the default value in
        Matplotlib is used. This is typically 12.
    extend: boolean or list of boolean values
        For those lines for which this keyword is True, a dashed line
        will be drawn from the tip of the annotation to the flux at the
        line.
    kwargs: key value pairs
        All of these keywords are optional.

        The following keys are recognized:

          ax : Matplotlib Axes
              The Axes in which the labels are to be placed. If not
              given a new Axes is created.
          fig: Matplotlib Figure
              The figure in which the labels are to be placed. If `ax`
              if given then keyword is then ignored. The figure
              associated with `ax` is used. If `fig` and `ax` are not
              given then a new figure is created and an axes is added
              to it.
          arrow_tip: scalar or list of floats
              The location of the annotation point, in data coords. If
              the value is scalar then it is used for all. Default
              value is the upper bound of the Axes, at the time of
              plotting.
          box_loc: scalar or list of floats
              The y axis location of the text label boxes, in data
              units. The default is to place it above the `arrow_tip`
              by `box_axes_space` units in figure fraction length.
          box_axes_space: float
              If no `box_loc` is given then the y position of label
              boxes is set to `arrow_tip` + this many figure fraction
              units. The default is 0.06. This ensures that the label
              layout appearance is independent of the y data range.
          max_iter: int
              Maximum iterations to use. Default is set to 1000.

    Returns
    -------
    fig, ax: Matplotlib Figure, Matplotlib Axes
        Figure instance on which the labels were placed and the Axes
        instance on which the labels were placed. These can be used for
        further customizations. For example, some labels can be hidden
        by accessing the corresponding `Text` instance form the
        `ax.texts` list.

    Notes
    -----
    + By default the labels are placed along the top of the Axes. The
      annotation point is on the top boundary of the Axes at the y
      location of the line. The y location of the boxes are 0.06 figure
      fraction units above the annotation location. This value can be
      customized using the `box_axes_space` parameter. The value must
      be in figure fractions units. Y location of both labels and
      annotation points can be changed using `arrow_tip` and `box_loc`
      parameters.
    + If `arrow_tip` parameter is given then it is used as the
      annotation point. This can be a list in which case each line can
      have its own annotation point.
    + If `box_loc` is given, then the boxes are placed at this
      position. This too can be a list.
    + `arrow_tip` and `box_loc` are the "y" components of `xy` and
      `xyann` (deprecated name `xytext`) parameters accepted by the `annotate`
      function in Matplotlib.
    + If the `extend` keyword is True then a line is drawn from the
      annotation point to the flux at the line wavelength. The flux is
      calculated by linear interpolation. This parameter can be a list,
      with one value for each line.
    + The maximum iterations to be used can be customized using the
      `max_iter` keyword parameter.

    """
    wave = np.array(wave)
    flux = np.array(flux)
    line_wave = np.array(line_wave)
    line_label1 = np.array(line_label1)

    nlines = len(line_wave)
    assert nlines == len(line_label1), "Each line must have a label."

    if label1_size == None:
        label1_size = np.array([12] * nlines)
    label1_size = _convert_to_array(label1_size, nlines, "lable1_size")

    extend = _convert_to_array(extend, nlines, "extend")

    # Sort.
    indx = np.argsort(wave)
    wave[:] = wave[indx]
    flux[:] = flux[indx]
    indx = np.argsort(line_wave)
    line_wave[:] = line_wave[indx]
    line_label1[:] = line_label1[indx]
    label1_size[:] = label1_size[indx]

    # Flux at the line wavelengths.
    line_flux = get_line_flux(line_wave, wave, flux)

    # Figure and Axes. If Axes is given then use it. If not, create
    # figure, if not given, and add Axes to it using a default
    # layout. Also plot the data in the Axes.
    ax = kwargs.get("ax", None)
    if not ax:
        fig = kwargs.get("fig", None)
        fig, ax = prepare_axes(wave, flux, fig)
    else:
        fig = ax.figure

    # Find location of the tip of the arrow. Either the top edge of the
    # Axes or the given data coordinates.
    ax_bounds = ax.get_ybound()
    arrow_tip = kwargs.get("arrow_tip", ax_bounds[1])
    arrow_tip = _convert_to_array(arrow_tip, nlines, "arrow_tip")

    # The y location of boxes from the arrow tips. Either given heights
    # in data coordinates or use `box_axes_space` in figure
    # fraction. The latter has a default value which is used when no
    # box locations are given. Figure coordiantes are used so that the
    # y location does not dependent on the data y range.
    box_loc = kwargs.get("box_loc", None)
    if not box_loc:
        box_axes_space = kwargs.get("box_axes_space", 0.06)
        box_loc = get_box_loc(fig, ax, line_wave, arrow_tip, box_axes_space)
    else:
        box_loc = _convert_to_array(box_loc, nlines, "box_loc")
        box_loc = tuple(zip(line_wave, box_loc))

    # If any labels are repeated add "_num_#" to it. If there are 3 "X"
    # then the first gets "X_num_3". The result is passed as the label
    # parameter of annotate. This makes it easy to find the box
    # corresponding to a label using Figure.findobj.
    label_u = unique_labels(line_label1)

    # Draw boxes at initial (x, y) location.
    for i in range(nlines):
        ax.annotate(line_label1[i], xy=(line_wave[i], arrow_tip[i]),
                    xytext=(box_loc[i][0],
                            box_loc[i][1]),
                    xycoords="data", textcoords="data",
                    rotation=90, horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=label1_size[i],
                    arrowprops=dict(arrowstyle="-",
                                    relpos=(0.5, 0.0)),
                    label=label_u[i])
        if extend[i]:
            ax.plot([line_wave[i]] * 2, [arrow_tip[i], line_flux[i]],
                    linestyle="--", color="k",
                    scalex=False, scaley=False,
                    label=label_u[i] + "_line")

    # Draw the figure so that get_window_extent() below works.
    fig.canvas.draw()

    # Get annotation boxes and convert their dimensions from display
    # coordinates to data coordinates. Specifically, we want the width
    # in wavelength units. For each annotation box, transform the
    # bounding box into data coordinates and extract the width.
    ax_inv_trans = ax.transData.inverted()  # display to data
    box_widths = []  # box width in wavelength units.
    line_wave = []
    for box in ax.texts:
        b_ext = box.get_window_extent()
        box_widths.append(b_ext.transformed(ax_inv_trans).width)
        line_wave.append(box.xy[0])

    # Find final x locations of boxes so that they don't overlap.
    # Function adjust_boxes uses a direct translation of the equivalent
    # code in lineid_plot.pro in IDLASTRO.
    max_iter = kwargs.get('max_iter', 1000)
    adjust_factor = kwargs.get('adjust_factor', 0.35)
    factor_decrement = kwargs.get('factor_decrement', 3.0)
    wlp, niter, changed = adjust_boxes(line_wave, box_widths,
                                       np.min(wave), np.max(wave),
                                       adjust_factor=adjust_factor,
                                       factor_decrement=factor_decrement,
                                       max_iter=max_iter)

    # Redraw the boxes at their new x location.
    for i in range(nlines):
        box = ax.texts[i]
        if hasattr(box, 'xyann'):
            box.xyann = (wlp[i], box.xyann[1])
        else:
            box.xytext = (wlp[i], box.xytext[1])

    # Update the figure
    fig.canvas.draw()

    # Return Figure and Axes so that they can be used for further
    # manual customization.
    return fig, ax

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
                    ax.hist(x, int(bins[bin_index]), histtype='step')
                except TypeError:
                    bins[bin_index] = 50
                    ax.hist(x, int(bins[bin_index]), histtype='step')
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
