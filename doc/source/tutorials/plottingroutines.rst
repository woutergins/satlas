
Plotting routines
=================

For ease-of-use, standard implementations for plotting spectra have been
implemented. Each :class:`HFSModel` has a method to plot to an axis,
while both :class:`MultiModel` and :class:`CombinedModel` call this
plotting routine for the underlying spectrum.

Overview plotting
-----------------

Considering a :class:`HFSModel`, the standard plotting routines finds
out where the peaks in the spectrum are located, and samples around this
area taking the FWHM into account. Take this toy example of a spectrum
on a constant background:

.. code:: python

    import satlas as s
    import numpy as np
    np.random.seed(0)

    I = 1.0
    J = [1.0, 2.0]

    ABC = [1000, 500, 30, 400, 0, 0]
    df = 0
    scale = 10
    background = [1]

    model = s.HFSModel(I, J, ABC, df, background_params=background, scale=scale)
    model.plot()


.. parsed-literal::

    C:\Anaconda3\lib\site-packages\IPython\html.py:14: ShimWarning: The `IPython.html` package has been deprecated. You should import from `notebook` instead. `IPython.html.widgets` has moved to `ipywidgets`.
      "`IPython.html.widgets` has moved to `ipywidgets`.", ShimWarning)
    C:\Anaconda3\lib\site-packages\matplotlib\__init__.py:872: UserWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      warnings.warn(self.msg_depr % (key, alt_key))



.. image:: output_1_1.png




This provides a quick overview of the entire spectrum.

Plotting with data
------------------

When data is available, it can be plotted alongside the spectrum.

.. code:: python

    x = np.linspace(model.locations.min() - 300, model.locations.max() + 300, 50)
    y = model(x) + 0.5*np.random.randn(x.size) * model(x)**0.5
    y = np.where(y<0, 0, y)
    model.plot(x=x, y=y)



.. image:: output_3_0.png




Errorbars can be plotted by either supplying them in the *yerr* keyword,
or by using the *plot\_spectroscopic* method. this method, instead of
using the symmetric errorbars provided by calculating the square root of
the data point, calculate the asymmetric 68% coverage of the Poisson
distribution with the mean provided by the data point. Especially at
lower statistics, this is evident by the fact that the errorbars do not
cross below 0 counts.

.. code:: python

    model.plot_spectroscopic(x=x, y=y)



.. image:: output_5_0.png




Uncertainty on model
--------------------

The spectrum itself can also be displayed by showing the uncertainty on
the model value, interpreting the model value as the mean of the
corresponding Poisson distribution. The probability is then calculated
on a 2D grid of points, and colored depending on the value of the
Poisson pdf. A thin line is also drawn, representing the modelvalue and
thus the mean of the distribution.

.. code:: python

    model.plot(model=True)



.. image:: output_7_0.png




This plot can be displayed in each colormap provided by matplotlib by
specifying the colormap as a string.

.. code:: python

    model.plot(model=True, colormap='gnuplot2_r')
    model.plot(model=True, colormap='plasma')



.. image:: output_9_0.png



.. image:: output_9_1.png





The data can also be plotted on top of this imagemap.

.. code:: python

    model.plot(x=x, y=y, model=True, colormap='gnuplot2_r')



.. image:: output_11_0.png
