BaseModel creation and evaluation
=================================

HFSModel creation and adaption
------------------------------

Creation and value change
~~~~~~~~~~~~~~~~~~~~~~~~~

For normal basemodel creation, the only package needed is the satlas package itself.

.. code:: python

    import satlas as s

First, define the nuclear spin, the electronic spins of both levels and the hyperfine parameters.

.. code:: python

    I = 1.0
    J = [1.0, 2.0]

    ABC = [100, 200, 100, 200, 0, 0]

Other parameters, such as the FWHM, centroid, scale and background can also be set at creation.

.. code:: python

    fwhm = [10, 10]
    centroid = 500
    scale = 100
    background = 10

Then, the basemodel can be created by instantiating a :class:`.HFSModel` object:

.. code:: python

    basemodel_low = s.HFSModel(I, J, ABC, centroid, fwhm=fwhm, scale=scale, background=10)

If a value has to be changed, pass the value and the parameter name to :meth:`.set_value`. For multiple values, a list of values and names can be given.

.. code:: python

    basemodel_low.set_value(0, 'Bl')  #Set Bl to 0.
    values = [200, 0]
    names = ['Au', 'Bu']
    basemodel_low.set_value(values, names)  #Sets Au to 200 and Bu to 0

Setting conditions
~~~~~~~~~~~~~~~~~~

When fitting, it might be desirable to restrict parameters to a certain boundary. Since this brings about possible numerical instabilities, only a few parameters have standard restrictions. The FWHM, amplitude and scale, and the Poisson intensity have been restricted to have **at least** a value of 0, while the Poisson offset has been restricted to have a value of **at most** 0. All other values have no restrictions placed on them.
In order to impose these restrictions, or to overwrite them, create a dictionary with parameter names as keys, and map them to a dictionary containing the *min* and *max* keys with a value. Pass this dictionary to :meth:`.set_boundaries`.

.. code:: python

    boundariesDict = {'Al': {'min': 50, 'max': 150},  #Constrain Al to be between 50 and 150 MHz.
                      'scale': {'min': None, 'max': None}}  #Remove the constraints on the scale
    basemodel_low.set_boundaries(boundariesDict)

In case a certain parameter is known, it can also be fixed so the fitting routines do not change it. This is done by creating a dictionary, again using the parameter names as keys, and mapping them to either *True* (meaning vary) or *False* (meaning fix).

.. code:: python

    variationDict = {'Background': False}  #Fixes the background to the current value
    basemodel_low.set_variation(variationDict)

.. note::

    Please note that the parameter *N*, responsible for the number of sidepeaks that appear in the basemodel, will **never** be varied. This value always has to be changed manually!

Another option is restricting the amplitude of the peaks to Racah amplitudes. This is done by default. If this is not desired, either pass to option *racah_int=False* to the initialiser, or change the attribute later on:

.. code:: python

    basemodel_low.racah_int = False

A final condition that can be placed is the restriction of the ratio of the hyperfine parameters. Using the method :meth:`.fix_ratio`, the value, target and parameter are specified. The target is defined as the parameter which will be calculated using the value.

.. code:: python

    basemodel_low.fix_ratio(2, target='upper', parameter='A')  #Fixes Au to 2*Al
    basemodel_low.fix_ratio(0.5, target='lower', parameter='B')  #Fixes Bl to 0.5*Bl

Additionally, the location of the peaks can be easily retrived by looking at :attr:`.locations`, with the labelling of the peaks being saved in :attr:`ftof`.

MultiModel creation
-------------------

In order to make an :class:`.MultiModel`, which takes another isomer or isotope into account, two options are available for creation, with both being equivalent. The first option is initialising the :class:`.MultiModel` with a list containing :class:`.HFSModel` objects.

.. code:: python

    I = 4.0
    centroid = 0

    basemodel_high = s.HFSModel(I, J, ABC, centroid, scale=scale)  #Make another basemodel, with a different nuclear spin and centroid

    basemodel_both = s.MultiModel([basemodel_low, basemodel_high])

The other option is simply adding the :class:`.HFSModel` objects together, making use of operator overloading.

.. code:: python

    basemodel_both = basemodel_low + basemodel_high  #Both methods give the exact same result!

There is no restriction on how many spectra can be combined in either way. Afterwards, the easiest way to add another :class:`.HFSModel` is by summing this with the :class:`.MultiModel`.

.. code:: python

    centroid = 600

    basemodel_high_shifted = s.HFSModel(I, J, ABC, centroid, scale=scale)

    basemodel_three = basemodel_both + basemodel_high_shifted  #Adds a third basemodel

When combining spectra in this way, parameters can be forced to be a shared value. This is done by accessing the :attr:`.MultiModel.shared` attribute.
By default this is set to an empty list, meaning no parameters are shared.

.. code:: python

    basemodel_both.shared = ['FWHMG', 'FWHML']  #Makes sure the same linewidth is used for all spectra


CombinedModel creation
----------------------

Making a :class:`.CombinedModel` uses the same syntax as the first method of creating an :class:`.MultiModel`:

.. code:: python

    basemodel_seperate = s.CombinedModel([basemodel_low, basemodel_low])

In the same way as for an :class:`.MultiModel`, parameters can be shared between spectra. By default, this is set to the hyperfine parameters and the sidepeak offset.

Evaluating spectra
------------------

The response of the basemodel for a frequency (which is the estimated average number of counts) is calculated by calling any :class:`.BaseModel` object with the frequency. There are some caveats:

    #. For a :class:`.CombinedModel`, a float cannot be given. The method expects a list of floats, or list of arrays, with a length equal to the number of spectra that have been combined. The output, in contrast to the other objects, is again a list of floats or arrays.
    #. When evaluating an :class:`.MultiModel`, the response is the **total** response. If the seperate response of each basemodel is required, the convenience method :meth:`.MultiModel.seperate_response` takes a list of floats or arrays and outputs the response of each basemodel. Note the keyword *background* in this method, which changes the output significantly.

.. code:: python

    import numpy as np

    lowest_freq = 0
    highest_freq = 10  #This is a toy example, so the values don't matter.
    freq_range = np.linspace(0, 10, 20)  #Consult the NumPy documentation for more information about generating ranges.

    response_hfsmodel = basemodel_low(freq_range)
    response_multimodel = basemodel_both(freq_range)
    response_combinedmodel = basemodel_seperate([freq_range, freq_range])
