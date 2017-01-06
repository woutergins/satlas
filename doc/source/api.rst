.. _api_ref:

API reference
=============

.. currentmodule:: satlas

Models
------

General Models
~~~~~~~~~~~~~~
.. note::

    The abstract baseclass :class:`.BaseModel` defines a few methods for retrieving information about the current state of the fit. These methods are not documented in the child classes, but will be regularly used.

.. autosummary::
    :toctree: generated/

    satlas.models.basemodel.BaseModel
    satlas.models.summodel.SumModel
    satlas.models.linkedmodel.LinkedModel
    satlas.models.models.MiscModel
    satlas.models.models.PolynomialModel

Specialized Models
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    satlas.models.hfsmodel.HFSModel
    satlas.models.transformmodel.TransformHFSModel

Fitting routines
----------------
.. automodule:: satlas.stats.fitting

.. autosummary::
      :toctree: fitting/

      satlas.stats.fitting.calculate_analytical_uncertainty
      satlas.stats.fitting.chisquare_fit
      satlas.stats.fitting.chisquare_model
      satlas.stats.fitting.chisquare_spectroscopic_fit
      satlas.stats.fitting.likelihood_fit
      satlas.stats.fitting.likelihood_walk
      satlas.stats.fitting.likelihood_x_err
      satlas.stats.fitting.likelihood_lnprob
      satlas.stats.fitting.likelihood_loglikelihood
      satlas.stats.fitting.create_band
      satlas.stats.fitting.assign_hessian_estimate
      satlas.stats.fitting.process_walk

Likelihood calculations
-----------------------

.. automodule:: satlas.loglikelihood

.. autosummary::
      :toctree: loglikelihood/

      satlas.loglikelihood.poisson_llh
      satlas.loglikelihood.create_gaussian_llh
      satlas.loglikelihood.create_gaussian_priormap

Lineshapes
----------

.. automodule:: satlas.profiles

   .. autosummary::
      :toctree: profiles/

      satlas.profiles.Crystalball
      satlas.profiles.Gaussian
      satlas.profiles.Lorentzian
      satlas.profiles.Voigt
      satlas.profiles.PseudoVoigt


Utilities
---------

.. automodule:: satlas.utilities.utilities

   .. autosummary::
      :toctree: utilities/

      satlas.utilities.utilities.generate_spectrum
      satlas.utilities.utilities.poisson_interval
      satlas.utilities.utilities.weighted_average
      satlas.utilities.utilities.beta
      satlas.utilities.utilities.dopplerfactor

Visualisations
--------------

.. automodule:: satlas.style

    .. autosummary::
      :toctree: style/

      satlas.style.set
      satlas.style.get_available_styles
      satlas.style.set_font

.. automodule:: satlas.utilities.plotting

    .. autosummary::
      :toctree: plotting/

      satlas.utilities.plotting.generate_correlation_map
      satlas.utilities.plotting.generate_correlation_plot
      satlas.utilities.plotting.generate_walk_plot
