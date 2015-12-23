.. _api_ref:

API reference
=============

.. currentmodule:: satlas

Models
------

.. note::

    The abstract baseclass :class:`.BaseModel` defines a few methods for retrieving information about the current state of the fit. These methods are not documented in the child classes, but will be regularly used.

.. autosummary::
    :toctree: generated/

    satlas.basemodel.BaseModel
    satlas.hfsmodel.HFSModel
    satlas.multimodel.MultiModel
    satlas.combinedmodel.CombinedModel

Fitting routines
----------------

.. autosummary::
      :toctree: fitting/

      satlas.fitting.calculate_analytical_uncertainty
      satlas.fitting.chisquare_fit
      satlas.fitting.chisquare_model
      satlas.fitting.chisquare_spectroscopic_fit
      satlas.fitting.likelihood_fit
      satlas.fitting.likelihood_walk
      satlas.fitting.likelihood_x_err
      satlas.fitting.likelihood_lnprob
      satlas.fitting.likelihood_loglikelihood

Lineshapes
----------

.. automodule:: satlas.profiles

   .. autosummary::
      :toctree: profiles/

      satlas.profiles.Crystalball
      satlas.profiles.Gaussian
      satlas.profiles.Lorentzian
      satlas.profiles.Voigt


Utilities
---------

.. automodule:: satlas.utilities

   .. autosummary::
      :toctree: utilities/

      satlas.utilities.generate_correlation_plot
      satlas.utilities.generate_correlation_map
      satlas.utilities.generate_spectrum
      satlas.utilities.load_model
      satlas.utilities.poisson_interval
      satlas.utilities.weighted_average
