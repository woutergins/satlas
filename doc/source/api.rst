.. _api_ref:

API reference
=============

.. currentmodule:: satlas

BaseModel creation
------------------

.. note::

    The abstract baseclass :class:`.baseModel` defines a few methods for retrieving information about the current state of the fit. These methods are not documented in the child classes, but will be regularly used.

.. autosummary::
    :toctree: generated/

    satlas.basemodel.BaseModel
    satlas.hfsmodel.HFSModel
    satlas.multimodel.MultiModel
    satlas.combinedmodel.CombinedModel

Fitting routines
----------------

.. autosummary::
    :toctree: generated/

    satlas.fitting

Lineshapes
----------

.. autosummary::
    :toctree: generated/

    satlas.profiles

Utilities
---------

.. autosummary::
    :toctree: generated/

    satlas.utilities
