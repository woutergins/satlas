*************************************************************
SATLAS -- Statistical Analysis Toolbox for Laser Spectroscopy
*************************************************************
.. image:: https://zenodo.org/badge/10132/woutergins/satlas.svg
    :target: https://zenodo.org/badge/latestdoi/10132/woutergins/satlas
    :alt: DOI Identifier
    :scale: 100%

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :alt: License
    :scale: 100%

.. image:: https://img.shields.io/badge/Python-3.5-green.svg
    :alt: Python version
    :scale: 100%

.. image:: https://img.shields.io/badge/Tested_on-Windows-green.svg
    :alt: Supported Platform
    :scale: 100%

.. image:: https://img.shields.io/badge/Not_tested_on-Linux/Mac-red.svg
    :alt: Unsupported platform
    :scale: 100%

.. image:: https://travis-ci.org/woutergins/satlas.svg?branch=master
    :alt: Build status
    :scale: 100%


Purpose
=======
.. sidebar:: Contributors

    * Wouter Gins (wouter.gins@kuleuven.be)
    * Ruben de Groote (ruben.degroote@kuleuven.be)
    * Kara Marie Lynch (kara.marie.lynch@cern.ch)

This Python package has been created with the goal of creating an easier interface for the analysis of data gathered from laser spectroscopy experiments. Support for fitting the spectra, using both :math:`\chi^2`-fitting and Maximum Likelihood Estimation routines, are present.

Dependencies
============
This package depends on the following packages:

    * `NumPy <http://www.numpy.org/>`_
    * `SciPy <http://www.scipy.org/>`_
    * `Matplotlib <http://matplotlib.org/>`_
    * `sympy <http://www.sympy.org/>`_
    * `h5py <http://docs.h5py.org/en/latest/index.html>`_
    * `numdifftools <http://numdifftools.readthedocs.io/en/latest/>`_

The following packages are distributed as part of SATLAS:

    * `emcee <http://dan.iel.fm/emcee/current/>`_
    * `LMFIT <http://lmfit.github.io/lmfit-py/index.html>`_
    * `tqdm <https://github.com/tqdm/tqdm>`_

Code can be downloaded `here <https://github.com/woutergins/satlas>`_. Parts of the code have been based on other resources; this is signaled in the documentation when this is the case. Inspiration has been drawn from the `triangle.py` script, written by Dan Foreman-Mackey et al. :cite:`Foreman-Mackey2014`, for the correlation plot code.

Contents
========
.. toctree::
    :maxdepth: 1

    api
    tutorial
    references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
