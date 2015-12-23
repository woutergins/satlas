SATLAS -- Statistical Analysis Toolbox for Laser Spectroscopy
=============================================================
![alt text](https://zenodo.org/badge/10132/woutergins/satlas.svg 'DOI Identifier')
![alt text](https://img.shields.io/badge/License-MIT-blue.svg 'License')
![alt text](https://img.shields.io/badge/Python-3.4-green.svg 'Python version')
![alt text](https://img.shields.io/badge/Tested_on-Windows-green.svg 'Supported platform')
![alt text](https://img.shields.io/badge/Not_tested_on-Linux/Mac-red.svg 'Unsupported platform')
[![Build Status](https://travis-ci.org/woutergins/satlas.svg?branch=master)](https://travis-ci.org/woutergins/satlas)


Purpose
-------
Contributors:
* Wouter Gins (wouter.gins@fys.kuleuven.be)
* Ruben de Groote (ruben.degroote@fys.kuleuven.be)
* Kara Marie Lynch (kara.marie.lynch@cern.ch)

This Python package has been created with the goal of creating an easier interface for the analysis of data gathered from laser spectroscopy experiments. Support for fitting the spectra, using both :math:`\chi^2`-fitting and Maximum Likelihood Estimation routines, are present.

Dependencies
------------
This package makes use of the following packages:
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [LMFIT](http://lmfit.github.io/lmfit-py/index.html)
* [Matplotlib](http://matplotlib.org/)
* [emcee](http://dan.iel.fm/emcee/current/)
* [sympy](http://www.sympy.org/)
* [h5py](http://docs.h5py.org/en/latest/index.html)
* [progressbar2](http://progressbar-2.readthedocs.org/en/latest/)

Parts of the code have been based on other resources; this is signaled in the documentation when this is the case. Inspiration has been drawn from the `triangle.py` script, written by Dan Foreman-Mackey et al., for the correlation plot code.

A detailed documentation of the code can be found [here](http://woutergins.github.io/satlas/).

Installation
------------
A package is available on PyPi, so 'pip install satlas' should provide a working environment. Please note that the package is still in beta, and bugs might be present.
