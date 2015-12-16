SATLAS -- Statistical Analysis Toolbox for Laser Spectroscopy
=============================================================
![alt text](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18444-blue.svg 'DOI Identifier')
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
* Hanne Heylen (hanne.heylen@fys.kuleuven.be)

This Python package has been created with the goal of creating an easier interface for the analysis of data gathered from laser spectroscopy experiments. Support for fitting the spectra, using both :math:`\chi^2`-fitting and Maximum Likelihood Estimation routines, are present, as well as interfaces for simulation of polarization and spin-lattice relaxation phenomena.

Dependencies
------------
This package makes heavy use of the following packages::
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [LMFIT](http://lmfit.github.io/lmfit-py/index.html)
* [PyQtGraph](http://www.pyqtgraph.org/)
* [emcee](http://dan.iel.fm/emcee/current/)
* [seaborn](http://stanford.edu/~mwaskom/software/seaborn/)

Parts of the code have been based on other resources; this is signaled in the documentation when this is the case. Also included in the package are a temporary bugfix for the `sympy.physics.wigner` module, in order to calculate the Wigner symbols. Also included is the `triangle.py` script, written by Dan Foreman-Mackey et al.

A detailed documentation of the code can be found [here](http://woutergins.github.io/satlas/).

Installation
------------
At the moment, automatic installation is not yet possible. The best option is to download the master branch as a zip file, and unzip to your Python path.