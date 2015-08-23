"""
.. module:: loglikelihood
    :platform: Windows
    :synopsis: Implementation of calculation of the
     loglikelihood for common distributions

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""

import numpy as np
from numba import autojit

@autojit
def Poisson(x, l):
    """Returns the loglikelihood for a Poisson distribution.
    In this calculation, it is assumed that the parameters
    are true, and the loglikelihood that the data is drawn from
    the distribution established by the parameters is calculated.

    Parameters
    ----------
    x : array_like
        Data that has to be tested.
    l : array_like
        Parameter for the Poisson distribution.

    Returns
    -------
    array_like
        Array with loglikelihoods for the data."""
    return x * np.log(l) - l


@autojit
def Gaussian(x, l):
    """Returns the loglikelihood for a Gaussian distribution,
    assuming the variance is given by the square root of the data
    points. It is assumed that the parameters are true, and the
    loglikelihood that the data is drawn from the distribution
    established by the parameters is calculated.

    Parameters
    ----------
    x : array_like
        Data that has to be tested.
    l : array_like
        Parameter for the Poisson distrbution.

    Returns
    -------
    array_like
        Array with the loglikelihoods for the data"""
    s = x ** 0.5
    return -((x - l)/(2 * s)) ** 2
    # return -np.log(np.sqrt(2*np.pi)*s)-(x-l)**2/(2.0*s**2)
