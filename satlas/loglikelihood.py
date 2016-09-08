"""
Implementation of calculation of the loglikelihood for common distributions.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import numpy as np
import scipy as sp

sqrt2pi = np.sqrt(2*np.pi)
__all__ = ['poisson_llh', 'create_gaussian_llh']


def poisson_llh(y, f, x):
    """Returns the loglikelihood for a Poisson distribution.
    In this calculation, it is assumed that the parameters
    are true, and the loglikelihood that the data is drawn from
    the distribution established by the parameters is calculated.

    Parameters
    ----------
    y : array_like
        Data to which is being fitted.
    l : array_like
        Result from the model.

    Returns
    -------
    array_like
        Array with loglikelihoods for the data."""
    l = np.hstack(f(x))
    return y * np.log(l) - l

def create_gaussian_llh(yerr=1, xerr=None, func=None):
    """Returns the loglikelihood-function for a Gaussian distribution,
    with the given uncertainty on the data points. The input parameters
    will be (in order) the data to be fitted and the model response.

    Parameters
    ----------
    yerr : array_like
        Measured uncertainties on the datapoint.

    Returns
    -------
    function
        Function that calculates the loglikelihood for the given data and model values."""

    if func is not None:
        if xerr is not None:
            def gaussian_llh(y, f, x, xerr=xerr):
                l = np.hstack(f(x))
                yerr = func(l)
                xerr = np.hstack((sp.misc.derivative(f, x, dx=1E-6) * xerr))
                bottom = np.sqrt(yerr * yerr + xerr * xerr)
                return -0.5*( (y - l) / bottom)**2
            return gaussian_llh
        else:
            def gaussian_llh(y, f, x):
                l = np.hstack(f(x))
                bottom = func(l)
                return -0.5*( (y - l) / bottom)**2
            return gaussian_llh
    else:
        if xerr is not None:
            def gaussian_llh(y, f, x, xerr=xerr, yerr=yerr):
                l = f(x)
                xerr = np.hstack((sp.misc.derivative(f, x, dx=1E-6) * xerr))
                bottom = np.sqrt(yerr * yerr + xerr * xerr)
                return -0.5*( (y - l) / bottom)**2
            return gaussian_llh
        else:
            def gaussian_llh(y, f, x, yerr=yerr):
                l = np.hstack(f(x))
                return -0.5*( (y - l) / yerr)**2
            return gaussian_llh

def create_gaussian_priormap(literature_value, uncertainty):
    """Generates a function that describes a Gaussian prior mapping around
    the given literature value with the given uncertainty.

    Parameters
    ----------
    literature_value : float
        Value for the parameter which is optimal.
    uncertainty : float
        Value for the uncertainty on the parameter.

    Returns
    -------
    function
        Function that calculates the prior value for the given
        parameter value."""
    def func(value):
        """Returns the Gaussian prior with center {:.2f}
        and {:.2f} sigma.

        Parameters
        ----------
        value : float
            Current value of the parameter.

        Returns
        -------
        float
            Value of the prior.""".format(literature_value, uncertainty)
        deviation = (value - literature_value) / uncertainty
        return -0.5 * deviation * deviation
    return func
