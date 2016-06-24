"""
Implementation of calculation of the loglikelihood for common distributions.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import numpy as np

sqrt2pi = np.sqrt(2*np.pi)
__all__ = ['poisson_llh', 'gaussian_llh', 'create_gaussian_llh']


def poisson_llh(y, l):
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
    return y * np.log(l) - l


def gaussian_llh(y, l):
    """Returns the loglikelihood for a Gaussian distribution,
    assuming the variance is given by the square root of the fit value.
    It is assumed that the parameters are true, and the
    loglikelihood that the data is drawn from the distribution
    established by the parameters is calculated.

    Parameters
    ----------
    y : array_like
        Data to which is being fitted.
    l : array_like
        Result from the model.

    Returns
    -------
    array_like
        Array with the loglikelihoods for the data."""
    s = l ** 0.5
    deviation = (y-l)/s
    return -(0.5 * deviation * deviation + np.log(sqrt2pi*s))


def create_gaussian_llh(yerr=1):
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
    def func(y, l):
        """Returns the loglikelihood for a Gaussian distribution,
        with the given sigma for each datapoint.

        Parameters
        ----------
        y : array_like
            Data to which is being fitted.
        l : array_like
            Result from the model.

        Returns
        -------
        array_like
            Array with the loglikelihoods for the data."""
        deviation = (y - l) / yerr
        return -0.5 * deviation * deviation
    return func

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
