import satlas.loglikelihood
import numpy as np

def test_poissonlikelihood():
    x, l = 0, 1
    return satlas.loglikelihood.poisson_llh(x, l) == (x * np.log(l) - l)

def test_gaussianlikelihood():
    x, l = 0, 1
    return satlas.loglikelihood.gaussian_llh(x, l) == -(((x - l) / (2 * (l**0.5))) ** 2 + np.log(np.sqrt(2 * np.pi) * l**0.5))