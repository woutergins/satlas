import satlas.profiles
import numpy as np

def test_gaussian():
    mu = 0
    fwhm = 1
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    amp = 1
    prof = satlas.profiles.Gaussian(mu=mu, fwhm=fwhm, amp=amp, ampIsArea=False)
    return np.isclose(prof(mu), amp)

def test_lorentzian():
    mu = 0
    fwhm = 1
    gamma = fwhm / 2
    amp = 1
    prof = satlas.profiles.Lorentzian(mu=mu, fwhm=fwhm, amp=amp, ampIsArea=False)
    return np.isclose(prof(mu), amp)

def test_voigt():
    mu = 0
    fwhm = 1
    gamma = fwhm / 2
    amp = 1
    prof = satlas.profiles.Voigt(mu=mu, fwhm=fwhm, amp=amp, ampIsArea=False)
    return np.isclose(prof(mu), amp)