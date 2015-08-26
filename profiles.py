"""
.. module:: profiles
    :platform: Windows
    :synopsis: Implementation of classes for different lineshapes,
     creating callables for easy and intuitive calculations.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import numpy as np
from scipy.special import wofz
from scipy.interpolate import interp1d


__all__ = ['Gaussian', 'Lorentzian', 'Voigt', 'PseudoVoigt',
           'ExtendedVoigt', 'Irrational', 'HyperbolicSquared']
sqrt2 = 2 ** 0.5
sqrt2pi = (2 * np.pi) ** 0.5
sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
base_e = np.exp(1)


class Profile(object):

    def __init__(self, fwhm=None, mu=None, amp=None, ampIsArea=False):
        super(Profile, self).__init__()
        self.ampIsArea = ampIsArea
        self.fwhm = np.abs(fwhm) if fwhm is not None else np.abs(1.0)
        self.mu = mu if mu is not None else 0.0
        self.amp = amp if amp is not None else 1.0

    def __repr__(self):
        s = str(type(self)) + 'FWHM: {}, mu: {}, amp: {}'
        s = s.format(self.fwhm, self.mu, self.amp)
        return s

    def __call__(self, vals):
        if self.ampIsArea:
            factor = 1.0
        else:
            factor = self._normFactor
        vals = vals / factor
        return self.amp * vals


class Gaussian(Profile):

    r"""A callable normalized Gaussian profile.

Parameters
----------
fwhm: float
    Full Width At Half Maximum, defaults to 1.
mu: float
    Location of the center, defaults to 0.
amp: float
    Amplitude of the profile, defaults to 1.

Returns
-------
Gaussian
    Callable instance, evaluates the Gaussian profile in the arguments
    supplied.

Note
----
    The used formula is taken from the MathWorld webpage
    http://mathworld.wolfram.com/GaussianFunction.html:

        .. math::
            G(x;\mu, \sigma) &= \frac{\exp\left(-\frac{1}{2}\left(\frac{x-\mu}
            {\sigma}\right)^2\right)}{\sqrt{2\pi}\sigma}

            FWHM &= s\sqrt{2\ln\left(2\right)}\sigma"""

    def __init__(self, fwhm=None, mu=None, amp=None, **kwargs):
        super(Gaussian, self).__init__(fwhm=fwhm, mu=mu,
                                       amp=amp, **kwargs)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.sigma = self.fwhm / (sqrt2log2t2)
        if not self.ampIsArea:
            self._normFactor = (self.sigma * sqrt2pi) ** (-1)

    def __call__(self, x):
        x = x - self.mu
        expPart = np.exp(-0.5 * (x / self.sigma) ** 2)
        normPart = self.sigma * sqrt2pi
        return super(Gaussian, self).__call__(expPart / normPart)


class Lorentzian(Profile):

    r"""A callable normalized Lorentzian profile.

Parameters
----------
    fwhm: float
        Full Width At Half Maximum, defaults to 1.
    mu: float
        Location of the center, defaults to 0.
    amp: float
        Amplitude of the profile, defaults to 1.

Returns
-------
Lorentzian
    Callable instance, evaluates the Lorentzian profile in the arguments
    supplied.

Note
----
The formula used is taken from the MathWorld webpage
http://mathworld.wolfram.com/LorentzianFunction.html:

    .. math::
        \mathcal{L}\left(x; \mu, \gamma\right) &= \frac{\gamma}
        {\pi\left(\left(x-\mu\right)^2+\gamma^2\right)}

        FWHM &= 2\gamma"""

    def __init__(self, fwhm=None, mu=None, amp=None, **kwargs):
        super(Lorentzian, self).__init__(fwhm=fwhm, mu=mu,
                                         amp=amp, **kwargs)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.gamma = 0.5 * self.fwhm
        if not self.ampIsArea:
            self._normFactor = 1.0 / (self.gamma * np.pi)

    def __call__(self, x):
        x = x - self.mu
        topPart = self.gamma
        bottomPart = (x ** 2 + self.gamma ** 2) * np.pi
        return super(Lorentzian, self).__call__(topPart / bottomPart)


class Voigt(Profile):

    r"""A callable normalized Voigt profile.

Parameters
----------
fwhm: list of 2 floats
    Full Width At Half Maximum of the components, defaults to 1.
    Ordered as Gaussian, then Lorentzian.
mu: float
    Location of the center, defaults to 0.
amp: float
    Amplitude of the profile, defaults to 1.

Attributes
----------
totalfwhm: float
    Approximation of the width based on the underlying widths.

Returns
-------
Voigt
    Callable instance, evaluates the Voigt profile in the arguments supplied.

Note
----
The formula used is taken from the Wikipedia webpage
http://en.wikipedia.org/wiki/Voigt_profile, with :math:`w(z)` the Faddeeva
function, and the values supplied as FWHM are appropriately transformed to
:math:`\sigma` and :math:`\gamma`:

    .. math::
        V\left(x;\mu, \sigma, \gamma\right) &= \frac{\Re\left[w\left(z\right)
        \right]}{\sigma\sqrt{2\pi}}

        z&=\frac{x+i\gamma}{\sigma\sqrt{2\pi}}"""

    def __init__(self, fwhm=None, mu=None, amp=None, **kwargs):
        self._fwhmNorm = np.array([sqrt2log2t2, 2])
        super(Voigt, self).__init__(fwhm=fwhm, mu=mu,
                                    amp=amp, **kwargs)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            seperate = value[0:2]
            self.fwhmG, self.fwhmL = seperate
            G, L = seperate
            self._fwhm = 0.5346 * self.fwhmL + \
                         (0.2166 * self.fwhmL ** 2 + self.fwhmG ** 2) ** 0.5
            self.sigma, self.gamma = seperate / self._fwhmNorm
        else:
            self.fwhmG, self.fwhmL = value, value
            self._fwhm = 0.6144031129489123 * value
            self.sigma, self.gamma = self._fwhm / self._fwhmNorm

        if not self.ampIsArea:
            z = (0 + 1j * self.gamma) / (self.sigma * sqrt2)
            top = wofz(z).real / (self.sigma * sqrt2pi)
            self._normFactor = top

    def __call__(self, x):
        x = x - self.mu
        z = (x + 1j * self.gamma) / (self.sigma * sqrt2)
        top = wofz(z).real / (self.sigma * sqrt2pi)
        return super(Voigt, self).__call__(top)


class Irrational(Profile):

    r"""A callable normalized Irrational profile.

Parameters
----------
fwhm: float
    Full Width At Half Maximum, defaults to 1.
mu: float
    Location of the center, defaults to 0.
amp: float
    Amplitude of the profile, defaults to 1.

Returns
-------
Irrational
    Callable instance, evaluates the irrational profile in the arguments
    supplied.

Note
----
The used formula is taken from T. Ida et al. :cite:`Ida2000`,
code inspired by the PhD thesis of Deyan Yordanov :cite:`Yordanov2007`.

    .. math::
        \mathcal{I}\left(x; \mu, g\right) &= \frac{g}{2}\left[1+\left(\frac{x-
        \mu}{g}\right)^2\right]^{-3/2}

        FWHM &= \sqrt{2^{2/3}-1}g"""

    def __init__(self, fwhm=None, mu=None, amp=None, **kwargs):
        super(Irrational, self).__init__(fwhm=fwhm, mu=mu,
                                         amp=amp, **kwargs)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.gamma = self.fwhm / np.sqrt(np.power(2, 2.0 / 3) - 1)
        if not self.ampIsArea:
            self._normFactor = (1.0 ** (-1.5)) / (2 * self.gamma)

    def __call__(self, x):
        x = x - self.mu
        val = ((1.0 + (x / self.gamma) ** 2) ** (-1.5)) / (2 * self.gamma)
        return super(Irrational, self).__call__(val)


class HyperbolicSquared(Profile):

    r"""A callable normalized HyperbolicSquared profile.

Parameters
----------
fwhm: float
    Full Width At Half Maximum, defaults to 1.
mu: float
    Location of the center, defaults to 0.
amp: float
    Amplitude of the profile, defaults to 1.

Returns
-------
Hyperbolic
    Callable instance, evaluates the hyperbolic profile in the arguments
    supplied.

Note
----
The used formula is taken from T. Ida et al. :cite:`Ida2000`, code inspired by the PhD thesis of
Deyan Yordanov :cite:`Yordanov2007`.

    .. math::
        H\left(x;\mu, g\right) &= \frac{1}{2g}\cosh^{-2}\left(\frac{x-\mu}{g}
        \right)

        FWHM &= 2g\ln\left(\sqrt{2}+1\right)"""

    def __init__(self, fwhm=None, mu=None, amp=None, **kwargs):
        super(HyperbolicSquared, self).__init__(fwhm=fwhm, mu=mu,
                                                amp=amp, **kwargs)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.gamma = self.fwhm / (2 * np.log(np.sqrt(2) + 1))
        if not self.ampIsArea:
            self._normFactor = 1.0 / (2 * self.gamma)

    def __call__(self, x):
        x = x - self.mu
        coshPart = (1.0 / np.cosh(x / self.gamma)) ** 2
        simplePart = 2 * self.gamma
        return super(HyperbolicSquared, self).__call__(coshPart / simplePart)


class PseudoVoigt(Profile):

    r"""A callable normalized PseudoVoigt profile.

Parameters
----------
fwhm: float
    Full Width At Half Maximum, defaults to 1.
mu: float
    Location of the center, defaults to 0.
amp: float
    Amplitude of the profile, defaults to 1.

Returns
-------
PseudoVoigt
    Callable instance, evaluates the pseudovoigt profile in the arguments
    supplied.

Note
----
The formula used is taken from the webpage
http://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_Approximation, and the
supplied FWHM is appropriately transformed for the Gaussian and Lorentzian
lineshapes:

    .. math::
        \mathcal{V}\left(x; \mu, \eta, \sigma, \gamma\right) = \eta \mathcal{L}
        (x; \gamma, \mu) + (1-\eta) G(x; \sigma, \mu)"""

    def __init__(self, eta=None, fwhm=None, mu=None,
                 amp=None, **kwargs):
        self.L = Lorentzian(**kwargs)
        self.G = Gaussian(**kwargs)
        self._n = np.abs(eta) if eta is not None else 0.5
        if self._n > 1:
            self._n = self._n - int(self._n)
        super(PseudoVoigt, self).__init__(fwhm=fwhm, mu=mu,
                                          amp=amp, **kwargs)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.L.fwhm = value
        self.G.fwhm = value
        if not self.ampIsArea:
            self._normFactor = self.n * self.L(0)
            self._normFactor += (1.0 - self.n) * self.G(0)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        value = np.abs(value)
        if value > 1:
            value = value - int(value)
        self._n = value
        if not self.ampIsArea:
            self._normFactor = self.n * self.L(0)
            self._normFactor += (1.0 - self.n) * self.G(0)

    def __call__(self, x):
        x = x - self.mu
        val = self.n * self.L(x) + (1.0 - self.n) * self.G(x)
        return super(PseudoVoigt, self).__call__(val)


class ExtendedVoigt(Profile):

    r"""A callable normalized extended Voigt profile.

Parameters
----------
fwhm: list of 2 floats
    Full Width At Half Maximum, defaults to 1, ordered as Gaussian and
    Lorentzian width.
mu: float
    Location of the center, defaults to 0.
amp: float
    Amplitude of the profile, defaults to 1.

Attributes
----------
totalfwhm: float
    Approximation of the total width, based on the underlying widths.

Returns
-------
ExtendedVoigt
    Callable instance, evaluates the extended Voigt profile in the arguments
    supplied.

Note
----
Formula taken from T. Ida et al. :cite:`Ida2000`, code
inspired by the PhD thesis of Deyan Yordanov :cite:`Yordanov2007`.

This class uses a weighted sum of the Gaussian,
Lorentzian, Irrational and HyperbolicSquared profiles."""

    def __init__(self, fwhm=None, mu=None, amp=None, **kwargs):
        self.kwargs = kwargs
        super(ExtendedVoigt, self).__init__(fwhm=fwhm, mu=mu,
                                            amp=amp, **kwargs)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            seperate = value[0:2]
            self.fwhmG, self.fwhmL = seperate
            self._fwhm = 0.5346 * self.fwhmL + \
                         np.sqrt(0.2166 * self.fwhmL ** 2 + self.fwhmG ** 2)
        else:
            self.fwhmG, self.fwhmL = value, value
            self._fwhm = 0.6144031129489123 * value
        self.setParams()

    def setParams(self):
        a = np.array(
            [-2.95553, 8.48252, -9.48291,
             4.74052, -1.24984, 0.15021, 0.66])
        b = np.array(
            [3.19974, -16.50453, 29.14158,
             -23.45651, 10.30003, -1.25693, -0.42179])
        c = np.array(
            [-17.80614, 57.92559, -73.61822,
             47.06071, -15.36331,  1.43021, 1.19913])
        d = np.array(
            [-1.26571, 4.05475, -4.55466,
             2.76622, -0.68688, -0.47745, 1.10186])
        f = np.array(
            [3.7029, -21.18862, 34.96491,
             -24.10743, 9.3155, -1.38927, -0.30165])
        g = np.array(
            [9.76947, -24.12407, 22.10544,
             -11.09215, 3.23653, -0.14107, 0.25437])
        h = np.array(
            [-10.02142, 32.83023, -39.71134,
             23.59717, -9.21815, 1.50429, 1.01579])

        self.rho = self.fwhmL / (self.fwhmL + self.fwhmG)
        self.wG = np.polyval(a, self.rho)
        self.wL = np.polyval(b, self.rho)
        self.wI = np.polyval(c, self.rho)
        self.wH = np.polyval(d, self.rho)
        self.nL = np.polyval(f, self.rho)
        self.nI = np.polyval(g, self.rho)
        self.nH = np.polyval(h, self.rho)

        self.wG = s * (1 - self.rho * self.wG)
        self.wL = s * (1 - (1 - self.rho) * self.wL)
        self.wI = s * self.wI
        self.wH = s * self.wH
        self.nL = self.rho * (1 + (1 - self.rho) * self.nL)
        self.nI = self.rho * (1 - self.rho) * self.nI
        self.nH = self.rho * (1 - self.rho) * self.nH

        self.G = Gaussian(fwhm=self.wG, **self.kwargs)
        self.L = Lorentzian(fwhm=self.wL, **self.kwargs)
        self.I = Irrational(fwhm=self.wI, **self.kwargs)
        self.H = HyperbolicSquared(fwhm=self.wH, **self.kwargs)

        self.fwhmV = (self.fwhmG ** 5 +
                      2.69269 * (self.fwhmG ** 4) * self.fwhmL +
                      2.42843 * (self.fwhmG ** 3) * (self.fwhmL ** 2) +
                      4.47163 * (self.fwhmG ** 2) * (self.fwhmL ** 3) +
                      0.07842 * self.fwhmG * (self.fwhmL ** 4) +
                      self.fwhmL ** 5
                      ) ** (1.0 / 5)
        if not self.ampIsArea:
            Gauss = (1 - self.nL - self.nI - self.nH) * self.G(0)
            Lorentz = self.nL * self.L(0)
            Irrat = self.nI * self.I(0)
            Hyper = self.nH * self.H(0)
            val = Gauss + Lorentz + Irrat + Hyper
            self._normFactor = val

    def __call__(self, x):
        x = x - self.mu
        Gauss = (1 - self.nL - self.nI - self.nH) * self.G(x)
        Lorentz = self.nL * self.L(x)
        Irrat = self.nI * self.I(x)
        Hyper = self.nH * self.H(x)
        val = Gauss + Lorentz + Irrat + Hyper
        return super(ExtendedVoigt, self).__call__(val)
