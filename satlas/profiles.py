"""
Implementation of classes for different lineshapes, creating callables for easy and intuitive calculations.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
.. moduleauthor:: Kara Marie Lynch <kara.marie.lynch@cern.ch>
"""
import numpy as np
from scipy.special import wofz


__all__ = ['Gaussian', 'Lorentzian', 'Voigt', 'PseudoVoigt', 'Crystalball']
sqrt2 = 2 ** 0.5
sqrt2pi = (2 * np.pi) ** 0.5
sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
base_e = np.exp(1)


class Profile(object):
    """Abstract baseclass for defining lineshapes."""

    def __init__(self, fwhm=None, mu=None, amp=None, ampIsArea=False):
        super(Profile, self).__init__()
        self.ampIsArea = ampIsArea
        self.fwhm = np.abs(fwhm) if fwhm is not None else np.abs(1.0)
        self.mu = mu if mu is not None else 0.0
        self.amp = amp if amp is not None else 1.0

    @property
    def mu(self):
        """Location of the peak."""
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def fwhm(self):
        """FWHM of the peak."""
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value

    @property
    def amp(self):
        """Peak amplitude."""
        return self._amp

    @amp.setter
    def amp(self, value):
        self._amp = value

    @property
    def ampIsArea(self):
        """Boolean controlling the behaviour of *amp*."""
        return self._ampIsArea

    @ampIsArea.setter
    def ampIsArea(self, value):
        self._ampIsArea = value

    def __repr__(self):
        s = str(type(self)) + 'FWHM: {}, mu: {}, amp: {}'
        s = s.format(self.fwhm, self.mu, self.amp)
        return s

    def __call__(self, vals):
        """Evaluates the lineshape in the given values.

        Parameters
        ----------
        vals: array_like
            Array of values to evaluate the lineshape in.

        Returns
        -------
        array_like
            Array of seperate response values of the lineshape."""
        if self.ampIsArea:
            factor = 1.0
        else:
            factor = self._normFactor
        vals = vals / factor
        return self.amp * vals


class Gaussian(Profile):

    r"""A callable normalized Gaussian profile."""

    def __init__(self, fwhm=None, mu=None, amp=None, ampIsArea=False):
        """Creates a callable object storing the fwhm, amplitude and location
        of a Gaussian lineshape.

        Parameters
        ----------
        fwhm: float
            Full Width At Half Maximum, defaults to 1.
        mu: float
            Location of the center, defaults to 0.
        amp: float
            Amplitude of the profile, defaults to 1.
        ampIsArea: boolean
            Sets if the amplitude is the integral or the peakheight. Defaults
            to False.

        Returns
        -------
        Gaussian
            Callable instance, evaluates the Gaussian profile in the arguments
            supplied."""
        super(Gaussian, self).__init__(fwhm=fwhm, mu=mu,
                                       amp=amp, ampIsArea=ampIsArea)

    @property
    def fwhm(self):
        """FWHM of the peak."""
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.sigma = self.fwhm / (sqrt2log2t2)
        if not self.ampIsArea:
            self._normFactor = (self.sigma * sqrt2pi) ** (-1)

    def __call__(self, x):
        r"""Evaluates the lineshape in the given values.

        Parameters
        ----------
        vals: array_like
            Array of values to evaluate the lineshape in.

        Returns
        -------
        array_like
            Array of seperate response values of the lineshape.

        Note
        ----
        The used formula is taken from the MathWorld webpage
        http://mathworld.wolfram.com/GaussianFunction.html:

            .. math::
                G(x;\mu, \sigma) &= \frac{\exp\left(-\frac{1}{2}\left(\frac{x-\mu}
                {\sigma}\right)^2\right)}{\sqrt{2\pi}\sigma}

                FWHM &= s\sqrt{2\ln\left(2\right)}\sigma"""
        x = x - self.mu
        expPart = np.exp(-0.5 * (x / self.sigma) ** 2)
        normPart = self.sigma * sqrt2pi
        return super(Gaussian, self).__call__(expPart / normPart)


class Lorentzian(Profile):

    """A callable normalized Lorentzian profile."""

    def __init__(self, fwhm=None, mu=None, amp=None, ampIsArea=False):
        """Creates a callable object storing the fwhm, amplitude and location
        of a Lorentzian lineshape.

        Parameters
        ----------
        fwhm: float
            Full Width At Half Maximum, defaults to 1.
        mu: float
            Location of the center, defaults to 0.
        amp: float
            Amplitude of the profile, defaults to 1.
        ampIsArea: boolean
            Sets if the amplitude is the integral or the peakheight. Defaults
            to False.

        Returns
        -------
        Lorentzian
            Callable instance, evaluates the Lorentzian profile in the arguments
            supplied."""
        super(Lorentzian, self).__init__(fwhm=fwhm, mu=mu,
                                         amp=amp, ampIsArea=ampIsArea)

    @property
    def fwhm(self):
        """FWHM of the peak."""
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.gamma = 0.5 * self.fwhm
        if not self.ampIsArea:
            self._normFactor = 1.0 / (self.gamma * np.pi)

    def __call__(self, x):
        r"""Evaluates the lineshape in the given values.

        Parameters
        ----------
        vals: array_like
            Array of values to evaluate the lineshape in.

        Returns
        -------
        array_like
            Array of seperate response values of the lineshape.

        Note
        ----
        The formula used is taken from the MathWorld webpage
        http://mathworld.wolfram.com/LorentzianFunction.html:

            .. math::
                \mathcal{L}\left(x; \mu, \gamma\right) &= \frac{\gamma}
                {\pi\left(\left(x-\mu\right)^2+\gamma^2\right)}

                FWHM &= 2\gamma"""
        x = x - self.mu
        topPart = self.gamma
        bottomPart = (x ** 2 + self.gamma ** 2) * np.pi
        return super(Lorentzian, self).__call__(topPart / bottomPart)


class Voigt(Profile):

    r"""A callable normalized Voigt profile."""

    def __init__(self, fwhm=None, mu=None, amp=None, ampIsArea=False):
        """Creates a callable object storing the fwhm, amplitude and location
        of a Voigt lineshape.

        Parameters
        ----------
        fwhm: float
            Full Width At Half Maximum, defaults to 1.
        mu: float
            Location of the center, defaults to 0.
        amp: float
            Amplitude of the profile, defaults to 1.
        ampIsArea: boolean
            Sets if the amplitude is the integral or the peakheight. Defaults
            to False.

        Returns
        -------
        Voigt
            Callable instance, evaluates the Voigt profile in the arguments supplied."""
        self._fwhmNorm = np.array([sqrt2log2t2, 2])
        super(Voigt, self).__init__(fwhm=fwhm, mu=mu,
                                    amp=amp, ampIsArea=ampIsArea)

    @property
    def fwhm(self):
        """FWHM of the peak."""
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
        r"""Evaluates the lineshape in the given values.

        Parameters
        ----------
        vals: array_like
            Array of values to evaluate the lineshape in.

        Returns
        -------
        array_like
            Array of seperate response values of the lineshape.

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
        x = x - self.mu
        z = (x + 1j * self.gamma) / (self.sigma * sqrt2)
        top = wofz(z).real / (self.sigma * sqrt2pi)
        return super(Voigt, self).__call__(top)

class Crystalball(Profile):

    r"""A callable Crystalball profile."""

    def __init__(self, fwhm=None, mu=None, amp=None, alpha=None, n=None, ampIsArea=False):
        """Creates a callable object storing the fwhm, amplitude and location
        of a Crystalball lineshape.

        Parameters
        ----------
        fwhm: float
            Full Width At Half Maximum, defaults to 1.
        mu: float
            Location of the center, defaults to 0.
        amp: float
            Amplitude of the profile, defaults to 1.
        alpha: float
            Location of the tail.
        n: float
            Relative amplitude of the tail.
        ampIsArea: boolean
            Sets if the amplitude is the integral or the peakheight. Defaults
            to False.

        Returns
        -------
        Crystalball
            Callable instance, evaluates the Crystalball profile in the arguments supplied."""
        super(Crystalball, self).__init__(fwhm=fwhm, mu=mu,
                                          amp=amp, ampIsArea=ampIsArea)

    @property
    def mu(self):
        return self._mu
    
    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def fwhm(self):
        """FWHM of the peak."""
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value
        self.sigma = self.fwhm / (sqrt2log2t2)

        if not self.ampIsArea:
            self._normFactor = 1

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value

    def _bigger(self, x):
        ret = np.exp(-0.5 * x * x / (self.sigma * self.sigma))
        return ret

    def _smaller(self, x):
        a = np.abs(self.alpha)
        n = self.n
        b = n / a - a
        a = ((n / a)**n) * np.exp(-0.5*a*a)
        x = x / self.sigma
        return a / (b - x) ** n

    def __call__(self, x):
        r"""Evaluates the lineshape in the given values.

        Parameters
        ----------
        vals: array_like
            Array of values to evaluate the lineshape in.

        Returns
        -------
        array_like
            Array of seperate response values of the lineshape."""
        x = (x - self.mu) * np.sign(self.alpha)
        y = np.piecewise(x, x >= -np.abs(self.alpha), [self._bigger, self._smaller])
        return super(Crystalball, self).__call__(y)
