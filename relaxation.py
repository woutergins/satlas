"""
.. module:: relaxation
    :platform: Windows
    :synopsis: Implementation of classes for easy and
     intuitive spin-lattice relaxation simulations.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Hanne Heylen <hanne.heylen@fys.kuleuven.be>
"""
import numpy as np
from scipy.integrate import odeint

__all__ = ['KorringaRelaxation']


class KorringaRelaxation(object):

    r"""Class to evaluate the Korringa relaxation of a system.

    Parameters
    ----------
    I : integer or half-integer
        Nuclear spin.
    t1_2 : float
        Half-life of the nucleus in seconds.
    Tl : float
        Lattice temperature in Kelvin.
    Tint : float
        Interaction temperature for equidistant splitting, in Kelvin.
    T1 : float
        Spin-lattice relaxation rate in seconds.

    Other parameters
    ----------------
    implant : callable, optional
        If implantation has to be taken into account, give a callable defining
        the number of implanted nuclei as a function of time.

    Raises
    ------
    TypeError
        When :attr:`implant` is not a callable.

    Attributes
    ----------
    Ck : float
        Calculated Korringa constant using

        .. math::
            T_1=\frac{2 C_k}{T_{int}}\tanh\left(\frac{T_{int}}{2T_l}\right)

        If the relaxation time :math:`T_1`, the interaction temperature
        :math:`T_{int}` or the lattice temperature :math:`T_l` are changed,
        :math:`C_k` is recalculated.
    source: array_like
        Distribution of the implanted nuclei in the different :math:`m_I`
        states. Defaults to equal populations in all states.
    initial: array_like
        Initial distribution in the :math:`m_I` states."""

    def __init__(self, I, t1_2, Tl, Tint, T1, implant=None):
        super(KorringaRelaxation, self).__init__()
        self.t1_2 = t1_2
        self.I = I
        self.lambd = np.log(2) / self.t1_2
        self._Tl = Tl
        self._Tint = Tint
        self.T1 = T1

        if implant is None:
            self.implant = lambda x: 0
        else:
            self.implant = implant
        self.source = np.array([x == 2 * I for x in np.arange(2 * I + 1)],
                               dtype='float')
        self.initial = np.ones(self.source.shape) / len(self.source)

    @property
    def Tint(self):
        return self._Tint

    @Tint.setter
    def Tint(self, value):
        self._Tint = value
        self.Ck = self.calculateCk()
        self.W = self.TransitionMatrix()

    @property
    def Tl(self):
        return self._Tl

    @Tl.setter
    def Tl(self, value):
        self._Tl = value
        self.Ck = self.calculateCk()
        self.W = self.TransitionMatrix()

    @property
    def T1(self):
        return self._T1

    @T1.setter
    def T1(self, value):
        self._T1 = value
        self.Ck = self.calculateCk()
        self.W = self.TransitionMatrix()

    @property
    def implant(self):
        return self._implant

    @implant.setter
    def implant(self, value):
        if callable(value):
            self._implant = value
        else:
            m = 'implant is not a callable, but {}!'.format(type(value))
            raise TypeError(m)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if len(value) == 2 * self.I + 1:
            self._source = value / sum(value)

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, value):
        if len(value) == 2 * self.I + 1:
            self._initial = value

    def calculateCk(self):
        return self.T1 * self._Tint / (2 * np.tanh(self._Tint / (2 * self.Tl)))

    def TransitionMatrix(self):
        # Creates the needed transition matrix for spin-lattice relaxation
        I = self.I
        Tl = self.Tl
        Ck = self.Ck

        if hasattr(self, 'Tint'):
            Tint = self.Tint
            Wup = [Tint * (I * (I + 1) - m * (m + 1)) /
                   (2 * Ck * (1 - np.exp(-Tint / Tl)))
                   for m in np.arange(-I, I)]
            Wdown = [Tint * (I * (I + 1) - m * (m + 1)) /
                     (2 * Ck * (np.exp(Tint / Tl) - 1))
                     for m in np.arange(-I, I)]
        else:
            dE = self.dE
            kB = self.kB
            Wup = [dE[i] * (I * (I + 1) - m * (m + 1)) /
                   (2 * kB * Ck * (1 - np.exp(-dE[i] / Tl / kB)))
                   for i, m in enumerate(np.arange(-I, I))]
            Wdown = [dE[i] * (I * (I + 1) - m * (m + 1)) /
                     (2 * kB * Ck * (np.exp(dE[i] / Tl / kB) - 1))
                     for i, m in enumerate(np.arange(-I, I))]
        Wup = np.diagflat(Wup, 1)  # Upper-diagonal: W_{m+1, m}
        Wdown = np.diagflat(Wdown, -1)  # Lower-diagonal: W_{m, m+1}
        W = Wup + Wdown
        np.fill_diagonal(W, -W.sum(axis=0))  # Fill the diagonal.
        return W

    def rhs(self, y, t, W, lambd, implant, source):
        if source is None:
            source = np.ones(y.shape)
        return np.dot(W, y) - lambd * y + implant(t) * source

    def Simulate(self, t, pol=True, activity=True):
        """Takes the timevector t and simulates the activity and polarization
        at these times.

        Parameters
        ----------
        t: array_like
            Monotonically increasing sequence of times, needed for input in
            :func:`scipy.integrate.odeint`.

        Other parameters
        ----------------
        pol: boolean, optional
            Boolean to check if the user is interested in the polarization.
        activity: boolean, optional
            Boolean to check if the user is interested in the activity.

        Returns
        -------
        tuple: array_likes
            The first entry is the polarization, second entry is the activity.
            If the polarization or activity is not requested, this vector is
            replaced by the value :class:`None`."""
        args = (self.W, self.lambd, self.implant, self.source)
        result = odeint(self.rhs, self.initial, t, args=args)
        m = np.arange(-self.I, self.I + 1)
        returns = ()
        if pol:
            def pol(x):
                return (x * m).sum() / (self.I * x.sum()) * 100.0
            pol = np.apply_along_axis(pol, 1, result)
            returns += (pol,)
        else:
            returns += (None,)
        if activity:
            def activity(x):
                return x.sum() * self.lambd
            activity = np.apply_along_axis(activity, 1, result)
            returns += (activity,)
        else:
            returns += (None,)
        return returns
