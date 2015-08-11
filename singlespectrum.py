"""
.. module:: SingleSpectrum
    :platform: Windows
    :synopsis: Implementation of classes for the analysis of hyperfine
     structure spectra, including simultaneous fitting, various fitting
     routines and isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import abc
import emcee as mcmc
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import satlas.loglikelihood as llh
import satlas.profiles as p
import satlas.utilities as utils
from satlas.wigner import wigner_6j as W6J
from satlas.spectrum import Spectrum

class SingleSpectrum(Spectrum):

    r"""Class for the construction of a HFS spectrum, consisting of different
    peaks described by a certain profile. The number of peaks and their
    positions is governed by the atomic HFS.
    Calling an instance of the Spectrum class returns the response value of the
    HFS spectrum for that frequency in MHz.

    Parameters
    ----------
    I: float
        The nuclear spin.
    J: list of 2 floats
        The spins of the fine structure levels.
    ABC: list of 6 floats
        The hyperfine structure constants A, B and C for ground- and excited
        fine level. The list should be given as [A :sub:`lower`,
        A :sub:`upper`, B :sub:`lower`, B :sub:`upper`, C :sub:`upper`,
        C :sub:`lower`].
    df: float
        Center of Gravity of the spectrum.
    fwhm: float or list of 2 floats, optional
        Depending on the used shape, the FWHM is defined by one or two floats.
        Defaults to [50.0, 50.0]
    scale: float, optional
        Sets the strength of the spectrum, defaults to 1.0. Comparable to the
        amplitude of the spectrum.

    Other parameters
    ----------------
    shape : string, optional
        Sets the transition shape. String is converted to lowercase. For
        possible values, see :attr:`Spectrum.__shapes__.keys()`.
        Defaults to Voigt if an incorrect value is supplied.
    racah_int: Boolean, optional
        If True, fixes the relative peak intensities to the Racah intensities.
        Otherwise, gives them equal intensities and allows them to vary during
        fitting.
    shared_fwhm: Boolean, optional
        If True, the same FWHM is used for all peaks. Otherwise, give them all
        the same initial FWHM and let them vary during the fitting.

    Attributes
    ----------
    fwhm : (list of) float or list of 2 floats
        Sets the FWHM for all the transtions. If :attr:`shared_fwhm` is True,
        this attribute is a list of FWHM values for each peak.
    relAmp : list of floats
        Sets the relative intensities of the transitions.
    scale : float
        Sets the amplitude of the global spectrum.
    background : float
        Sets the background of the global spectrum.
    ABC : list of 6 floats
        List of the hyperfine structure constants, organised as
        [A :sub:`lower`, A :sub:`upper`, B :sub:`lower`, B :sub:`upper`,
        C :sub:`upper`, C :sub:`lower`].
    n : integer
        Sets the number of Poisson sidepeaks.
    offset : float
        Sets the offset for the Poisson sidepeaks.
        The sidepeaks are located at :math:`i\cdot \text{offset}`,
        with :math:`i` the number of the sidepeak.
        Note: this means that a negative value indicates a sidepeak
        to the left of the main peak.
    poisson : float
        Sets the Poisson-factor for the Poisson sidepeaks.
        The amplitude of each sidepeak is multiplied by
        :math:`\text{poisson}^i/i!`, with :math:`i` the number of the sidepeak.

    Note
    ----
    The listed attributes are commonly accessed attributes for the end user.
    More are used, and should be looked up in the source code."""

    __shapes__ = {'gaussian': p.Gaussian,
                  'lorentzian': p.Lorentzian,
                  'irrational': p.Irrational,
                  'hyperbolic': p.HyperbolicSquared,
                  'extendedvoigt': p.ExtendedVoigt,
                  'pseudovoigt': p.PseudoVoigt,
                  'voigt': p.Voigt}

    def __init__(self, I, J, ABC, df, fwhm=[50.0, 50.0], scale=1.0,
                 background=0.1, shape='voigt', racah_int=True,
                 shared_fwhm=True):
        super(SingleSpectrum, self).__init__()
        shape = shape.lower()
        if shape not in self.__shapes__:
            print("""Given profile shape not yet supported.
            Defaulting to Voigt lineshape.""")
            shape = 'voigt'
            fwhm = [50.0, 50.0]

        self.shape = shape
        self._relAmp = []
        self._racah_int = racah_int
        self.shared_fwhm = shared_fwhm
        self.parts = []
        self._I = I
        self._J = J
        self._ABC = ABC
        self.abc_limit = 30000.0
        self.fwhm_limit = 0.1
        self._df = df

        self.scale = scale if racah_int else 1.0
        self._background = background

        self._energies = []
        self._mu = []

        self.n = 0
        self.poisson = 0.609
        self.offset = 0

        self._vary = {}
        self.ratio = [None, None, None]

        self.calculateLevels()
        self.relAmp = [f * scale for f in self.relAmp]
        self.calculateLevels()
        self.fwhm = fwhm

    def set_variation(self, varyDict):
        """Sets the variation of the fitparameters as supplied in the
        dictionary.

        Parameters
        ----------
        varydict: dictionary
            A dictionary containing 'key: True/False' mappings

        Note
        ----
        The list of usable keys:

        * :attr:`FWHM` (only for profiles with one float for the FWHM)
        * :attr:`eta`  (only for the Pseudovoigt profile)
        * :attr:`FWHMG` (only for profiles with two floats for the FWHM)
        * :attr:`FWHML` (only for profiles with two floats for the FWHM)
        * :attr:`Al`
        * :attr:`Au`
        * :attr:`Bl`
        * :attr:`Bu`
        * :attr:`Cl`
        * :attr:`Cu`
        * :attr:`df`
        * :attr:`Background`
        * :attr:`Poisson` (only if the attribute *n* is greater than 0)
        * :attr:`Offset` (only if the attribute *n* is greater than 0)"""
        for k in varyDict.keys():
            self._vary[k] = varyDict[k]

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, value):
        self._I = value
        self.calculateLevels()

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, value):
        self._J = value
        self.calculateLevels()

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, value):
        self._background = value

    @property
    def ABC(self):
        return self._ABC

    @ABC.setter
    def ABC(self, value):
        self._ABC = value
        self._calculate_transitions()

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        self._calculate_transitions()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = np.abs(value)

    @property
    def racah_int(self):
        return self._racah_int

    @racah_int.setter
    def racah_int(self, value):
        self._racah_int = value
        self._calculate_intensities()

    @property
    def relAmp(self):
        return self._relAmp

    @relAmp.setter
    def relAmp(self, value):
        if len(value) is len(self.parts):
            value = np.array(value, dtype='float')
            self._relAmp = np.abs(value)
            for prof, val in zip(self.parts, value):
                prof.amp = np.abs(val)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        if self.shared_fwhm:
            self._fwhm = value
            for prof in self.parts:
                prof.fwhm = value
        else:
            if (self.shape in ['extendedvoigt', 'voigt']
                and all(map(lambda x: isinstance(x, float), value))
                and 2 == len(self.parts)) or (not len(value) ==
                                              len(self.parts)):
                self._fwhm = [value for _ in range(len(self.parts))]
            else:
                self._fwhm = value
            for prof, v in zip(self.parts, self.fwhm):
                prof.fwhm = v

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if len(value) is len(self.parts):
            self._mu = value
            for prof, val in zip(self.parts, value):
                prof.mu = val

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = int(value)

    @property
    def poisson(self):
        return self._poisson

    @poisson.setter
    def poisson(self, value):
        self._poisson = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    def calculateLevels(self):
        self._F = [np.arange(abs(self._I - self._J[0]),
                             self._I + self._J[0] + 1, 1),
                   np.arange(abs(self._I - self._J[1]),
                             self._I + self._J[1] + 1, 1)]

        self._calculate_transitions()
        self._calculate_intensities()

    def _calculate_transitions(self):
        self._energies = [[self.calculate_F_level_energy(0, F)
                           for F in self._F[0]],
                          [self.calculate_F_level_energy(1, F)
                           for F in self._F[1]]]

        mu = []
        for i, F1 in enumerate(self._F[0]):
            for j, F2 in enumerate(self._F[1]):
                if abs(F2 - F1) <= 1 and not F2 == F1 == 0.0:
                    mu.append(self._energies[1][j] - self._energies[0][i])

        if not len(self.parts) is len(mu):
            self.parts = tuple(
                self.__shapes__[self.shape]() for _ in range(len(mu)))
        self.mu = mu

    def _calculate_intensities(self):
        ampl = []
        if self.I == 0:
            ampl = [1.0]
        else:
            for i, F1 in enumerate(self._F[0]):
                for j, F2 in enumerate(self._F[1]):
                    a = self.calculate_racah_intensity(self._J[0],
                                                       self._J[1],
                                                       F1,
                                                       F2)
                    if a != 0.0:
                        ampl.append(a)
        self.relAmp = ampl

    def calculate_racah_intensity(self, J1, J2, F1, F2, order=1.0):
        return (2 * F1 + 1) * (2 * F2 + 1) * \
            W6J(J2, F2, self._I, F1, J1, order) ** 2

    def calculate_F_level_energy(self, level, F):
        r"""The hyperfine addition to a central frequency (attribute :attr:`df`)
        for a specific level is calculated. The formula comes from
        :cite:`Schwartz1955` and in a simplified form, reads

        .. math::
            C_F &= F(F+1) - I(I+1) - J(J+1)

            D_F &= \frac{3 C_F (C_F + 1) - 4 I (I + 1) J (J + 1)}{2 I (2 I - 1)
            J (2 J - 1)}

            E_F &= \frac{10 (\frac{C_F}{2})^3 + 20(\frac{C_F}{2})^2 + C_F(-3I(I
            + 1)J(J + 1) + I(I + 1) + J(J + 1) + 3) - 5I(I + 1)J(J + 1)}{I(I -
            1)(2I - 1)J(J - 1)(2J - 1)}

            E &= df + \frac{A C_F}{2} + \frac{B D_F}{4} + C E_F

        A, B and C are the dipole, quadrupole and octupole hyperfine
        parameters. Octupole contributions are calculated when both the
        nuclear and electronic spin is greater than 1, quadrupole contributions
        when they are greater than 1/2, and dipole contributions when they are
        greater than 0.

        Parameters
        ----------
        level: int, 0 or 1
            Integer referring to the lower (0) level, or the upper (1) level.
        F: integer or half-integer
            F-quantum number for which the hyperfine-corrected energy has to be
            calculated.

        Returns
        -------
        energy: float
            Energy in MHz."""
        I = self._I
        J = self._J[level]
        A = self._ABC[level]
        B = self._ABC[level + 2]
        C = self._ABC[level + 4]

        if level == 0:
            df = 0
        else:
            df = self._df

        if (I == 0 or J == 0):
            C_F = 0
            D_F = 0
            E_F = 0
        elif (I == .5 or J == .5):
            C_F = F * (F + 1) - I * (I + 1) - J * (J + 1)
            D_F = 0
            E_F = 0
        elif (I == 1. or J == 1.):
            C_F = F * (F + 1) - I * (I + 1) - J * (J + 1)
            D_F = (3 * C_F * (C_F + 1) -
                   4 * I * (I + 1) * J * (J + 1)) /\
                  (2 * I * (2 * I - 1) * J * (2 * J - 1))
            E_F = 0
        else:
            C_F = F * (F + 1) - I * (I + 1) - J * (J + 1)
            D_F = (3 * C_F * (C_F + 1) -
                   4 * I * (I + 1) * J * (J + 1)) /\
                  (2 * I * (2 * I - 1) * J * (2 * J - 1))
            E_F = (10 * (0.5 * C_F) ** 3 + 20 * (0.5 * C_F) ** 2
                   + C_F * (-3 * I * (I + 1) * J * (J + 1) +
                            I * (I + 1) + J * (J + 1) + 3) -
                   5 * I * (I + 1) * J * (J + 1)) /\
                  (I * (I - 1) * (2 * I - 1) * J * (J - 1) * (2 * J - 1))

        return df + 0.5 * A * C_F + 0.25 * B * D_F + C * E_F

    def sanitize_input(self, x, y, yerr=None):
        return x, y, yerr

    def var_from_params(self, params):
        """Given a Parameters instance 'params', the value-fields for all the
        parameters are extracted and used to set the values of the spectrum.
        Will raise a KeyError exception if an unsuitable instance is
        supplied.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters to set all values."""
        if self.shape not in ['extendedvoigt', 'voigt']:
            if self.shared_fwhm:
                self.fwhm = params['FWHM'].value
            else:
                self.fwhm = [params['FWHM'+str(i)].value
                             for i in range(len(self.parts))]
            if self.shape in ['pseudovoigt']:
                for part in self.parts:
                    part.n = params['eta'].value
        else:
            if self.shared_fwhm:
                self.fwhm = [params['FWHMG'].value, params['FWHML'].value]
            else:
                self.fwhm = [[params['FWHMG' + str(i)].value,
                              params['FWHML' + str(i)].value]
                             for i in range(len(self.parts))]

        self.scale = params['scale'].value
        self.relAmp = [params['Amp' + str(i)].value
                       for i in range(len(self.parts))]

        self.ABC = [params['Al'].value, params['Au'].value,
                    params['Bl'].value, params['Bu'].value,
                    params['Cl'].value, params['Cu'].value]

        self.df = params['df'].value

        self.background = params['Background'].value
        self.n = params['N'].value
        if self.n > 0:
            self.Poisson = params['Poisson'].value
            self.Offset = params['Offset'].value

    def params_from_var(self):
        """Goes through all the relevant parameters of the spectrum,
        and returns a Parameters instance containing all the information. User-
        supplied information in the self._vary dictionary is used to set
        the variation of parameters during the fitting, while
        making sure that the A, B and C parameters are not used if the spins
        do not allow it.

        Returns
        -------
        Parameters
            Instance suitable for the method :meth:`var_from_params`."""
        par = lm.Parameters()
        if self.shape not in ['extendedvoigt', 'voigt']:
            if self.shared_fwhm:
                par.add('FWHM', value=self.fwhm, vary=True,
                        min=self.fwhm_limit)
            else:
                for i, val in enumerate(self.fwhm):
                    par.add('FWHM' + str(i), value=val, vary=True,
                            min=self.fwhm_limit)
            if self.shape in ['pseudovoigt']:
                par.add('eta', value=self.parts[0].n, vary=True, min=0, max=1)
        else:
            if self.shared_fwhm:
                par.add('FWHMG', value=self.fwhm[0], vary=True,
                        min=self.fwhm_limit)
                par.add('FWHML', value=self.fwhm[1], vary=True,
                        min=self.fwhm_limit)
                val = 0.5346 * self.fwhm[1] + np.sqrt(0.2166 *
                                                      self.fwhm[1] ** 2
                                                      + self.fwhm[0] ** 2)
                par.add('TotalFWHM', value=val, vary=False,
                        expr='0.5346*FWHML+sqrt(0.2166*FWHML**2+FWHMG**2)')
            else:
                for i, val in enumerate(self.fwhm):
                    par.add('FWHMG' + str(i), value=val[0], vary=True,
                            min=self.fwhm_limit)
                    par.add('FWHML' + str(i), value=val[1], vary=True,
                            min=self.fwhm_limit)
                    val = 0.5346 * val[1] + np.sqrt(0.2166 * val[1] ** 2
                                                    + val[0] ** 2)
                    par.add('TotalFWHM' + str(i), value=val, vary=False,
                            expr='0.5346*FWHML' + str(i) +
                                 '+sqrt(0.2166*FWHML' + str(i) +
                                 '**2+FWHMG' + str(i) + '**2)')

        par.add('scale', value=self.scale, vary=self.racah_int, min=0)
        for i, prof in enumerate(self.parts):
            par.add('Amp' + str(i), value=self._relAmp[i],
                    vary=not self.racah_int, min=0)

        b = (None, None) if self.abc_limit is None else (-self.abc_limit,
                                                         self.abc_limit)
        par.add('Al', value=self._ABC[0], vary=True, min=b[0], max=b[1])
        par.add('Au', value=self._ABC[1], vary=True, min=b[0], max=b[1])
        par.add('Bl', value=self._ABC[2], vary=True, min=b[0], max=b[1])
        par.add('Bu', value=self._ABC[3], vary=True, min=b[0], max=b[1])
        par.add('Cl', value=self._ABC[4], vary=True, min=b[0], max=b[1])
        par.add('Cu', value=self._ABC[5], vary=True, min=b[0], max=b[1])

        if self.ratio[0] is not None:
            par['Au'].expr = str(self.ratio[0]) + '*Al'
            par['Au'].vary = False
        if self.ratio[1] is not None:
            par['Bu'].expr = str(self.ratio[1]) + '*Bl'
            par['Bu'].vary = False
        if self.ratio[2] is not None:
            par['Cu'].expr = str(self.ratio[2]) + '*Cl'
            par['Cu'].vary = False

        par.add('df', value=self._df, vary=True)

        par.add('Background', value=self.background, vary=True, min=0)
        par.add('N', value=self._n, vary=False)
        if self._n > 0:
            par.add('Poisson', value=self._poisson, vary=True, min=0)
            par.add('Offset', value=self._offset, vary=True, min=None, max=-0.01)
        for key in self._vary.keys():
            if key in par.keys():
                par[key].vary = self._vary[key]
        par['N'].vary = False
        if self._I == 0.0:
            par['Al'].vary = False
            par['Al'].value = 0
            par['Au'].vary = False
            par['Au'].value = 0
        if self._I <= 0.5:
            par['Bl'].vary = False
            par['Bl'].value = 0
            par['Bu'].vary = False
            par['Bu'].value = 0
        if self._J[0] <= 0.5:
            par['Bl'].vary = False
            par['Bl'].value = 0
        if self._J[1] <= 0.5:
            par['Bu'].vary = False
            par['Bu'].value = 0
        if self._I <= 1.0:
            par['Cl'].vary = False
            par['Cl'].value = 0
            par['Cu'].vary = False
            par['Cu'].value = 0
        if self._J[0] <= 1.0:
            par['Cl'].vary = False
            par['Cl'].value = 0
        if self._J[1] <= 1.0:
            par['Cu'].vary = False
            par['Cu'].value = 0
        return par

    def bootstrap(self, x, y, bootstraps=100, samples=None, selected=True):
        """Given an experimental spectrum of counts, generate a number of
        bootstrapped resampled spectra, fit these, and return a pandas
        DataFrame containing result of fitting these resampled spectra.

        Parameters
        ----------
        x: array_like
            Frequency in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.

        Other Parameters
        ----------------
        bootstraps: integer, optional
            Number of bootstrap samples to generate, defaults to 100.
        samples: integer, optional
            Number of counts in each bootstrapped spectrum, defaults to
            the number of counts in the supplied spectrum.
        selected: boolean, optional
            Selects if only the parameters in :attr:`self.selected` are saved
            in the DataFrame. Defaults to True (saving only the selected).

        Returns
        -------
        DataFrame
            DataFrame containing the results of fitting the bootstrapped
            samples."""
        total = np.cumsum(y)
        dist = total / float(y.sum())
        names, var, varerr = self.vars(selection='chisquare')
        selected = self.selected if selected else names
        v = [name for name in names if name in selected]
        data = pd.DataFrame(index=np.arange(0, bootstraps + 1),
                            columns=v)
        stderrs = pd.DataFrame(index=np.arange(0, bootstraps + 1),
                               columns=v)
        v = [var[i] for i, name in enumerate(names) if name in selected]
        data.loc[0] = v
        v = [varerr[i] for i, name in enumerate(names) if name in selected]
        stderrs.loc[0] = v
        if samples is None:
            samples = y.sum()
        length = len(x)

        for i in range(bootstraps):
            newy = np.bincount(
                    np.searchsorted(
                            dist,
                            np.random.rand(samples)
                            ),
                    minlength=length
                    )
            self.chisquare_spectroscopic_fit(x, newy)
            names, var, varerr = self.vars(selection='chisquare')
            v = [var[i] for i, name in enumerate(names) if name in selected]
            data.loc[i + 1] = v
            v = [varerr[i] for i, name in enumerate(names) if name in selected]
            stderrs.loc[i + 1] = v
        pan = {'data': data, 'stderr': stderrs}
        pan = pd.Panel(pan)
        return pan

    def __add__(self, other):
        """Add two spectra together to get an :class:`IsomerSpectrum`.

        Parameters
        ----------
        other: Spectrum
            Other spectrum to add.

        Returns
        -------
        IsomerSpectrum
            An Isomerspectrum combining both spectra."""
        if isinstance(other, SingleSpectrum):
            l = [self, other]
        from satlas.isomerspectrum import IsomerSpectrum
        return IsomerSpectrum(l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def seperate_response(self, x):
        """Get the response for each seperate spectrum for the values :attr:`x`
        , without background.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz.

        Returns
        -------
        list of floats or NumPy arrays
            Seperate responses of spectra to the input :attr:`x`."""
        return [self(x)]

    def __call__(self, x):
        """Get the response for frequency :attr:`x` (in MHz) of the spectrum.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz

        Returns
        -------
        float or NumPy array
            Response of the spectrum for each value of :attr:`x`."""
        if self._n > 0:
            s = np.zeros(x.shape)
            for i in range(self._n + 1):
                s += (self.poisson ** i) * sum([prof(x + i * self.offset)
                                                for prof in self.parts]) \
                    / np.math.factorial(i)
            s = s * self.scale
        else:
            s = self.scale * sum([prof(x) for prof in self.parts])
        return s + self.background

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None,
             no_of_points=10**4, ax=None, show=True, label=True,
             legend=None, data_legend=None):
        """Routine that plots the hfs, possibly on top of experimental data.
        Parameters
        ----------
        x: array
            Experimental x-data. If None, a suitable region around
            the peaks is chosen to plot the hfs.
        y: array
            Experimental y-data.
        yerr: array
            Experimental errors on y.
        no_of_points: int
            Number of points to use for the plot of the hfs.
        ax: matplotlib axes object
            If provided, plots on this axis
        show: Boolean
            If True, the plot will be shown at the end.
        label: Boolean
            If True, the plot will be labeled.
        legend: String, optional
            If given, an entry in the legend will be made for the spectrum.
        data_legend: String, optional
            If given, an entry in the legend will be made for the experimental
            data.
        Returns
        -------
        None"""

        if ax is None:
            fig, ax = plt.subplots(1, 1)
            toReturn = fig, ax
        else:
            toReturn = None

        if x is None:
            ranges = []

            ## Hack alert!!!!
            if type(self.fwhm) == list:
                fwhm = np.sqrt(self.fwhm[0]**2 + self.fwhm[0]**2)
            else:
                fwhm = self.fwhm
            ## end of hack

            for pos in self.mu:
                r = np.linspace(pos - 4 * fwhm,
                                pos + 4 * fwhm,
                                2 * 10**2)
                ranges.append(r)
            superx = np.sort(np.concatenate(ranges))

        else:
            superx = np.linspace(x.min(), x.max(), no_of_points)

        if x is not None and y is not None:
            ax.errorbar(x, y, yerr, fmt='o', markersize=5, label=data_legend)
        ax.plot(superx, self(superx), lw=3.0, label=legend)
        if label:
            ax.set_xlabel('Frequency (MHz)', fontsize=16)
            ax.set_ylabel('Counts', fontsize=16)
        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, **kwargs):
        """Routine that plots the hfs, possibly on top of
        experimental data. It assumes that the y data is drawn from
        a Poisson distribution (e.g. counting data).
        Parameters
        ----------
        x: array
            Experimental x-data. If None, a suitable region around
            the peaks is chosen to plot the hfs.
        y: array
            Experimental y-data.
        no_of_points: int
            Number of points to use for the plot of the hfs.
        ax: matplotlib axes object
            If provided, plots on this axis
        show: Boolean
            if True, the plot will be shown at the end.
        Returns
        -------
        None
        """
        y = kwargs.get('y', None)
        if y is not None:
            yerr = np.sqrt(y + 1)
        else:
            yerr = None
        kwargs['yerr'] = yerr
        self.plot(**kwargs)