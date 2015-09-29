"""
Implementation of a class for the analysis of hyperfine structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import satlas.profiles as p
import scipy.optimize as optimize
from fractions import Fraction

from .isomerspectrum import IsomerSpectrum
from .spectrum import Spectrum
from .wigner import wigner_6j, wigner_3j
from .utilities import poisson_interval
from .loglikelihood import poisson_llh
W6J = wigner_6j
W3J = wigner_3j

__all__ = ['SingleSpectrum']


class SingleSpectrum(Spectrum):

    r"""Constructs a HFS spectrum, consisting of different
    peaks described by a certain profile. The number of peaks is governed by
    the nuclear spin and the atomic spins of the levels."""

    __shapes__ = {'gaussian': p.Gaussian,
                  'lorentzian': p.Lorentzian,
                  'voigt': p.Voigt}

    def __init__(self, I, J, ABC, centroid, fwhm=[50.0, 50.0], scale=1.0,
                 background=0.1, shape='voigt', racah_int=True, saturation=0,
                 shared_fwhm=True, n=0, poisson=0.68, offset=0):
        """Builds the HFS with the given atomic and nuclear information.

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
        centroid: float
            Centroid of the spectrum.

        Other parameters
        ----------------
        fwhm: float or list of 2 floats, optional
            Depending on the used shape, the FWHM is defined by one or two floats.
            Defaults to [50.0, 50.0]
        scale: float, optional
            Sets the strength of the spectrum, defaults to 1.0. Comparable to the
            amplitude of the spectrum.
        background: float, optional
            Sets the constant background to the supplied value. Defaults to 0.1.
        shape : string, optional
            Sets the transition shape. String is converted to lowercase. For
            possible values, see *Spectrum__shapes__*.keys()`.
            Defaults to Voigt if an incorrect value is supplied.
        racah_int: boolean, optional
            If True, fixes the relative peak intensities to the Racah intensities.
            Otherwise, gives them equal intensities and allows them to vary during
            fitting.
        saturation: float, optional
            If different than 0, calculate the saturation effect on the intensity of
            transition intensity. This is done by an exponential transition between
            Racah intensities and the saturated intensities.
        shared_fwhm: boolean, optional
            If True, the same FWHM is used for all peaks. Otherwise, give them all
            the same initial FWHM and let them vary during the fitting.
        n: int, optional
            Sets the number of sidepeaks that are present in the spectrum.
            Defaults to 0.
        poisson: float, optional
            Sets the relative intensity of the first side peak. The intensity of the
            other sidepeaks is calculated from the Poisson-factor.
        offset: float, optional
            Sets the distance (in MHz) of each sidepeak in the spectrum.

        Note
        ----
        The list of parameter keys is:
            * *FWHM* (only for profiles with one float for the FWHM)
            * *FWHMG* (only for profiles with two floats for the FWHM)
            * *FWHML* (only for profiles with two floats for the FWHM)
            * *Al*
            * *Au*
            * *Bl*
            * *Bu*
            * *Cl*
            * *Cu*
            * *Centroid*
            * *Background*
            * *Poisson* (only if the attribute *n* is greater than 0)
            * *Offset* (only if the attribute *n* is greater than 0)
            * *Amp* (with the correct labeling of the transition)
            * *scale*"""
        super(SingleSpectrum, self).__init__()
        shape = shape.lower()
        if shape not in self.__shapes__:
            print("""Given profile shape not yet supported.
            Defaulting to Voigt lineshape.""")
            shape = 'voigt'
            fwhm = [50.0, 50.0]

        self.I_value = {0.0: ((False, 0), (False, 0), (False, 0),
                              (False, 0), (False, 0), (False, 0)),
                        0.5: ((True, 1), (True, 1),
                              (False, 0), (False, 0), (False, 0), (False, 0)),
                        1.0: ((True, 1), (True, 1),
                              (True, 1), (True, 1),
                              (False, 0), (False, 0))
                        }
        self.J_lower_value = {0.0: ((False, 0), (False, 0), (False, 0)),
                              0.5: ((True, 1),
                                    (False, 0), (False, 0)),
                              1.0: ((True, 1),
                                    (True, 1), (False, 0))
                              }
        self.J_upper_value = {0.0: ((False, 0), (False, 0), (False, 0)),
                              0.5: ((True, 1),
                                    (False, 0), (False, 0)),
                              1.0: ((True, 1),
                                    (True, 1), (False, 0))
                              }
        self.shape = shape
        self._racah_int = racah_int
        self.shared_fwhm = shared_fwhm
        self.I = I
        self.J = J
        self._calculate_F_levels()
        self._calculate_energy_coefficients()
        self._calculate_transitions()

        self._vary = {}
        self._constraints = {}

        self.ratioA = (None, 'lower')
        self.ratioB = (None, 'lower')
        self.ratioC = (None, 'lower')

        self._populate_params(ABC, fwhm, scale, background, n,
                              poisson, offset, centroid, saturation)

    @property
    def locations(self):
        """Contains the locations of the peaks."""
        return self._locations

    @locations.setter
    def locations(self, locations):
        self._locations = locations
        for p, l in zip(self.parts, locations):
            p.mu = l

    @property
    def racah_int(self):
        """Boolean to set the behaviour to Racah intensities (True)
        or to individual amplitudes (False)."""
        return self._racah_int

    @racah_int.setter
    def racah_int(self, value):
        self._racah_int = value
        self.params['Scale'].vary = self._racah_int
        self.params['Saturation'].vary = self._racah_int
        for label in self.ftof:
            self.params['Amp' + label].vary = not self._racah_int

    @property
    def params(self):
        """Instance of lmfit.Parameters object characterizing the
        shape of the HFS."""
        self._params = self._check_variation(self._params)
        return self._params

    @params.setter
    def params(self, params):
        self._params = params
        # When changing the parameters, the energies and
        # the locations have to be recalculated
        self._calculate_energies()
        self._calculate_transition_locations()
        if not self.racah_int:
            # When not using set amplitudes, they need
            # to be changed after every iteration
            self._set_amplitudes()
        else:
            self._set_transitional_amplitudes()
        # Finally, the fwhm of each peak needs to be set
        self._set_fwhm()

    def _set_transitional_amplitudes(self):
        values = self._calculate_transitional_intensities(self._params['Saturation'].value)
        for p, l, v in zip(self.parts, self.ftof, values):
            self._params['Amp' + l].value = v
            p.amp = v

    @property
    def ftof(self):
        """List of transition labels, of the form *Flow__Fhigh* (half-integers
        have an underscore instead of a division sign), same ordering
        as given by the attribute :attr:`.locations`."""
        return self._ftof

    @ftof.setter
    def ftof(self, value):
        self._ftof = value

    def _calculate_energies(self):
        r"""The hyperfine addition to a central frequency (attribute *centroid*)
        for a specific level is calculated. The formula comes from
        :cite:`Schwartz1955` and in a simplified form, reads

        .. math::
            C_F &= F(F+1) - I(I+1) - J(J+1)

            D_F &= \frac{3 C_F (C_F + 1) - 4 I (I + 1) J (J + 1)}{2 I (2 I - 1)
            J (2 J - 1)}

            E_F &= \frac{10 (\frac{C_F}{2})^3 + 20(\frac{C_F}{2})^2 + C_F(-3I(I
            + 1)J(J + 1) + I(I + 1) + J(J + 1) + 3) - 5I(I + 1)J(J + 1)}{I(I -
            1)(2I - 1)J(J - 1)(2J - 1)}

            E &= centroid + \frac{A C_F}{2} + \frac{B D_F}{4} + C E_F

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
        A = np.append(np.ones(self.num_lower) * self.params['Al'].value,
                      np.ones(self.num_upper) * self.params['Au'].value)
        B = np.append(np.ones(self.num_lower) * self.params['Bl'].value,
                      np.ones(self.num_upper) * self.params['Bu'].value)
        C = np.append(np.ones(self.num_lower) * self.params['Cl'].value,
                      np.ones(self.num_upper) * self.params['Cu'].value)
        centr = np.append(np.zeros(self.num_lower),
                          np.ones(self.num_upper) * self.params['Centroid'].value)
        self.energies = centr + self.C * A + self.D * B + self.E * C

    def _calculate_transition_locations(self):
        self.locations = [self.energies[ind_high] - self.energies[ind_low]
                          for (ind_low, ind_high) in self.transition_indices]

    def _set_amplitudes(self):
        for p, label in zip(self.parts, self.ftof):
            p.amp = self._params['Amp' + label].value

    def _set_fwhm(self):
        if self.shape.lower() == 'voigt':
            fwhm = [[self._params['FWHMG'].value, self._params['FWHML'].value] for _ in self.ftof] if self.shared_fwhm else [[self._params['FWHMG' + label].value, self._params['FWHML' + label].value] for label in self.ftof]
        else:
            fwhm = [self._params['FWHM'].value for _ in self.ftof] if self.shared_fwhm else [self._params['FWHM' + label].value for label in self.ftof]
        for p, f in zip(self.parts, fwhm):
            p.fwhm = f

    ####################################
    #      INITIALIZATION METHODS      #
    ####################################

    def _populate_params(self, ABC, fwhm, scale, background,
                        n, poisson, offset, centroid, saturation):
        # Prepares the params attribute with the initial values
        par = lm.Parameters()
        if not self.shape.lower() == 'voigt':
            if self.shared_fwhm:
                par.add('FWHM', value=fwhm, vary=True, min=0)
            else:
                for label, val in zip(self.ftof, fwhm):
                    par.add('FWHM' + label, value=val, vary=True, min=0)
        else:
            if self.shared_fwhm:
                par.add('FWHMG', value=fwhm[0], vary=True, min=0)
                par.add('FWHML', value=fwhm[1], vary=True, min=0)
                val = 0.5346 * fwhm[1] + np.sqrt(0.2166 * fwhm[1] ** 2 + fwhm[0] ** 2)
                par.add('TotalFWHM', value=val, vary=False,
                        expr='0.5346*FWHML+sqrt(0.2166*FWHML**2+FWHMG**2)')
            else:
                for label, val in zip(self.ftof, fwhm):
                    par.add('FWHMG' + label, value=val[0], vary=True, min=0)
                    par.add('FWHML' + label, value=val[1], vary=True, min=0)
                    val = 0.5346 * val[1] + np.sqrt(0.2166 * val[1] ** 2
                                                    + val[0] ** 2)
                    par.add('TotalFWHM' + label, value=val, vary=False,
                            expr='0.5346*FWHML' + label +
                                 '+sqrt(0.2166*FWHML' + label +
                                 '**2+FWHMG' + str(i) + '**2)')

        par.add('Scale', value=scale, vary=self.racah_int, min=0)
        par.add('Saturation', value=saturation, vary=self.racah_int, min=0)
        amps = self._calculate_transitional_intensities(saturation)
        for label, amp in zip(self.ftof, amps):
            label = 'Amp' + label
            par.add(label, value=amp, vary=not self.racah_int, min=0)

        par.add('Al', value=ABC[0], vary=True)
        par.add('Au', value=ABC[1], vary=True)
        par.add('Bl', value=ABC[2], vary=True)
        par.add('Bu', value=ABC[3], vary=True)
        par.add('Cl', value=ABC[4], vary=True)
        par.add('Cu', value=ABC[5], vary=True)

        ratios = (self.ratioA, self.ratioB, self.ratioC)
        labels = (('Al', 'Au'), ('Bl', 'Bu'), ('Cl', 'Cu'))
        for r, (l, u) in zip(ratios, labels):
            if r[0] is not None:
                if r[1].lower() == 'lower':
                    fixed, free = l, u
                else:
                    fixed, free = u, l
                par[fixed].expr = str(r[0]) + '*' + free
                par[fixed].vary = False

        par.add('Centroid', value=centroid, vary=True)

        par.add('Background', value=background, vary=True, min=0)
        par.add('N', value=n, vary=False)
        if n > 0:
            par.add('Poisson', value=poisson, vary=False, min=0)
            par.add('Offset', value=offset, vary=False, min=None, max=0)

        self.params = self._check_variation(par)

    def _set_ratios(self, par):
        # Process the set ratio's for the hyperfine parameters.
        ratios = (self.ratioA, self.ratioB, self.ratioC)
        labels = (('Al', 'Au'), ('Bl', 'Bu'), ('Cl', 'Cu'))
        for r, (l, u) in zip(ratios, labels):
            if r[0] is not None:
                if r[1].lower() == 'lower':
                    fixed, free = l, u
                else:
                    fixed, free = u, l
                par[fixed].expr = str(r[0]) + '*' + free
                par[fixed].vary = False
                par[free].vary = True
        return par

    def _check_variation(self, par):
        # Make sure the variations in the params are set correctly.
        for key in self._vary.keys():
            if key in par.keys():
                par[key].vary = self._vary[key]
        par['N'].vary = False

        if self.I in self.I_value:
            Al, Au, Bl, Bu, Cl, Cu = self.I_value[self.I]
            if not Al[0]:
                par['Al'].vary, par['Al'].value = Al
            if not Au[0]:
                par['Au'].vary, par['Au'].value = Au
            if not Bl[0]:
                par['Bl'].vary, par['Bl'].value = Bl
            if not Bu[0]:
                par['Bu'].vary, par['Bu'].value = Bu
            if not Cl[0]:
                par['Cl'].vary, par['Cl'].value = Cl
            if not Cu[0]:
                par['Cu'].vary, par['Cu'].value = Cu
        if self.J[0] in self.J_lower_value:
            Al, Bl, Cl = self.J_lower_value[self.J[0]]
            if not Al[0]:
                par['Al'].vary, par['Al'].value = Al
            if not Bl[0]:
                par['Bl'].vary, par['Bl'].value = Bl
            if not Cl[0]:
                par['Cl'].vary, par['Cl'].value = Cl
        if self.J[self.num_lower] in self.J_upper_value:
            Au, Bu, Cu = self.J_upper_value[self.J[self.num_lower]]
            if not Au[0]:
                par['Au'].vary, par['Au'].value = Au
            if not Bu[0]:
                par['Bu'].vary, par['Bu'].value = Bu
            if not Cu[0]:
                par['Cu'].vary, par['Cu'].value = Cu

        for key in self._constraints.keys():
            for bound in self._constraints[key]:
                if bound.lower() == 'min':
                    par[key].min = self._constraints[key][bound]
                elif bound.lower() == 'max':
                    par[key].max = self._constraints[key][bound]
                else:
                    pass
        return par

    def _calculate_F_levels(self):
        F1 = np.arange(abs(self.I - self.J[0]), self.I+self.J[0]+1, 1)
        self.num_lower = len(F1)
        F2 = np.arange(abs(self.I - self.J[1]), self.I+self.J[1]+1, 1)
        self.num_upper = len(F2)
        F = np.append(F1, F2)
        self.J = np.append(np.ones(len(F1)) * self.J[0],
                           np.ones(len(F2)) * self.J[1])
        self.F = F

    def _calculate_transitions(self):
        f_f = []
        indices = []
        amps = []
        sat_amp = []
        for i, F1 in enumerate(self.F[:self.num_lower]):
            for j, F2 in enumerate(self.F[self.num_lower:]):
                if abs(F2 - F1) <= 1 and not F2 == F1 == 0.0:
                    sat_amp.append(2*F1+1)
                    j += self.num_lower
                    intensity = self._calculate_racah_intensity(self.J[i],
                                                               self.J[j],
                                                               self.F[i],
                                                               self.F[j])
                    if intensity > 0:
                        amps.append(intensity)
                        indices.append([i, j])
                        s = ''
                        temp = Fraction(F1).limit_denominator()
                        if temp.denominator == 1:
                            s += str(temp.numerator)
                        else:
                            s += str(temp.numerator) + '_' + str(temp.denominator)
                        s += '__'
                        temp = Fraction(F2).limit_denominator()
                        if temp.denominator == 1:
                            s += str(temp.numerator)
                        else:
                            s += str(temp.numerator) + '_' + str(temp.denominator)
                        f_f.append(s)
        self.ftof = f_f  # Stores the labels of all transitions, in order
        self.transition_indices = indices  # Stores the indices in the F and energy arrays for the transition

        self.racah_amplitudes = np.array(amps)  # Sets the initial amplitudes to the Racah intensities
        self.racah_amplitudes = self.racah_amplitudes / self.racah_amplitudes.max()

        self.saturated_amplitudes = np.array(sat_amp)
        self.saturated_amplitudes = self.saturated_amplitudes / self.saturated_amplitudes.max()

        self.parts = tuple(self.__shapes__[self.shape](amp=a) for a in self.racah_amplitudes)

    def _calculate_transitional_intensities(self, s):
        if s <= 0:
            return self.racah_amplitudes
        else:
            sat = self.saturated_amplitudes
            rac = self.racah_amplitudes
            transitional = -sat*np.expm1(-rac * s / sat)
            return transitional / transitional.max()

    def _calculate_racah_intensity(self, J1, J2, F1, F2, order=1.0):
        return float((2 * F1 + 1) * (2 * F2 + 1) * \
                     W6J(J2, F2, self.I, F1, J1, order) ** 2)  # DO NOT REMOVE CAST TO FLOAT!!!

    def _calculate_energy_coefficients(self):
        # Since I, J and F do not change, these factors can be calculated once
        # and then stored.
        I, J, F = self.I, self.J, self.F
        C = (F*(F+1) - I*(I+1) - J*(J + 1)) * (J/J) if I > 0 else 0 * J  #*(J/J) is a dirty trick to avoid checking for J=0
        D = (3*C*(C+1) - 4*I*(I+1)*J*(J+1)) / (2*I*(2*I-1)*J*(2*J-1))
        E = (10*(0.5*C)**3 + 20*(0.5*C)**2 + C*(-3*I*(I+1)*J*(J+1) + I*(I+1) + J*(J+1) + 3) - 5*I*(I+1)*J*(J+1)) / (I*(I-1)*(2*I-1)*J*(J-1)*(2*J-1))
        C = np.where(np.isfinite(C), 0.5 * C, 0)
        D = np.where(np.isfinite(D), 0.25 * D, 0)
        E = np.where(np.isfinite(E), E, 0)
        self.C, self.D, self.E = C, D, E

    ##########################
    #      USER METHODS      #
    ##########################

    def set_variation(self, varyDict):
        """Sets the variation of the fitparameters as supplied in the
        dictionary.

        Parameters
        ----------
        varyDict: dictionary
            A dictionary containing 'key: True/False' mappings"""
        for k in varyDict.keys():
            self._vary[k] = varyDict[k]

    def set_boundaries(self, boundaryDict):
        """Sets the boundaries of the fitparameters as supplied in the
        dictionary.

        Parameters
        ----------
        boundaryDict: dictionary
            A dictionary containing "key: {'min': value, 'max': value}" mappings.
            A value of *None* or a missing key gives no boundary
            in that direction."""
        for k in boundaryDict.keys():
            self._constraints[k] = boundaryDict[k]

    def fix_ratio(self, value, target='upper', parameter='A'):
        """Fixes the ratio for a given hyperfine parameter to the given value.

        Parameters
        ----------
        value: float
            Value to which the ratio is set
        target: {'upper', 'lower'}
            Sets the target level. If 'upper', the upper parameter is
            calculated as lower * ratio, 'lower' calculates the lower
            parameter as upper * ratio.
        parameter: {'A', 'B', 'C'}
            Selects which hyperfine parameter to set the ratio for."""
        if target.lower() not in ['lower', 'upper']:
            raise KeyError("Target must be 'lower' or 'upper'.")
        if parameter.lower() not in ['a', 'b', 'c']:
            raise KeyError("Parameter must be 'A', 'B' or 'C'.")
        if parameter.lower() == 'a':
            self.ratioA = (value, target)
        if parameter.lower() == 'b':
            self.ratioB = (value, target)
        if parameter.lower() == 'c':
            self.ratioC = (value, target)
        self.params = self._set_ratios(self._params)

    def set_value(self, value, name):
        """Sets the value of the selected parameter to the given value.

        Parameters
        ----------
        value: float
            Value for the parameter.
        name: string
            Parameter name.

        Note
        ----
        If an iterable is supplied for both *value* and *name*,
        all parameters in *name* are set to the corresponding *value*."""
        par = self._params
        try:
            for v, n in zip(value, name):
                par[n].value = v
        except:
            par[name].value = value
        self.params = par

    def set_expr(self, expr, name):
        """Sets the expression of the selected parameter
        to the given expression.

        Parameters
        ----------
        expr: str
            Expression for the parameter evaluation.
        name: str
            Parameter name.

        Note
        ----
        If an iterable is supplied for both *expr* and *name*,
        all parameters in *name* are set to the corresponding *expr*."""
        par = self.params
        if isinstance(name, list):
            for n, e in zip(name, expr):
                par[n].expr = e
        else:
            par[name].expr = expr
        self.params = par

    #######################################
    #      METHODS CALLED BY FITTING      #
    #######################################

    def _sanitize_input(self, x, y, yerr=None):
        return x, y, yerr

    def seperate_response(self, x):
        """Wraps the output of the :meth:`__call__` in a list, for
        ease of coding in the fitting routines.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz.

        Returns
        -------
        list of floats or NumPy arrays
            Seperate responses of spectra to the input *x*."""
        return [self(x)]

    ###########################
    #      MAGIC METHODS      #
    ###########################

    def __add__(self, other):
        """Add two spectra together to get an :class:`.IsomerSpectrum`.

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
        elif isinstance(other, IsomerSpectrum):
            l = [self] + other.spectra
        return IsomerSpectrum(l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __call__(self, x):
        """Get the response for frequency *x* (in MHz) of the spectrum.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz

        Returns
        -------
        float or NumPy array
            Response of the spectrum for each value of *x*."""
        if self._params['N'].value > 0:
            s = np.zeros(x.shape)
            for i in range(self._params['N'].value + 1):
                s += (self._params['Poisson'].value ** i) * sum([prof(x + i * self._params['Offset'].value)
                                                for prof in self.parts]) \
                    / np.math.factorial(i)
            s = s * self._params['Scale'].value
        else:
            s = self._params['Scale'].value * sum([prof(x) for prof in self.parts])
        return s + self._params['Background'].value

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None,
             no_of_points=10**3, ax=None, show=True, legend=None,
             data_legend=None, xlabel='Frequency (MHz)', ylabel='Counts',
             indicate=False, bayesian=False, colormap='bone_r'):
        """Plot the hfs, possibly on top of experimental data.

        Parameters
        ----------
        x: array
            Experimental x-data. If None, a suitable region around
            the peaks is chosen to plot the hfs.
        y: array
            Experimental y-data.
        yerr: array or dict('high': array, 'low': array)
            Experimental errors on y.
        no_of_points: int
            Number of points to use for the plot of the hfs if
            experimental data is given.
        ax: matplotlib axes object
            If provided, plots on this axis.
        show: boolean
            If True, the plot will be shown at the end.
        legend: string, optional
            If given, an entry in the legend will be made for the spectrum.
        data_legend: string, optional
            If given, an entry in the legend will be made for the experimental
            data.
        xlabel: string, optional
            If given, sets the xlabel to this string. Defaults to 'Frequency (MHz)'.
        ylabel: string, optional
            If given, sets the ylabel to this string. Defaults to 'Counts'.
        bayesian: boolean, optional
            If given, the region around the fitted line will be shaded, with
            the luminosity indicating the pmf of the Poisson
            distribution characterized by the value of the fit. Note that
            the argument *yerr* is ignored if *bayesian* is True.

        Returns
        -------
        fig, ax: matplotlib figure and axis
            Figure and axis used for the plotting."""

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        toReturn = fig, ax

        if x is None:
            ranges = []
            fwhm = self.parts[0].fwhm

            for pos in self.locations:
                r = np.linspace(pos - 4 * fwhm,
                                pos + 4 * fwhm,
                                2 * 10**2)
                ranges.append(r)
            superx = np.sort(np.concatenate(ranges))
            superx = np.linspace(superx.min(), superx.max(), 10**3)
        else:
            superx = np.linspace(x.min(), x.max(), int(no_of_points))

        if 'sigma_x' in self._params:
            xerr = self._params['sigma_x'].value
        else:
            xerr = 0

        if x is not None and y is not None:
            if not bayesian:
                try:
                    ax.errorbar(x, y, yerr=[yerr['low'], yerr['high']],
                                xerr=xerr, fmt='o', label=data_legend)
                except:
                    ax.errorbar(x, y, yerr=yerr, fmt='o', label=data_legend)
            else:
                ax.plot(x, y, 'o')
        if bayesian:
            range = (superx.min(), superx.max())
            max_counts = np.ceil(-optimize.brute(lambda x: -self(x), (range,), full_output=True)[1])
            min_counts = self._params['Background'].value
            min_counts = np.floor(max(0, min_counts - 3 * min_counts ** 0.5))
            y = np.arange(min_counts, max_counts + 3 * max_counts ** 0.5 + 1)
            x, y = np.meshgrid(superx, y)
            z = poisson_llh(self(x), y)
            z = np.exp(z - z.max(axis=0))

            z = z / z.sum(axis=0)
            ax.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), cmap=plt.get_cmap(colormap))
        if bayesian:
            line, = ax.plot(superx, self(superx), label=legend, lw=0.5)
        else:
            line, = ax.plot(superx, self(superx), label=legend)
        if indicate:
            height = self(superx).min()
            for (p, l) in zip(self.locations, self.ftof):
                lab = l.split('__')
                lableft = '/'.join(lab[0].split('_'))
                labright = '/'.join(lab[1].split('_'))
                lab = '$' + lableft + '\\rightarrow' + labright + '$'
                ax.annotate(lab, xy=(p, height), rotation=90, color=line.get_color(),
                            weight='bold', size=14, ha='center', va='bottom')
                ax.axvline(p, linewidth=0.5, linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show:
            plt.show()
        return toReturn

    def plot_spectroscopic(self, **kwargs):
        """Plots the hfs on top of experimental data
        with errorbar given by the square root of the data.

        Parameters
        ----------
        x: array
            Experimental x-data. If None, a suitable region around
            the peaks is chosen to plot the hfs.
        y: array
            Experimental y-data.
        yerr: array or dict('high': array, 'low': array)
            Experimental errors on y.
        no_of_points: int
            Number of points to use for the plot of the hfs if
            experimental data is given.
        ax: matplotlib axes object
            If provided, plots on this axis.
        show: boolean
            If True, the plot will be shown at the end.
        legend: string, optional
            If given, an entry in the legend will be made for the spectrum.
        data_legend: string, optional
            If given, an entry in the legend will be made for the experimental
            data.

        Returns
        -------
        fig, ax: matplotlib figure and axis
            Figure and axis used for the plotting."""
        y = kwargs.get('y', None)
        if y is not None:
            ylow, yhigh = poisson_interval(y)
            yerr = {'low': y - ylow, 'high': yhigh - y}
        else:
            yerr = None
        kwargs['yerr'] = yerr
        return self.plot(**kwargs)
