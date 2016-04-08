"""
Implementation of a class for the analysis of hyperfine structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import copy
from fractions import Fraction

from . import lmfit as lm
from .basemodel import BaseModel
from .lineid_plot import plot_line_ids
from .loglikelihood import poisson_llh
from .summodel import SumModel
from .utilities import poisson_interval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import satlas.profiles as p
import scipy.optimize as optimize
from sympy.physics.wigner import wigner_6j, wigner_3j

W6J = wigner_6j
W3J = wigner_3j

__all__ = ['HFSModel']


class HFSModel(BaseModel):

    r"""Constructs a HFS spectrum, consisting of different
    peaks described by a certain profile. The number of peaks is governed by
    the nuclear spin and the atomic spins of the levels."""

    __shapes__ = {'gaussian': p.Gaussian,
                  'lorentzian': p.Lorentzian,
                  'crystalball': p.Crystalball,
                  'voigt': p.Voigt}

    def __init__(self, I, J, ABC, centroid, fwhm=[50.0, 50.0], scale=1.0, background_params=[0.001],
                 shape='voigt', use_racah=False, use_saturation=False, saturation=0.001,
                 shared_fwhm=True, n=0, poisson=0.68, offset=0, tailamp=1, tailloc=1):
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
        background_params: list of float, optional
            Sets the coefficients of the polynomial background to the given values.
            Order of polynomial is equal to the number of parameters given minus one.
            Highest order coefficient is the first element, etc.
        shape : string, optional
            Sets the transition shape. String is converted to lowercase. For
            possible values, see *HFSModel__shapes__*.keys()`.
            Defaults to Voigt if an incorrect value is supplied.
        use_racah: boolean, optional
            If True, fixes the relative peak intensities to the Racah intensities.
            Otherwise, gives them equal intensities and allows them to vary during
            fitting.
        use_saturation: boolean, optional
            If True, uses the saturation parameter to calculate relative intensities.
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
        tailamp: float, optional
            Sets the relative amplitude of the tail for the Crystalball shape function.
        tailloc: float, optional
            Sets the location of the tail for the Crystalball shape function.

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
        super(HFSModel, self).__init__()
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
        self._use_racah = use_racah
        self._use_saturation = use_saturation
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

        self._roi = (-np.inf, np.inf)

        self._populate_params(ABC, fwhm, scale, n,
                              poisson, offset, centroid, saturation,
                              tailamp, tailloc, background_params)

    @property
    def locations(self):
        """Contains the locations of the peaks."""
        return self._locations

    @locations.setter
    def locations(self, locations):
        self._locations = np.array(locations)
        for p, l in zip(self.parts, locations):
            p.mu = l

    @property
    def use_racah(self):
        """Boolean to set the behaviour to Racah intensities (True)
        or to individual amplitudes (False)."""
        return self._use_racah

    @use_racah.setter
    def use_racah(self, value):
        self._use_racah = value
        self.params['Scale'].vary = self._use_racah or self._use_saturation
        for label in self.ftof:
            self.params['Amp' + label].vary = not (self._use_racah or self._use_saturation)

    @property
    def use_saturation(self):
        """Boolean to set the behaviour to the saturation model (True)
        or not (False)."""
        return self._use_saturation

    @use_saturation.setter
    def use_saturation(self, value):
        self._use_saturation = value
        self.params['Saturation'].vary = value
        self.params['Scale'].vary = self._use_racah or self._use_saturation
        for label in self.ftof:
            self.params['Amp' + label].vary = not (self._use_racah or self._use_saturation)

    @property
    def roi(self):
        """Tuple of (left, right)-limits between which at least one of
        the peaks has to be located. If the peaks are all outside this
        region, the calculation of the prior returns infinity. Defaults
        to (-np.inf, np.inf)."""
        return self._roi

    @roi.setter
    def roi(self, value):
        self._roi = (value[0], value[1])

    def get_lnprior_mapping(self, params):
        # Implementation uses the 'fail early' paradigm to speed up calculations.
        # First, the easiest checks to fail are made, followed by slower ones.
        # If a check is failed, -np.inf is returned immediately.
        # Check if at least one of the peaks lies within the region of interest
        if not any((self.roi[0] < self.locations) & (self.locations < self.roi[1])):
            return -np.inf
        return super(HFSModel, self).get_lnprior_mapping(params)

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
        if not self.use_racah and not self.use_saturation:
            # When not using set amplitudes, they need
            # to be changed after every iteration
            self._set_amplitudes()
        elif self.use_saturation:
            self._set_transitional_amplitudes()
        else:
            pass
        # Finally, the fwhm of each peak needs to be set
        self._set_fwhm()
        if self.shape.lower() == 'crystalball':
            for part in self.parts:
                part.alpha = params['Taillocation'].value
                part.n = params['Tailamplitude'].value


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
        A = np.append(np.ones(self.num_lower) * self._params['Al'].value,
                      np.ones(self.num_upper) * self._params['Au'].value)
        B = np.append(np.ones(self.num_lower) * self._params['Bl'].value,
                      np.ones(self.num_upper) * self._params['Bu'].value)
        C = np.append(np.ones(self.num_lower) * self._params['Cl'].value,
                      np.ones(self.num_upper) * self._params['Cu'].value)
        centr = np.append(np.zeros(self.num_lower),
                          np.ones(self.num_upper) * self._params['Centroid'].value)
        self.energies = centr + self.C * A + self.D * B + self.E * C

    def _calculate_transition_locations(self):
        self.locations = [self.energies[ind_high] - self.energies[ind_low] for (ind_low, ind_high) in self.transition_indices]

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

    def _populate_params(self, ABC, fwhm, scale,
                         n, poisson, offset, centroid, saturation,
                         tailamp, tailloc, background_params):
        # Prepares the params attribute with the initial values
        par = lm.Parameters()
        if not self.shape.lower() == 'voigt':
            if self.shared_fwhm:
                par.add('FWHM', value=fwhm, vary=True, min=0.0001)
            else:
                if not len(fwhm) == len(self.ftof):
                    fwhm = fwhm[0]
                    fwhm = [fwhm for _ in range(len(self.ftof))]
                for label, val in zip(self.ftof, fwhm):
                    par.add('FWHM' + label, value=val, vary=True, min=0.0001)
        else:
            if self.shared_fwhm:
                par.add('FWHMG', value=fwhm[0], vary=True, min=0.0001)
                par.add('FWHML', value=fwhm[1], vary=True, min=0.0001)
                val = 0.5346 * fwhm[1] + np.sqrt(0.2166 * fwhm[1] ** 2 + fwhm[0] ** 2)
                par.add('TotalFWHM', value=val, vary=False,
                        expr='0.5346*FWHML+(0.2166*FWHML**2+FWHMG**2)**0.5')
            else:
                fwhm = np.array(fwhm)
                if not fwhm.shape[0] == len(self.ftof):
                    fwhm = np.array([[fwhm[0], fwhm[1]] for _ in range(len(self.ftof))])
                for label, val in zip(self.ftof, fwhm):
                    par.add('FWHMG' + label, value=val[0], vary=True, min=0.0001)
                    par.add('FWHML' + label, value=val[1], vary=True, min=0.0001)
                    val = 0.5346 * val[1] + np.sqrt(0.2166 * val[1] ** 2
                                                    + val[0] ** 2)
                    par.add('TotalFWHM' + label, value=val, vary=False,
                            expr='0.5346*FWHML' + label +
                                 '+(0.2166*FWHML' + label +
                                 '**2+FWHMG' + label + '**2)**0.5')
        if self.shape.lower() == 'crystalball':
            par.add('Taillocation', value=tailloc, vary=True)
            par.add('Tailamplitude', value=tailamp, vary=True)
            for part in self.parts:
                part.alpha = tailloc
                part.n = tailamp

        par.add('Scale', value=scale, vary=self.use_racah or self.use_saturation, min=0)
        par.add('Saturation', value=saturation * self.use_saturation, vary=self.use_saturation, min=0)
        amps = self._calculate_transitional_intensities(saturation)
        for label, amp in zip(self.ftof, amps):
            label = 'Amp' + label
            par.add(label, value=amp, vary=not (self.use_racah or self.use_saturation), min=0)

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

        for i, val in reversed(list(enumerate(background_params))):
            par.add('Background' + str(i), value=background_params[i], vary=True)
        par.add('N', value=n, vary=False)
        if n > 0:
            par.add('Poisson', value=poisson, vary=True, min=0, max=1)
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
        """Add two spectra together to get an :class:`.SumModel`.

        Parameters
        ----------
        other: HFSModel
            Other spectrum to add.

        Returns
        -------
        SumModel
            A SumModel combining both spectra."""
        if isinstance(other, HFSModel):
            l = [self, other]
        elif isinstance(other, SumModel):
            l = [self] + other.models
        return SumModel(l)

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
                # print(i, i * self._params['Offset'].value, self._params['Poisson'].value, (self._params['Poisson'].value ** i) / np.math.factorial(i))
                s += (self._params['Poisson'].value ** i) * (sum([prof(x - i * self._params['Offset'].value)
                                                                for prof in self.parts]) * self._params['Scale'].value) / np.math.factorial(i)
            s = s * self._params['Scale'].value
        else:
            s = self._params['Scale'].value * sum([prof(x) for prof in self.parts])
        background_params = [self._params[par_name].value for par_name in self._params if par_name.startswith('Background')]
        return s + np.polyval(background_params, x)

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, x=None, y=None, yerr=None,
             no_of_points=10**3, ax=None, show=True, plot_kws={}):
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
        plot_kws: dictionary
            A dictionary possibly containing the following entries:

            legend: string, optional
                If given, an entry in the legend will be made for the spectrum.
            data_legend: string, optional
                If given, an entry in the legend will be made for the experimental
                data.
            xlabel: string, optional
                If given, sets the xlabel to this string. Defaults to 'Frequency (MHz)'.
            ylabel: string, optional
                If given, sets the ylabel to this string. Defaults to 'Counts'.
            model: boolean, optional
                If given, the region around the fitted line will be shaded, with
                the luminosity indicating the pmf of the Poisson
                distribution characterized by the value of the fit. Note that
                the argument *yerr* is ignored if *model* is True.
            normalized: boolean, optional
                If True, the data and fit are plotted normalized such that the highest
                data point is one.
            background: boolean, optional
                If True, the background is used, otherwise the pure spectrum is plotted.

        Returns
        -------
        fig, ax: matplotlib figure and axis
            Figure and axis used for the plotting."""

        kws = copy.deepcopy(plot_kws)
        legend = kws.pop('legend', None,)
        data_legend = kws.pop('data_legend', None)
        xlabel = kws.pop('xlabel', 'Frequency (MHz)')
        ylabel = kws.pop('ylabel', 'Counts',)
        indicate = kws.pop('indicate', False)
        model = kws.pop('model', False)
        colormap = kws.pop('colormap', 'bone_r',)
        normalized = kws.pop('normalized', False)
        distance = kws.pop('distance', 4)
        background = kws.pop('background', True)

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        toReturn = fig, ax
        color_points = next(ax._get_lines.prop_cycler)['color']
        color_lines = next(ax._get_lines.prop_cycler)['color']

        if x is None:
            ranges = []
            fwhm = self.parts[0].fwhm

            for pos in self.locations:
                r = np.linspace(pos - distance * fwhm,
                                pos + distance * fwhm,
                                2 * 10**2*0+50)
                ranges.append(r)
            superx = np.sort(np.concatenate(ranges))
        else:
            superx = np.linspace(x.min(), x.max(), int(no_of_points))

        if 'sigma_x' in self._params:
            xerr = self._params['sigma_x'].value
        else:
            xerr = 0

        if normalized:
            norm = np.max(y)
            y,yerr = y/norm,yerr/norm
        else:
            norm = 1

        if x is not None and y is not None:
            if not model:
                try:
                    ax.errorbar(x, y, yerr=[yerr['low'], yerr['high']],
                                xerr=xerr, fmt='o', label=data_legend, color=color_points)
                except:
                    ax.errorbar(x, y, yerr=yerr, fmt='o', label=data_legend, color=color_points)
            else:
                ax.plot(x, y, 'o', color=color_points)
        if model:
            superx = np.linspace(superx.min(), superx.max(), len(superx))
            range = (self.locations.min(), self.locations.max())
            max_counts = np.ceil(-optimize.brute(lambda x: -self(x), (range,), full_output=True, Ns=1000, finish=optimize.fmin)[1])
            min_counts = [self._params[par_name].value for par_name in self._params if par_name.startswith('Background')][-1]
            min_counts = np.floor(max(0, min_counts - 3 * min_counts ** 0.5))
            y = np.arange(min_counts, max_counts + 3 * max_counts ** 0.5 + 1)
            x, y = np.meshgrid(superx, y)
            from scipy import stats
            z = stats.poisson(self(x)).pmf(y)

            z = z / z.sum(axis=0)
            ax.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), cmap=plt.get_cmap(colormap))
            line, = ax.plot(superx, self(superx) / norm, label=legend, lw=0.5, color=color_lines)
            # print(superx)
        else:
            if background:
                y = self(superx)
            else:
                background_params = [self._params[par_name].value for par_name in self._params if par_name.startswith('Background')]
                y = self(superx) - np.polyval(background_params, superx)
            line, = ax.plot(superx, y/norm, label=legend, color=color_lines)
        ax.set_xlim(superx.min(), superx.max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if indicate:
            if background:
                Y = self(self.locations)
            else:
                background_params = [self._params[par_name].value for par_name in self._params if par_name.startswith('Background')]
                Y = self(self.locations) - np.polyval(background_params, self.locations)
            # Y = self(self.locations)
            labels = []
            for l in self.ftof:
                lab = l.split('__')
                lableft = '/'.join(lab[0].split('_'))
                labright = '/'.join(lab[1].split('_'))
                lab = '$' + lableft + '\\rightarrow' + labright + '$'
                labels.append(lab)
            plot_line_ids(self.locations, Y, self.locations, labels, ax=ax)
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


    def plot_scheme(self, show=True, upper_color='#D55E00', lower_color='#009E73', arrow_color='#0072B2', distance=5):
        """Create a figure where both the splitting of the upper and lower state is drawn,
        and the hfs associated with this.

        Parameters
        ----------
        show: boolean, optional
            If True, immediately shows the figure. Defaults to True.
        upper_color: matplotlib color definition
            Sets the color of the upper state. Defaults to red.
        lower_color: matplotlib color definition
            Sets the color of the lower state. Defaults to black.
        arrow_color: matplotlib color definition
            Sets the color of the arrows indicating the transitions.
            Defaults to blue.

        Returns
        -------
        tuple
            Tuple containing the figure and both axes, also in a tuple."""
        from fractions import Fraction
        from matplotlib import lines
        length_plot = 0.4
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0.5, 0, length_plot, 0.5], axisbg=[1, 1, 1, 0])
        self.plot(ax=ax, show=False, plot_kws={'distance': distance})
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        locations = self.locations
        plotrange = ax.get_xlim()
        distances = (locations - plotrange[0]) / (plotrange[1] - plotrange[0]) * length_plot
        height = self(locations)
        plotrange = ax.get_ylim()
        height = (height - plotrange[0]) / (plotrange[1] - plotrange[0]) / 2
        A = np.append(np.ones(self.num_lower) * self._params['Al'].value,
                              np.ones(self.num_upper) * self._params['Au'].value)
        B = np.append(np.ones(self.num_lower) * self._params['Bl'].value,
                              np.ones(self.num_upper) * self._params['Bu'].value)
        C = np.append(np.ones(self.num_lower) * self._params['Cl'].value,
                              np.ones(self.num_upper) * self._params['Cu'].value)
        energies = self.C * A + self.D * B + self.E * C
        #energies -= energies.min()
        energies_upper = energies[self.num_lower:]
        energies_upper_norm = np.abs(energies_upper.max()) if not energies_upper.max()==0 else 1
        energies_upper = energies_upper / energies_upper_norm * 0.1
        energies_lower = energies[:self.num_lower]
        energies_lower_norm = np.abs(energies_lower.max()) if not energies_lower.max()==0 else 1
        energies_lower = energies_lower / energies_lower_norm * 0.1
        energies = np.append(energies_lower, energies_upper)

        ax2 = fig.add_axes([0, 0, 1, 1], axisbg=[1, 1, 1, 0])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.axis('off')

        # Lower state
        x = np.array([0, 0.3])
        y = np.array([0.625, 0.625])
        line = lines.Line2D(x, y, lw=2., color=lower_color)
        ax2.add_line(line)

        # Upper state
        x = np.array([0, 0.3])
        y = np.array([0.875, 0.875])
        line = lines.Line2D(x, y, lw=2., color=upper_color)
        ax2.add_line(line)

        for i, F in enumerate(self.F):
            # Level
            F = Fraction.from_float(F)
            x = np.array([0.5, 0.5 + length_plot])
            if i < self.num_lower:
                y = np.zeros(len(x)) + 0.625 + energies[i]
                color = lower_color
                starting = 0.625
            else:
                y = np.zeros(len(x)) + 0.875 + energies[i]
                color = upper_color
                starting = 0.875
            line = lines.Line2D(x, y, lw=2., color=color)
            ax2.add_line(line)

            x = np.array([0.3, x.min()])
            y = np.array([starting, y[0]])
            line = lines.Line2D(x, y, lw=2., color=color, alpha=0.4, linestyle='dashed')
            ax2.add_line(line)
            ax2.text(0.5 + length_plot, y[-1], 'F=' + str(F) + ' ', fontsize=20, fontdict={'horizontalalignment': 'left', 'verticalalignment': 'center'})

        for i, label in enumerate(self.ftof):
            lower, upper = label.split('__')
            if '_' in lower:
                lower = lower.split('_')
                lower = float(lower[0]) / float(lower[1])
            else:
                lower = float(lower)
            if '_' in upper:
                upper = upper.split('_')
                upper = float(upper[0]) / float(upper[1])
            else:
                upper = float(upper)

            x = np.array([distances[i], distances[i]]) + 0.5
            lower = energies[np.where(self.F[:self.num_lower]==lower)[0]]
            upper = energies[np.where(self.F[self.num_lower:]==upper)[0] + self.num_lower]
            y = np.array([lower + 0.625, upper + 0.875]).flatten()
            ax2.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], fc=arrow_color, ec=arrow_color, length_includes_head=True, overhang=0.5, zorder=10)

        ax2.text(0.15, 0.64, 'J=' + str(Fraction.from_float(self.J[0])), fontsize=20, fontdict={'horizontalalignment': 'center'})
        ax2.text(0.15, 0.89, 'J=' + str(Fraction.from_float(self.J[-1])), fontsize=20, fontdict={'horizontalalignment': 'center'})
        ax2.text(0.15, 0.765, 'I=' + str(Fraction.from_float(self.I)), fontsize=20, fontdict={'horizontalalignment': 'right'})
        if show:
            plt.show()
        return fig, (ax, ax2)
