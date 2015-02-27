"""
.. module:: spectrum
    :platform: Windows
    :synopsis: Implementation of classes for the analysis of hyperfine
     structure spectra, including simultaneous fitting, various fitting
     routines and isomeric presence.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import numpy as np
import lmfit as lm
import satlas.profiles as p
import satlas.loglikelihood as llh
import emcee as mcmc
import satlas.triangle as tri
from satlas.wigner import wigner_6j as W6J


class PriorParameter(lm.Parameter):

    """Extended the Parameter class from LMFIT to incorporate prior boundaries.
    """

    def __init__(self, name, value=None, vary=True, min=None, max=None,
                 expr=None, priormin=None, priormax=None):
        super(PriorParameter, self).__init__(name, value=value,
                                             vary=vary, min=min,
                                             max=max, expr=expr)
        self.priormin = priormin
        self.priormax = priormax


class Spectrum(object):

    """Baseclass for all spectra, such as :class:`SingleSpectrum`,
    :class:`CombinedSpectrum` and :class:`IsomerSpectrum`. For input, see these
    classes.

    Attributes
    ----------
    selected: list of strings
        When a walk is performed and a triangle plot is requested, the
        parameters with one of these strings in their name will be displayed in
        a seperate plot. Defaults to the hyperfine parameters and the centroid.
    """

    def __init__(self):
        super(Spectrum, self).__init__()
        self.selected = ['Al', 'Au', 'Bl', 'Bu', 'Cl', 'Cu', 'df']
        self.showSelected = True
        self.showAll = False
        self.atol = 0.1

    def sanitizeFitInput(self, x, y, yerr=None):
        return x, y, yerr

    def loglikeli(self, params, x, y):
        """Assuming a Poisson-distribution for the error of the measurements,
        returns the loglikelihoods for the given parameter dictionary 'params'.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters for which the loglikelihood has to be
            determined.
        x: array_like
            Frequencies in MHz.
        y: array_like
            Counts corresponding to :attr:`x`."""
        self.varFromParams(params)
        if any([np.isclose(X.min(), X.max(), atol=self.atol)
                for X in self.seperateResponse(x)]):
            return -np.inf
        return llh.Poisson(y, self(x))

    def lnprob(self, params, x, y):
        """Calculates the sum of the loglikelihoods given the parameters
        :attr:`params`, while also checking the prior first. If this prior
        rejects the parameters, the parameters are not set for the spectrum.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters for which the sum loglikelihood has to be
            calculated.
        x: array_like
            Frequencies in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.

        Returns
        -------
        float
            Sum of the loglikelihoods plus the result of the prior."""
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        res = lp + np.sum(self.loglikeli(params, x, y))
        return res

    def LikelihoodSpectroscopicFit(self, x, y, walking=True, **kwargs):
        """Fit the spectrum to the spectroscopic data using the Maximum
        Likelihood technique. This is done by minimizing the negative sum of
        the loglikelihoods of the spectrum given the data (given by the method
        :meth:`lnprob`). Prints a statement regarding the success of the
        fitting.

        Parameters
        ----------
        x: array_like
            Frequency of the data, in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.
        walking: Boolean
            Determines if a Monte Carlo walk is performed after the
            minimization to determine the errorbars and distribution of the
            parameters.
        kwargs: misc
            Keyword arguments passed on to the method :meth:`walk`.

        Returns
        -------
        tuple or :class:`None`
            If any kind of plot is requested, a tuple containing these figures
            will be returned (see :meth:`walk` for more details). If no plot
            is requested, returns the value :class:`None`."""

        def negativeloglikelihood(*args, **kwargs):
            return -self.lnprob(*args, **kwargs)

        x, y, _ = self.sanitizeFitInput(x, y)
        params = self.paramsFromVar()
        result = lm.Minimizer(negativeloglikelihood, params, fcn_args=(x, y))
        result.scalar_minimize(method='Nelder-Mead')
        self.varFromParams(result.params)
        self.MLEFit = result.params

        print(result.message)
        self.DisplayMLEFit()
        if walking:
            return self.walk(x, y, **kwargs)
        else:
            return None

    def walk(self, x, y, showLikeli=False, showWalks=False, showTriangle=False,
             nsteps=2000, walkers=20, burnin=10.0):
        """Performs a random walk in the parameter space to determine the
        distribution for the best fit of the parameters.

        A message is printed before and after the walk.

        Parameters
        ----------
        x: array_like
            Frequency of the data, in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.
        showLikeli: Boolean, optional
            Set this to True if a plot depicting the 1D slice of the
            loglikelihood function for each parameter has to be plotted.
        showWalks: Boolean, optional
            Set this to True if a plot depicting the random walk has to be
            plotted.
        showTriangle: Boolean, optional
            Set this to True is a triangle plot depicting the distributions of
            the parameters in the random walk has to be plotted. Two plots are
            created: one with all the parameters, one with only the parameters
            as filtered using :attr:`selected`.
        nsteps: int, optional
            Number of steps to be taken, defaults to 2000.
        walkers: int, optional
            Number of walkers to be used, defaults to 20.
        burnin: float, optional
            Burn-in to be used for the walk. Expressed in percentage,
            defaults to 10.0.

        Returns
        -------
        returnfigs: tuple
            If any plots have been made by setting booleans, the figure handles
            are returned as a tuple, ordered as
            :attr:`(Likelihood, walks, triangleComplete, triangleSelected)`.
            Plots that have not been requested are not added. If no plots have
            been requested, returns an empty tuple."""
        params = self.paramsFromVar()
        self.MLEFit = self.paramsFromVar()
        var_names = []
        vars = []
        for key in params.keys():
            if params[key].vary:
                var_names.append(key)
                vars.append(params[key].value)
        ndim = len(vars)
        pos = [vars + 1e-4 * np.random.randn(ndim) for i in range(walkers)]
        x, y, _ = self.sanitizeFitInput(x, y)

        def lnprobList(fvars, x, y, groupParams):
            for val, n in zip(fvars, var_names):
                groupParams[n].value = val
            return self.lnprob(groupParams, x, y)
        groupParams = lm.Parameters()
        for key in params.keys():
            groupParams[key] = PriorParameter(key,
                                              value=params[key].value,
                                              vary=params[key].vary,
                                              expr=params[key].expr,
                                              priormin=params[key].min,
                                              priormax=params[key].max)
        sampler = mcmc.EnsembleSampler(walkers, ndim, lnprobList,
                                       args=(x, y, groupParams))
        burn = int(nsteps / burnin)
        print('Starting burn-in ({} steps)...'.format(burn))
        sampler.run_mcmc(pos, burn, storechain=False)
        print('Starting walk ({} steps)...'.format(nsteps - burn))
        sampler.run_mcmc(pos, nsteps - burn)
        print('Done.')
        samples = sampler.flatchain
        val = []
        err = []
        q = [16.0, 50.0, 84.0]
        for i, samp in enumerate(samples.T):
            q16, q50, q84 = np.percentile(samp, q)
            val.append(q50)
            err.append(max([q50 - q16, q84 - q50]))

        for n, v, e in zip(var_names, val, err):
            params[n].value = v
            params[n].stderr = e

        self.MLEFit = params
        self.varFromParams(params)

        returnfigs = ()

        if showLikeli or showWalks or showTriangle:
            import matplotlib.pyplot as plt
            try:
                import seaborn
                seaborn.set()
            except ImportError:
                pass
        if showLikeli:
            shape = int(np.ceil(np.sqrt(len(var_names))))
            figLikeli, axes = plt.subplots(shape, shape)
            axes = axes.flatten()
            for i, (n, truth, a) in enumerate(zip(var_names, val, axes)):
                st = samples.T
                left, right = (truth - 5 * np.abs(st[i, :].min()),
                               truth + 5 * np.abs(st[i, :].max()))
                # l = val[i] - 100 * self.MLEFit[n].stderr
                # r = val[i] + 100 * self.MLEFit[n].stderr
                # left = l if left is None else max(l, left)
                # right = r if right is None else min(r, right)
                xvalues = np.linspace(left, right, 1000)
                dummy = np.array(vars, dtype='float')
                yvalues = np.zeros(xvalues.shape[0])
                for j, value in enumerate(xvalues):
                    dummy[i] = value
                    yvalues[j] = lnprobList(dummy, x, y, groupParams)
                a.plot(xvalues, yvalues, color="k")
                a.axvline(truth, color="#888888", lw=2)
                a.set_ylabel(n)
                self.varFromParams(self.MLEFit)
            plt.tight_layout()

            returnfigs += (figLikeli,)

        if showWalks:
            shape = int(np.ceil(np.sqrt(len(var_names))))
            figWalks, axes = plt.subplots(shape, shape, sharex=True)
            axes = axes.flatten()

            for i, (n, truth, a) in enumerate(zip(var_names, vars, axes)):
                a.plot(sampler.chain[:, :, i].T,
                       color="k", alpha=0.4)
                a.axhline(truth, color="#888888", lw=2)
                a.set_xlim([0, nsteps])
                a.set_ylabel(n)
            plt.tight_layout()

            returnfigs += (figWalks,)

        if showTriangle:
            if self.showAll:
                figTri1 = tri.corner(samples,
                                     labels=var_names,
                                     truths=vars,
                                     plot_datapoints=False,
                                     show_titles=True,
                                     quantiles=[0.16, 0.5, 0.84],
                                     verbose=False)
                returnfigs += (figTri1,)

            if self.showSelected:
                s = []
                for i, v in enumerate(var_names):
                    for r in self.selected:
                        if r in v:
                            s.append(i)
                selected_samples = np.zeros((samples.shape[0], len(s)))
                for i, val in enumerate(s):
                    selected_samples[:, i] = samples[:, val]
                samples = selected_samples
                var_names = (np.array(var_names)[s]).tolist()
                vars = (np.array(vars)[s]).tolist()

                figTri = tri.corner(samples,
                                    labels=var_names,
                                    truths=vars,
                                    plot_datapoints=False,
                                    show_titles=True,
                                    quantiles=[0.16, 0.5, 0.84],
                                    verbose=False)

                returnfigs += (figTri,)
        return returnfigs

    def DisplayMLEFit(self):
        """Give a readable overview of the result of the MLE fitting routine.
        """
        lm.report_fit(self.MLEFit)

    def FitToSpectroscopicData(self, x, y):
        """Use the :meth:`FitToData` method, automatically estimating the errors on the
        counts by the square root."""
        x, y, _ = self.sanitizeFitInput(x, y)
        yerr = np.sqrt(y)
        yerr[np.isclose(yerr, 0.0)] = 1.0
        return self.FitToData(x, y, yerr)

    def FitToData(self, x, y, yerr):
        """Use a non-linear least squares minimization (Levenberg-Marquardt)
        algorithm to minimize the chi-square of the fit to data :attr:`x` and
        :attr:`y` with errorbars :attr:`yerr`. Reasonable bounds are used on
        parameters, and the user-supplied :attr:`self._vary` dictionary is
        consulted to see if a parameter should be varied or not.

        Parameters
        ----------
        x: array_like
            Frequency of the data, in MHz.
        y: array_like
            Counts corresponding to :attr:`x`.
        yerr: array_like
            Error bars on :attr:`y`."""

        x, y, yerr = self.sanitizeFitInput(x, y, yerr)

        def Model(params):
            self.varFromParams(params)
            return (y - self(x)) / yerr

        params = self.paramsFromVar()
        result = lm.minimize(Model, params)

        self.ChiSquareFit = result

    def DisplayFit(self, **kwargs):
        """Display all relevent info of the least-squares fitting routine,
        if this has been performed.

        Parameters
        ----------
        kwargs: misc
            Keywords passed on to :func:`fit_report` from the LMFit package."""
        if hasattr(self, 'ChiSquareFit'):
            print('Scaled errors estimated from covariance matrix.')
            print(lm.fit_report(self.ChiSquareFit, **kwargs))
        else:
            print('Spectrum has not yet been fitted!')


class CombinedSpectrum(Spectrum):

    """A class for combining different spectra (:class:`CombinedSpectrum`) or
    combining isomers/isotopes (:class:`IsomerSpectrum`, child class).

    Parameters
    ----------
    spectra: list of :class:`IsomerSpectrum` or :class:`SingleSpectrum` objects
        A list defining the different spectra."""

    def __init__(self, spectra):
        super(CombinedSpectrum, self).__init__()
        self.spectra = spectra
        self.shared = ['Al',
                       'Au',
                       'Bl',
                       'Bu',
                       'Cl',
                       'Cu',
                       'Offset']

    def sanitizeFitInput(self, x, y, yerr=None):
        """Take the :attr:`x`, :attr:`y`, and :attr:`yerr` inputs, and sanitize
        them for the fit, meaning it should convert :attr:`y`/:attr:`yerr` to
        the output format of the class, and :attr:`x` to the input format of
        the class."""
        if isinstance(y, list):
            y = np.hstack(y)
        if yerr is not None:
            if isinstance(yerr, list):
                yerr = np.hstack(yerr)
        return x, y, yerr

    def paramsFromVar(self):
        """Combine the parameters from the subspectra into one Parameters
        instance.

        Returns
        -------
        params: Parameters instance describing the spectrum.

        Warning
        -------
        Black magic going on in here, especially in the block of code
        describing the shared parameters."""
        params = lm.Parameters()
        for i, s in enumerate(self.spectra):
            p = s.paramsFromVar()
            keys = list(p.keys())
            for old_key in keys:
                new_key = 's' + str(i) + '_' + old_key
                p[new_key] = p.pop(old_key)
                for o_key in keys:
                    if p[new_key].expr is not None:
                        n_key = 's' + str(i) + '_' + o_key
                        p[new_key].expr = p[new_key].expr.replace(o_key, n_key)
            params += p

        for i, s in enumerate(self.spectra):
            for key in self.shared:
                if i == 0:
                    continue
                if isinstance(self.spectra[i], IsomerSpectrum):
                    for j, _ in enumerate(self.spectra[i].spectra):
                        first_key = 's0_s' + str(j) + '_' + key
                        new_key = 's' + str(j) + '_' + key
                        for p in params.keys():
                            if new_key in p:
                                if p.startswith('s0_'):
                                    pass
                                else:
                                    params[p].expr = first_key
                                    params[p].vary = False
                else:
                    if isinstance(self.spectra[0], IsomerSpectrum):
                        first_key = 's0_s0_' + key
                    else:
                        first_key = 's0_' + key
                    new_key = 's' + str(i) + '_' + key
                    for p in params.keys():
                        if new_key in p:
                            params[p].expr = first_key
                            params[p].vary = False
        return params

    def varFromParams(self, params):
        """Given a Parameters instance such as returned by the method
        :meth:`paramsFromVar`, set the parameters of the subspectra to the
        appropriate values.

        Parameters
        ----------
        params: Parameters
            Parameters instance containing the information for the variables.
        """
        for i, s in enumerate(self.spectra):
            p = lm.Parameters()
            if isinstance(s, IsomerSpectrum):
                for j, spec in enumerate(s.spectra):
                    for key in params.keys():
                        k = 's{!s}_s{!s}_'.format(i, j)
                        if key.startswith(k):
                            dinkie = params[key]
                            new_name = key.split('_')
                            new_name = '_'.join(new_name[1:])
                            dinkie.name = new_name
                            p[new_name] = dinkie
            else:
                for key in params.keys():
                    if key.startswith('s' + str(i) + '_'):
                        dinkie = params[key]
                        new_name = key.split('_')[-1]
                        dinkie.name = new_name
                        p[new_name] = dinkie
            s.varFromParams(p)

    def splitParams(self, params):
        """Helper function to split the parameters of the IsomerSpectrum
        instance into a list of parameters suitable for each subspectrum.

        Parameters
        ----------
        params: Parameters
            Parameters of the :class:`IsomerSpectrum` instance.

        Returns
        -------
        p: list of Parameters
            A list of Parameters instances, each entry corresponding to the
            same entry in the attribute :attr:`spectra`."""
        p = []
        for i, _ in enumerate(self.spectra):
            par = lm.Parameters()
            for key in params:
                if key.startswith('s'+str(i)+'_'):
                    new_key = key[len('s'+str(i)+'_'):]
                    expr = params[key].expr
                    if expr is not None:
                        for k in params:
                            nk = k[len('s'+str(i)+'_'):]
                            expr = expr.replace(k, nk)
                    par.add(new_key,
                            value=params[key].value,
                            vary=params[key].vary,
                            min=params[key].min,
                            max=params[key].max,
                            expr=expr)
            p.append(par)
        return p

    def lnprior(self, params):
        """Defines the (uninformative) prior for all parameters.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters with values to be used in the fit/walk

        Returns
        -------
        float
            If any of the parameters are out of bounds, returns :data:`-np.inf`
            , otherwise 1.0 is returned"""
        params = self.splitParams(params)
        return np.sum([s.lnprior(par) for s, par in zip(self.spectra, params)])

    def seperateResponse(self, x):
        return [s.seperateResponse(X) for s, X in zip(self.spectra, x)]

    def __call__(self, x):
        return np.hstack([s(X) for s, X in zip(self.spectra, x)])


class IsomerSpectrum(CombinedSpectrum):

    """Create a spectrum containing the information of multiple hyperfine
    structures. Most common use will be to fit a spectrum containing an isomer,
    hence the name of the class.

    Parameters
    ----------
    spectra: list of :class:`SingleSpectrum` instances
        A list containing the base spectra"""

    def __init__(self, spectra):
        super(IsomerSpectrum, self).__init__(spectra)
        self.shared = []

    def sanitizeFitInput(self, x, y, yerr=None):
        """Doesn't do anything yet."""
        x, y = np.array(x), np.array(y)
        if yerr is not None:
            yerr = np.array(yerr)
        return x, y, yerr

    def paramsFromVar(self):
        """Combine the parameters from the subspectra into one Parameters
        instance.

        Returns
        -------
        params: Parameters instance describing the spectrum"""
        params = super(IsomerSpectrum, self).paramsFromVar()
        for i, s in enumerate(self.spectra):
            if i == 0:
                continue
            else:
                new_key = 's' + str(i) + '_Background'
                params[new_key].value = 0
                params[new_key].vary = False
                params[new_key].expr = None
        return params

    def walk(self, x, y, showLikeli=False, showWalks=False, showTriangle=False,
             nsteps=2000, walkers=40):
        """Set the default number of walkers to 40, for the higher number
        of parameters."""
        super(IsomerSpectrum, self).walk(x, y, showLikeli, showWalks,
                                         showTriangle, nsteps, walkers)

    def seperateResponse(self, x):
        """Get the response for each seperate spectrum for the values x,
        without background.

        Parameters
        ----------
        x : float or array_like
            Frequency in MHz.

        Returns
        -------
        list of floats or NumPy arrays
            Seperate responses of spectra to the input :attr:`x`."""
        return [s(x) - s.background for s in self.spectra]

    def __add__(self, other):
        if isinstance(other, IsomerSpectrum):
            spectra = self.spectra + other.spectra
        elif isinstance(other, SingleSpectrum):
            spectra = self.spectra
            spectra.append(other)
        else:
            raise TypeError('unsupported operand type(s)')
        return IsomerSpectrum(spectra)

    def __call__(self, x):
        return np.sum([s(x) for s in self.spectra], axis=0)


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
    rAmp: Boolean, optional
        If True, fixes the relative peak intensities to the Racah intensities.
        Otherwise, gives them equal intensities and allows them to vary during
        fitting.
    sameFWHM: Boolean, optional
        If True, the same FWHM is used for all peaks. Otherwise, give them all
        the same initial FWHM and let them vary during the fitting.

    Attributes
    ----------
    fwhm : (list of) float or list of 2 floats
        Sets the FWHM for all the transtions. If :attr:`sameFWHM` is True,
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
                 background=0.0, shape='voigt', rAmp=True, sameFWHM=True):
        super(SingleSpectrum, self).__init__()
        shape = shape.lower()
        if shape not in self.__shapes__:
            print("""Given profile shape not yet supported.
            Defaulting to Voigt lineshape.""")
            shape = 'voigt'
            fwhm = [50.0, 50.0]

        self.shape = shape
        self._relAmp = []
        self._rAmp = rAmp
        self.sameFWHM = sameFWHM
        self.parts = []
        self._I = I
        self._J = J
        self._ABC = ABC
        self.ABCLimit = 30000.0
        self.FWHMLimit = 0.1
        self._df = df

        self.scale = scale
        self._background = background

        self._energies = []
        self._mu = []

        self.n = 0
        self.poisson = 0.609
        self.offset = 0

        self._vary = {}
        self.ratio = [None, None, None]

        self.calculateLevels()
        self.fwhm = fwhm

    def setVary(self, varyDict):
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
        self.calculateTransitions()

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        self.calculateTransitions()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = np.abs(value)

    @property
    def rAmp(self):
        return self._rAmp

    @rAmp.setter
    def rAmp(self, value):
        self._rAmp = value
        self.calculateInt()

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
        if self.sameFWHM:
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

        self.calculateTransitions()
        self.calculateInt()

    def calculateTransitions(self):
        self._energies = [[self.calculateEnergy(0, F) for F in self._F[0]],
                          [self.calculateEnergy(1, F) for F in self._F[1]]]

        mu = []
        for i, F1 in enumerate(self._F[0]):
            for j, F2 in enumerate(self._F[1]):
                if abs(F2 - F1) <= 1 and not F2 == F1 == 0.0:
                    mu.append(self._energies[1][j] - self._energies[0][i])

        if not len(self.parts) is len(mu):
            self.parts = tuple(
                self.__shapes__[self.shape]() for _ in range(len(mu)))
        self.mu = mu

    def calculateInt(self):
        ampl = []
        if self.I == 0:
            ampl = [1.0]
        else:
            for i, F1 in enumerate(self._F[0]):
                for j, F2 in enumerate(self._F[1]):
                    a = self.calculateRacah(self._J[0],
                                            self._J[1], F1, F2)
                    if a != 0.0:
                        ampl.append(a)
        self.relAmp = ampl

    def calculateRacah(self, J1, J2, F1, F2, order=1.0):
        return (2 * F1 + 1) * (2 * F2 + 1) * \
            W6J(J2, F2, self._I, F1, J1, order) ** 2

    def calculateEnergy(self, level, F):
        r"""The hyperfine addition to a central frequency (attribute *df*) for
        a specific level is calculated. The formula used is

        .. math::
            C_F &= F(F+1) - I(I+1) - J(J+1)

            D_F &= \frac{3 C_F (C_F + 1) - 4 I (I + 1) J (J + 1)}{2 I (2 I - 1)
            J (2 J - 1)}

            E_F &= \frac{10 (\frac{C_F}{2})^3 + 20(\frac{C_F}{2})^2 + C_F(-3I(I
            + 1)J(J + 1) + I(I + 1) + J(J + 1) + 3) - 5I(I + 1)J(J + 1)}{I(I -
            1)(2I - 1)J(J - 1)(2J - 1)}

            E &= df + \frac{A C_F}{2} + \frac{B D_F}{4} + C E_F

        A, B and C are the dipole, quadrupole and octupole hyperfine parameters.
        Octupole contributions are calculated when both the nuclear and
        electronic spin is greater than 1, quadrupole contributions when they
        are greater than 1/2, and dipole contributions when they are greater
        than 0.

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
            C_F = F*(F+1) - I*(I+1) - J*(J+1)
            D_F = (3 * C_F * (C_F + 1) -
                   4 * I * (I + 1) * J * (J + 1)) /\
                  (2 * I * (2 * I - 1) * J * (2 * J - 1))
            E_F = (10 * (0.5 * C_F) ** 3 + 20 * (0.5 * C_F) ** 2
                   + C_F * (-3 * I * (I + 1) * J * (J + 1) +
                            I * (I + 1) + J * (J + 1) + 3) -
                   5 * I * (I + 1) * J * (J + 1)) /\
                  (I * (I - 1) * (2 * I - 1) * J * (J - 1) * (2 * J - 1))

        return df + 0.5 * A * C_F + 0.25 * B * D_F + C * E_F

    def varFromParams(self, params):
        """Given a Parameters instance 'params', the value-fields for all the
        parameters are extracted and used to set the values of the spectrum.
        Will raise a KeyError exception if an unsuitable instance is
        supplied.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters to set all values."""
        if self.shape not in ['extendedvoigt', 'voigt']:
            if self.sameFWHM:
                self.fwhm = params['FWHM'].value
            else:
                self.fwhm = [params['FWHM'+str(i)].value
                             for i in range(len(self.parts))]
            if self.shape in ['pseudovoigt']:
                for part in self.parts:
                    part.n = params['eta'].value
        else:
            if self.sameFWHM:
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

    def paramsFromVar(self):
        """Goes through all the relevant parameters of the spectrum,
        and returns a Parameters instance containing all the information. User-
        supplied information in the self._vary dictionary is used to set
        the variation of parameters during the fitting, while
        making sure that the A, B and C parameters are not used if the spins
        do not allow it.

        Returns
        -------
        Parameters
            Instance suitable for the method :meth:`varFromParams`."""
        par = lm.Parameters()
        if self.shape not in ['extendedvoigt', 'voigt']:
            if self.sameFWHM:
                par.add('FWHM', value=self.fwhm, vary=True, min=self.FWHMLimit)
            else:
                for i, val in enumerate(self.fwhm):
                    par.add('FWHM' + str(i), value=val, vary=True,
                            min=self.FWHMLimit)
            if self.shape in ['pseudovoigt']:
                par.add('eta', value=self.parts[0].n, vary=True, min=0, max=1)
        else:
            if self.sameFWHM:
                par.add('FWHMG', value=self.fwhm[0], vary=True,
                        min=self.FWHMLimit)
                par.add('FWHML', value=self.fwhm[1], vary=True,
                        min=self.FWHMLimit)
                val = 0.5346 * self.fwhm[1] + np.sqrt(0.2166 *
                                                      self.fwhm[1] ** 2
                                                      + self.fwhm[0] ** 2)
                par.add('TotalFWHM', value=val, vary=False,
                        expr='0.5346*FWHML+sqrt(0.2166*FWHML**2+FWHMG**2)')
            else:
                for i, val in enumerate(self.fwhm):
                    par.add('FWHMG' + str(i), value=val[0], vary=True,
                            min=self.FWHMLimit)
                    par.add('FWHML' + str(i), value=val[1], vary=True,
                            min=self.FWHMLimit)
                    val = 0.5346 * val[1] + np.sqrt(0.2166 * val[1] ** 2
                                                    + val[0] ** 2)
                    par.add('TotalFWHM' + str(i), value=val, vary=False,
                            expr='0.5346*FWHML' + str(i) +
                                 '+sqrt(0.2166*FWHML' + str(i) +
                                 '**2+FWHMG' + str(i) + '**2)')

        par.add('scale', value=self.scale, vary=self.rAmp, min=0)
        for i, prof in enumerate(self.parts):
            par.add('Amp' + str(i), value=self._relAmp[i], vary=not self.rAmp,
                    min=0)

        b = (None, None) if self.ABCLimit is None else (-self.ABCLimit,
                                                        self.ABCLimit)
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
            par.add('Poisson', value=self._Poisson, vary=True, min=0)
            par.add('Offset', value=self._Offset, vary=True, max=0)
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

    def lnprior(self, params):
        """Defines the (uninformative) prior for all parameters.

        Parameters
        ----------
        params: Parameters
            Instance of Parameters with values to be used in the fit/walk

        Returns
        -------
        float
            If any of the parameters are out of bounds, returns :data:`-np.inf`
            , otherwise 1.0 is returned"""
        for key in params.keys():
            try:
                leftbound, rightbound = params[key].priormin, params[key].priormax
            except:
                leftbound, rightbound = params[key].min, params[key].max
            leftbound = -np.inf if leftbound is None else leftbound
            rightbound = np.inf if rightbound is None else rightbound
            if not leftbound <= params[key].value <= rightbound:
                return -np.inf
        return 1.0

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
        return IsomerSpectrum(l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def seperateResponse(self, x):
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
