"""
.. module:: CombinedSpectrum
    :platform: Windows
    :synopsis: Implementation of class for the simultaneous fitting of hyperfine
     structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
from satlas.spectrum import Spectrum


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

    def sanitize_input(self, x, y, yerr=None):
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

    def params_from_var(self):
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
        from satlas.isomerspectrum import IsomerSpectrum
        for i, s in enumerate(self.spectra):
            p = s.params_from_var()
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

    def var_from_params(self, params):
        """Given a Parameters instance such as returned by the method
        :meth:`params_from_var`, set the parameters of the subspectra to the
        appropriate values.

        Parameters
        ----------
        params: Parameters
            Parameters instance containing the information for the variables.
        """
        from satlas.isomerspectrum import IsomerSpectrum

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
                            p.add(new_name, value=dinkie.value,
                                  vary=dinkie.vary, min=dinkie.min,
                                  max=dinkie.max, expr=dinkie.expr)
            else:
                for key in params.keys():
                    if key.startswith('s' + str(i) + '_'):
                        dinkie = params[key]
                        new_name = key.split('_')[-1]
                        p.add(new_name, value=dinkie.value, vary=dinkie.vary,
                              min=dinkie.min, max=dinkie.max, expr=dinkie.expr)
            s.var_from_params(p)

    def split_parameters(self, params):
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
        params = self.split_parameters(params)
        return np.sum([s.lnprior(par) for s, par in zip(self.spectra, params)])

    def seperate_response(self, x):
        return np.squeeze([s.seperate_response(X)
                           for s, X in zip(self.spectra, x)])

    ###############################
    #      PLOTTING ROUTINES      #
    ###############################

    def plot(self, xs=None, ys=None, yerrs=None,
             no_of_points=10**4, ax=None):
        """Routine that plots the hfs of all the spectra,
        possibly on top of experimental data.

        Parameters
        ----------
        x: list of arrays
            Experimental x-data. If list of Nones, a suitable region around
            the peaks is chosen to plot the hfs.
        y: list of arrays
            Experimental y-data.
        yerr: list of arrays
            Experimental errors on y.
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
        if ax is None:
            fig, ax = plt.subplots(len(self.spectra), 1, sharex=True)
        if xs is None:
            xs = [None] * len(self.spectra)
        if ys is None:
            ys = [None] * len(self.spectra)
        if yerrs is None:
            yerrs = [None] * len(self.spectra)

        for i, (x, y, yerr, spec) in enumerate(zip(xs, ys, yerrs,
                                                   self.spectra)):
            if x is not None and y is not None:
                ax[i].errorbar(x, y, yerr, fmt='o', markersize=3)
            spec.plot(x, y, yerr, no_of_points, ax[i], show=False, label=False)

        ax[len(xs)-1].set_xlabel('Frequency (MHz)', fontsize=16)
        ax[0].set_ylabel('Counts', fontsize=16)

        plt.tight_layout()
        plt.show()

    def plot_spectroscopic(self, xs=None, ys=None,
                           no_of_points=10**4, ax=None):
        """Routine that plots the hfs of all the spectra, possibly on
        top of experimental data. It assumes that the y data is drawn from
        a Poisson distribution (e.g. counting data).

        Parameters
        ----------
        x: list of arrays
            Experimental x-data. If list of Nones, a suitable region around
            the peaks is chosen to plot the hfs.
        y: list of arrays
            Experimental y-data.
        yerr: list of arrays
            Experimental errors on y.
        no_of_points: int
            Number of points to use for the plot of the hfs.
        ax: matplotlib axes object
            If provided, plots on this axis
        show: Boolean
            if True, the plot will be shown at the end.

        Returns
        -------
        None"""

        if ys is not None and not any([y is None for y in ys]):
            yerrs = [np.sqrt(y + 1) for y in ys]
        else:
            yerrs = [None for y in ys]
        self.plot(xs, ys, yerrs, no_of_points, ax)

    def __call__(self, x):
        return np.hstack([s(X) for s, X in zip(self.spectra, x)])
