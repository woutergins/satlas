"""
Implementation of base class for the analysis of hyperfine structure spectra.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['Model']


class BaseModel(object):

    """Abstract baseclass for all spectra, such as :class:`.HFSModel`,
    :class:`.CombinedModel` and :class:`.MultiModel`. For input, see these
    classes."""

    def __init__(self):
        super(Model, self).__init__()
        self.selected = ['Al', 'Au', 'Bl', 'Bu', 'Cl', 'Cu', 'Centroid']

    @property
    def selected(self):
        """When a walk is performed and a triangle plot is created from the data,
        the parameters with one of these strings in their name will be
        displayed. Defaults to the hyperfine parameters and the centroid."""
        return self._selected

    @selected.setter
    def selected(self, value):
        self._selected = value

    def display_mle_fit(self, **kwargs):
        """Give a readable overview of the result of the MLE fitting routine.

        Warning
        -------
        The uncertainty shown is the largest of the asymmetrical errors! Work
        is being done to incorporate asymmetrical errors in the report; for
        now, rely on the correlation plot."""
        if hasattr(self, 'mle_fit'):
            if 'show_correl' not in kwargs:
                kwargs['show_correl'] = False
            print(lm.fit_report(self.mle_fit, **kwargs))
        else:
            print('Model has not yet been fitted with this method!')

    def display_chisquare_fit(self, **kwargs):
        """Display all relevent info of the least-squares fitting routine,
        if this has been performed.

        Parameters
        ----------
        kwargs: misc
            Keywords passed on to :func:`fit_report` from the LMFit package."""
        if hasattr(self, 'chisq_res_par'):
            print('Scaled errors estimated from covariance matrix.')
            print('NDoF: {:d}, Chisquare: {:.8G}, Reduced Chisquare: {:.8G}'.format(self.ndof, self.chisqr, self.redchi))
            print(lm.fit_report(self.chisq_res_par, **kwargs))
        else:
            print('Spectrum has not yet been fitted with this method!')

    def vars(self, selection='any'):
        """Return the variable names, values and estimated error bars for the
        parameters.

        Parameters
        ----------
        selection: string, optional
            Selects if the chisquare ('chisquare' or 'any') or MLE values are
            used. Defaults to 'any'.

        Returns
        -------
        names, values, uncertainties: tuple of lists
            Returns a 3-tuple of lists containing the names of the parameters,
            the values and the estimated uncertainties."""
        var, var_names, varerr = [], [], []
        if hasattr(self, 'chisq_res_par') and (selection.lower() == 'chisquare'
                                               or selection.lower() == 'any'):
            for key in sorted(self.chisq_res_par.keys()):
                if self.chisq_res_par[key].vary:
                    var.append(self.chisq_res_par[key].value)
                    var_names.append(self.chisq_res_par[key].name)
                    varerr.append(self.chisq_res_par[key].stderr)
        elif hasattr(self, 'mle_fit'):
            for key in sorted(self.mle_fit.params.keys()):
                if self.mle_fit.params[key].vary:
                    var.append(self.mle_fit.params[key].value)
                    var_names.append(self.mle_fit.params[key].name)
                    varerr.append(self.mle_fit.params[key].stderr)
        else:
            params = self.params
            for key in sorted(params.keys()):
                if params[key].vary:
                    var.append(params[key].value)
                    var_names.append(params[key].name)
                    varerr.append(None)
        return var_names, var, varerr

    def get_result_frame(self, method='chisquare',
                         selected=False, bounds=False,
                         vary=False):
        """Returns the data from the fit in a pandas DataFrame.

        Parameters
        ----------
        method: str, optional
            Selects which fitresults have to be loaded. Can be 'chisquare' or
            'mle'. Defaults to 'chisquare'.
        selected: boolean, optional
            Selects if only the parameters in :attr:`selected` have to be
            given or not. Defaults to :attr:`False`.
        bounds: boolean, optional
            Selects if the boundary also has to be given. Defaults to
            :attr:`False`.
        vary: boolean, optional
            Selects if only the parameters that have been varied have to
            be supplied. Defaults to :attr:`False`.

        Returns
        -------
        resultframe: DataFrame
            Dateframe with MultiIndex, using the variable names as main column names
            and either two subcolumns for the value and the uncertainty, or
            four subcolumns for the value, uncertainty and bounds."""
        if method.lower() == 'chisquare':
            values = self.chisq_res_par.values()
        elif method.lower() == 'mle':
            values = self.mle_fit.values()
        else:
            raise KeyError
        if selected:
            values = [v for n in self.selected for v in values if n in v.name]
        if vary:
            values = [v for v in values if v.vary]
        if bounds:
            ind = ['Value', 'Uncertainty', 'Upper Bound', 'Lower Bound']
            data = np.array([[p.value, p.stderr, p.max, p.min] for p in values]).flatten()
            columns = [[p.name for pair in zip(values, values) for p in pair],
                       [x for p in values for x in ind]]
        else:
            data = np.array([[p.value, p.stderr] for p in values]).flatten()
            ind = ['Value', 'Uncertainty']
            columns = [[p.name for pair in zip(values, values) for p in pair],
                       [x for p in values for x in ind]]
        columns = pd.MultiIndex.from_tuples(list(zip(*columns)))
        result = pd.DataFrame(data, index=columns).T
        return result
