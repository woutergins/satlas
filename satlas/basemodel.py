"""
Implementation of base class for extension to models describing actual data.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@fys.kuleuven.be>
"""
import copy

from . import lmfit as lm
from .loglikelihood import create_gaussian_priormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['Model']


class BaseModel(object):

    """Abstract baseclass for all models. For input, see these
    classes."""

    def __init__(self):
        super(BaseModel, self).__init__()
        self._expr = {}
        self._vary = {}
        self._constraints = {}
        self._params = None
        self._lnprior_mapping = {}
        self._chisquare_mapping = {}

    def set_literature_values(self, literatureDict):
        """Sets the lnprior and chisquare mapping to handle the given
        literature values and uncertainties.

        Parameters
        ----------
        literatureDict: dictionary
            Dictionary with the parameter names as keys. Each
            key should correspond to a dictionary containing
            a 'value' and 'uncertainty' key."""
        priorDict = {}
        chisquareDict = {}
        for k in literatureDict:
            v, u = literatureDict[k]['value'], literatureDict[k]['uncertainty']

            prior = create_gaussian_priormap(v, u)
            priorDict[k] = prior

            chisquare = lambda value: (value - v) / u
            chisquareDict[k] = chisquare
        self.set_chisquare_mapping(chisquareDict)
        self.set_lnprior_mapping(priorDict)

    def set_lnprior_mapping(self, mappingDict):
        """Sets the prior mapping for the different parameters.
        This will affect likelihood fits.

        Parameters
        ----------
        mappingDict: dictionary
            Dictionary containing the functions that give the
            prior for the given parameter value. Use the parameter
            names as keys."""
        for k in mappingDict.keys():
            self._lnprior_mapping[k] = copy.deepcopy(mappingDict[k])

    def set_chisquare_mapping(self, mappingDict):
        """Sets the prior mapping for the different parameters.
        This will affect chisquare fits.

        Parameters
        ----------
        mappingDict: dictionary
            Dictionary containing the functions that give the
            prior for the given parameter value. Use the parameter
            names as keys."""
        for k in mappingDict.keys():
            self._chisquare_mapping[k] = copy.deepcopy(mappingDict[k])

    def set_value(self, valueDict):
        """Sets the value of the given parameters to the given values.

        Parameters
        ----------
        valueDict: dictionary
            Dictionary containing the values for the parameters
            with the parameter names as keys."""
        par = self.params
        for key in valueDict:
            par[key].value = copy.deepcopy(valueDict[key])
        self.params = par

    def set_expr(self, exprDict):
        """Sets the expression of the selected parameters
        to the given expressions.

        Parameters
        ----------
        exprDict: dictionary
            Dictionary containing the expressions for the parameters
            with the parameter names as keys."""
        for k in exprDict.keys():
            self._expr[k] = copy.deepcopy(exprDict[k])

    def set_variation(self, varyDict):
        """Sets the variation of the fitparameters as supplied in the
        dictionary.

        Parameters
        ----------
        varyDict: dictionary
            A dictionary containing 'key: True/False' mappings with
            the parameter names as keys."""
        for k in varyDict.keys():
            self._vary[k] = copy.deepcopy(varyDict[k])

    def set_boundaries(self, boundaryDict):
        """Sets the boundaries of the fitparameters as supplied in the
        dictionary.

        Parameters
        ----------
        boundaryDict: dictionary
            A dictionary containing "key: {'min': value, 'max': value}" mappings.
            A value of *None* or a missing key gives no boundary
            in that direction. The parameter names have to be used as keys."""
        for k in boundaryDict.keys():
            self._constraints[k] = copy.deepcopy(boundaryDict[k])

    def _check_variation(self, par):
        # Make sure the variations in the params are set correctly.
        for key in self._vary.keys():
            if key in par.keys():
                par[key].vary = self._vary[key]

        for key in self._constraints.keys():
            for bound in self._constraints[key]:
                if bound.lower() == 'min':
                    par[key].min = self._constraints[key][bound]
                elif bound.lower() == 'max':
                    par[key].max = self._constraints[key][bound]
                else:
                    pass
        return par

    def get_chisquare_mapping(self):
        return np.array([self._chisquare_mapping[k](self._params[k].value) for k in self._chisquare_mapping.keys()])

    def get_lnprior_mapping(self, params):
        # Check if the parameter values are within the acceptable range.
        for key in params.keys():
            par = params[key]
            if par.vary:
                try:
                    leftbound, rightbound = (par.priormin,
                                             par.priormax)
                except AttributeError:
                    leftbound, rightbound = par.min, par.max
                leftbound = -np.inf if leftbound is None else leftbound
                rightbound = np.inf if rightbound is None else rightbound
                if not leftbound < par.value < rightbound:
                    return -np.inf
        # If defined, calculate the lnprior for each seperate parameter
        return_value = 1.0
        for key in self._lnprior_mapping.keys():
            return_value += self._lnprior_mapping[key](params[key].value)
        return return_value

    def display_mle_fit(self, scaled=False, **kwargs):
        """Give a readable overview of the result of the MLE fitting routine.

        Warning
        -------
        The uncertainty shown is the largest of the asymmetrical errors! Work
        is being done to incorporate asymmetrical errors in the report; for
        now, rely on the correlation plot."""
        if hasattr(self, 'mle_fit'):
            if 'show_correl' not in kwargs:
                kwargs['show_correl'] = False
            print('Chisquare: {:.8G}, Reduced Chisquare: {:.8G}'.format(self.mle_chisqr, self.mle_redchi))
            if scaled:
                print('Errors scaled with reduced chisquare.')
                par = copy.deepcopy(self.mle_fit)
                for p in par:
                    if par[p].stderr is not None:
                        par[p].stderr *= (self.mle_redchi**0.5)
                print(lm.fit_report(par, **kwargs))
            else:
                print('Errors not scaled with reduced chisquare.')
                print(lm.fit_report(self.mle_fit, **kwargs))
        else:
            print('Model has not yet been fitted with this method!')

    def display_chisquare_fit(self, scaled=False, **kwargs):
        """Display all relevent info of the least-squares fitting routine,
        if this has been performed.

        Parameters
        ----------
        kwargs: misc
            Keywords passed on to :func:`fit_report` from the LMFit package."""
        if hasattr(self, 'chisq_res_par'):
            print('NDoF: {:d}, Chisquare: {:.8G}, Reduced Chisquare: {:.8G}'.format(self.ndof, self.chisqr, self.redchi))
            if scaled:
                print('Errors scaled with reduced chisquare.')
                par = copy.deepcopy(self.chisq_res_par)
                for p in par:
                    if par[p].stderr is not None:
                        par[p].stderr *= (self.redchi**0.5)
                print(lm.fit_report(par, **kwargs))
            else:
                print('Errors not scaled with reduced chisquare.')
                print(lm.fit_report(self.chisq_res_par, **kwargs))
        else:
            print('Spectrum has not yet been fitted with this method!')

    def get_result(self, selection='any'):
        """Return the variable names, values and estimated error bars for the
        parameters as seperate lists.

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

    def get_result_frame(self, method='chisquare', selected=False, bounds=False, vary=False):
        """Returns the data from the fit in a pandas DataFrame.

        Parameters
        ----------
        method: str, optional
            Selects which fitresults have to be loaded. Can be 'chisquare' or
            'mle'. Defaults to 'chisquare'.
        selected: list of strings, optional
            Selects the parameters that have any string in the list
            as a substring in their name. Set to *None* to select
            all parameters. Defaults to *None*.
        bounds: boolean, optional
            Selects if the boundary also has to be given. Defaults to
            *False*.
        vary: boolean, optional
            Selects if only the parameters that have been varied have to
            be supplied. Defaults to *False*.

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

    def get_result_dict(self, method='chisquare', scaled=True):
        """Returns the fitted parameters in a dictionary of the form {name: [value, uncertainty]}.

        Parameters
        ----------
        method: {'chisquare', 'mle'}
            Selects which parameters have to be returned.
        scaled: boolean
            Selects if, in case of chisquare parameters, the uncertainty
            has to be scaled by sqrt(reduced_chisquare). Defaults to *True*.

        Returns
        -------
        dict
            Dictionary of the form described above."""
        if scaled:
            try:
                par = copy.deepcopy(self.chisq_res_par)
                for p in par:
                    par[p].stderr *= self.redchi**0.5
            except:
                pass
        if method.lower() == 'chisquare':
            if scaled:
                p = par
            else:
                p = self.chisq_res_par
        else:
            p = self.mle_fit
        returnDict = {P: [p[P].value, p[P].stderr] for P in p}
        return returnDict

    def save(self, path):
        """Saves the current spectrum, including the results of the fitting
        and the parameters, to the specified file.

        Parameters
        ----------
        path: string
            Name of the file to be created."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    #######################################
    #      METHODS CALLED BY FITTING      #
    #######################################

    def _sanitize_input(self, x, y, yerr=None):
        return x, y, yerr

    def seperate_response(self, x):
        """Wraps the output of the :meth:`__call__` in a list, for
        ease of coding in the fitting routines."""
        return [self(x)]

    def __add__(self, other):
        """Add two spectra together to get an :class:`.SumModel`.

        Parameters
        ----------
        other: BaseModel
            Other spectrum to add.

        Returns
        -------
        SumModel
            A SumModel combining both spectra."""
        from .summodel import SumModel
        if isinstance(other, SumModel):
            l = [self] + other.models
        else:
            l = [self, other]
        return SumModel(l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __call__(self, x):
        raise NotImplementedError("Method has to be implemented in subclass!")
