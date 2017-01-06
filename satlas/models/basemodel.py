"""
Implementation of base class for extension to models describing actual data.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@kuleuven.be>
"""
import copy

import lmfit as lm
from satlas.loglikelihood import create_gaussian_priormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy

__all__ = ['load_model']

class SATLASParameters(lm.Parameters):
    _prefix = ''

    def __getitem__(self, key):
        try:
            return super(SATLASParameters, self).__getitem__(key)
        except KeyError:
            try:
                return super(SATLASParameters, self).__getitem__(self._prefix + key)
            except KeyError:
                raise

    def __deepcopy__(self, memo):
        """Parameters deepcopy needs to make sure that
        asteval is available and that all individula
        parameter objects are copied"""
        _pars = SATLASParameters(asteval=None)

        # find the symbols that were added by users, not during construction
        sym_unique = self._asteval.user_defined_symbols()
        unique_symbols = {key: deepcopy(self._asteval.symtable[key], memo)
                              for key in sym_unique}
        _pars._asteval.symtable.update(unique_symbols)

        # we're just about to add a lot of Parameter objects to the newly
        parameter_list = []
        for key, par in self.items():
            if isinstance(par, lm.Parameter):
                param = lm.Parameter(name=par.name,
                                  value=par.value,
                                  min=par.min,
                                  max=par.max)
                param.vary = par.vary
                param.stderr = par.stderr
                param.correl = par.correl
                param.init_value = par.init_value
                param.expr = par.expr
                parameter_list.append(param)

        _pars.add_many(*parameter_list)
        _pars._prefix = self._prefix
        return _pars

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
        self._prefix = ''

    def _set_prefix(self, value):
        for p in self._parameters:
            if len(self._prefix) > 0:
                if self._parameters[p].expr is not None:
                    self._parameters[p].expr = self._parameters[p].expr.replace(self._prefix, value)
            else:
                if self._parameters[p].expr is not None:
                    for P in self._parameters:
                        if P in self._parameters[p].expr:
                            self._parameters[p].expr = self._parameters[p].expr.replace(P, value + P)
        for p in list(self._parameters.keys()):
            if len(self._prefix) > 0:
                self._parameters[p].name = self._parameters[p].name[len(self._prefix):]
            self._parameters[p].name = value + self._parameters[p].name
            self._parameters[value + p] = self._parameters.pop(p)
        self._parameters._prefix = value
        self._prefix = value

    def _add_prefix(self, value):
        for p in self._parameters:
            if self._parameters[p].expr is not None:
                if len(self._prefix) > 0:
                    self._parameters[p].expr = self._parameters[p].expr.replace(self._prefix, value + self._prefix)
                else:
                    for P in self._parameters:
                        if P in self._parameters[p].expr:
                            self._parameters[p].expr = self._parameters[p].expr.replace(P, value + P)
        for p in list(self._parameters.keys()):
            self._parameters[p].name = value + self._parameters[p].name
            self._parameters[value + p] = self._parameters.pop(p)
        self._parameters._prefix = value + self._parameters._prefix
        self._prefix = value + self._prefix

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
        for key in valueDict:
            try:
                self.params[key].value = valueDict[key]
            except KeyError:
                pass

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
            try:
                self.params[k].expr = self._expr[k]
            except KeyError:
                pass

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
            try:
                self.params[k].vary = self._vary[k]
            except KeyError:
                pass

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
            for bound in self._constraints[k].keys():
                try:
                    if bound.lower() == 'min':
                        self.params[k].min = self._constraints[k][bound]
                    elif bound.lower() == 'max':
                        self.params[k].max = self._constraints[k][bound]
                    else:
                        pass
                except KeyError:
                    pass

    def _check_variation(self, par):
        # Make sure the variations in the params are set correctly.
        for key in self._vary.keys():
            try:
                par[key].vary = self._vary[key]
            except KeyError:
                pass

        for key in self._constraints.keys():
            for bound in self._constraints[key]:
                try:
                    if bound.lower() == 'min':
                        par[key].min = self._constraints[key][bound]
                    elif bound.lower() == 'max':
                        par[key].max = self._constraints[key][bound]
                    else:
                        pass
                except KeyError:
                    pass
        for key in self._expr.keys():
            try:
                par[key].expr = self._expr[self._prefix + key]
            except KeyError:
                pass
        return par.copy()

    def get_chisquare_mapping(self):
        return np.array([self._chisquare_mapping[k](self.params[k].value) for k in self._chisquare_mapping.keys()])

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
        if hasattr(self, 'fit_mle'):
            if 'show_correl' not in kwargs:
                kwargs['show_correl'] = False
            print('NDoF: {:d}, Chisquare: {:.8G}, Reduced Chisquare: {:.8G}'.format(self.ndof_mle, self.chisqr_mle, self.redchi_mle))
            print('Akaike Information Criterium: {:.8G}, Bayesian Information Criterium: {:.8G}'.format(self.aic_mle, self.bic_mle))
            if scaled:
                print('Errors scaled with reduced chisquare.')
                par = copy.deepcopy(self.fit_mle)
                for p in par:
                    if par[p].stderr is not None:
                        par[p].stderr *= (self.redchi_mle**0.5)
                print(lm.fit_report(par, **kwargs))
            else:
                print('Errors not scaled with reduced chisquare.')
                print(lm.fit_report(self.fit_mle, **kwargs))
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
            print('NDoF: {:d}, Chisquare: {:.8G}, Reduced Chisquare: {:.8G}'.format(self.ndof_chi, self.chisqr_chi, self.redchi_chi))
            print('Akaike Information Criterium: {:.8G}, Bayesian Information Criterium: {:.8G}'.format(self.aic_chi, self.bic_chi))
            if scaled:
                print('Errors scaled with reduced chisquare.')
                par = copy.deepcopy(self.chisq_res_par)
                for p in par:
                    if par[p].stderr is not None:
                        par[p].stderr *= (self.redchi_chi**0.5)
                print(lm.fit_report(par, **kwargs))
            else:
                print('Errors not scaled with reduced chisquare.')
                print(lm.fit_report(self.chisq_res_par, **kwargs))
        else:
            print('Spectrum has not yet been fitted with this method!')

    def get_goodness_of_fit(self, selection='chisquare'):
        if selection.lower() == 'chisquare':
            return (self.ndof_chi, self.chisqr_chi, self.redchi_chi, self.aic_chi, self.bic_chi)
        elif selection.lower() == 'mle':
            return (self.ndof_mle, self.chisqr_mle, self.redchi_mle, self.aic_mle, self.bic_mle)


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
        elif hasattr(self, 'fit_mle'):
            for key in sorted(self.fit_mle.params.keys()):
                if self.fit_mle.params[key].vary:
                    var.append(self.fit_mle.params[key].value)
                    var_names.append(self.fit_mle.params[key].name)
                    varerr.append(self.fit_mle.params[key].stderr)
        else:
            params = self.params
            for key in sorted(params.keys()):
                if params[key].vary:
                    var.append(params[key].value)
                    var_names.append(params[key].name)
                    varerr.append(None)
        return var_names, var, varerr

    def get_result_frame(self, method='chisquare', selected=False, bounds=False, vary=False, scaled=True):
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
        scaled: boolean, optional
            Sets the uncertainty scaling with the reduced chisquare value. Default to *True*.

        Returns
        -------
        resultframe: DataFrame
            Dateframe with MultiIndex, using the variable names as main column names
            and either two subcolumns for the value and the uncertainty, or
            four subcolumns for the value, uncertainty and bounds."""
        if method.lower() == 'chisquare':
            if scaled:
                p = copy.deepcopy(self.chisq_res_par)
                for par in p:
                    p[par].stderr *= self.redchi_chi**0.5
            else:
                p = self.chisq_res_par
        elif method.lower() == 'mle':
            if scaled:
                p = copy.deepcopy(self.fit_mle)
                for par in p:
                    p[par].stderr *= self.redchi_mle**0.5
            else:
                p = self.fit_mle
        values = p.values()
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
        if method.lower() == 'chisquare':
            result.loc[:, 'Chisquare'] = pd.Series(np.array([self.chisqr_chi]), index=result.index)
            result.loc[:, 'Reduced chisquare'] = pd.Series(np.array([self.redchi_chi]), index=result.index)
            result.loc[:, 'NDoF'] = pd.Series(np.array([self.ndof_chi]), index=result.index)
        else:
            result.loc[:, 'Chisquare'] = pd.Series(np.array([self.chisqr_mle]), index=result.index)
            result.loc[:, 'Reduced chisquare'] = pd.Series(np.array([self.redchi_mle]), index=result.index)
            result.loc[:, 'NDoF'] = pd.Series(np.array([self.ndof_mle]), index=result.index)
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
        if method.lower() == 'chisquare':
            if scaled:
                p = copy.deepcopy(self.chisq_res_par)
                for par in p:
                    p[par].stderr *= self.redchi**0.5
            else:
                p = self.chisq_res_par
        else:
            if scaled:
                p = copy.deepcopy(self.fit_mle)
                for par in p:
                    p[par].stderr *= self.redchi_mle**0.5
            else:
                p = self.fit_mle
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

def load_model(path):
    """Loads the saved BaseModel and returns the reconstructed object.

    Parameters
    ----------
    path: string
        Location of the saved model.

    Returns
    -------
    model: BaseModel
        Saved BaseModel/child class instance."""
    import pickle
    with open(path, 'rb') as f:
        model = pickle.load(f)
    try:
        for m in model.models:
            m._parameters._prefix = m._prefix
    except:
        model._parameters._prefix = model._prefix
    return model
