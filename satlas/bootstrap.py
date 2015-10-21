

def bootstrap_ci(dataframe, kind='basic'):
    """Generate confidence intervals on the 1-sigma level for bootstrapped data
    given in a DataFrame.

    Parameters
    ----------
    dataframe: DataFrame
        DataFrame with the results of each bootstrap fit on a row. If the
        t-method is to be used, a Panel is required, with the data in
        the panel labeled 'data' and the uncertainties labeled 'stderr'
    kind: str, optional
        Selects which method to use: percentile, basic, or t-method (student).

    Returns
    -------
    DataFrame
        Dataframe containing the left and right limits for each column as rows."""
    if isinstance(dataframe, pd.Panel):
        data = dataframe['data']
        stderrs = dataframe['stderr']
        args = (data, stderrs)
    else:
        data = dataframe
        args = (data)

    def percentile(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        left = data.apply(lambda col: np.percentile(col, 15.865), axis=0)
        right = data.apply(lambda col: np.percentile(col, 84.135), axis=0)
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    def basic(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        left = data.apply(lambda col: 2 * col[0] - np.percentile(col[1:],
                                                                 84.135),
                          axis=0)
        right = data.apply(lambda col: 2 * col[0] - np.percentile(col[1:],
                                                                  15.865),
                           axis=0)
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    def student(data, stderrs=None):
        CI = pd.DataFrame(index=['left', 'right'], columns=data.columns)
        R = (data - data.loc[0]) / stderrs
        left = R.apply(lambda col: np.percentile(col[1:], 84.135), axis=0)
        right = R.apply(lambda col: np.percentile(col[1:], 15.865), axis=0)
        left = data.loc[0] - stderrs.loc[0] * left
        right = data.loc[0] - stderrs.loc[0] * right
        CI.loc['left'] = left
        CI.loc['right'] = right
        return CI

    method = {'basic': basic, 'percentile': percentile, 't': student}
    method = method.pop(kind.lower(), basic)
    return method(*args)

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