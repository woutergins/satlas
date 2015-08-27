
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