import matplotlib.pyplot as plt
import numpy as np
import satlas.spectrum as hs
import seaborn as sns


sns.set_style('ticks', rc={'xtick.direction': 'in', 'ytick.direction': 'in'})
sns.set_palette('colorblind')

# Set the general parameters
I = 1.0
J = [5.0 / 2, 3.0 / 2]
ABC = [-130, -1710, 0, 0, 0, 0]
varyDict = {'Bl': False, 'Bu': False, 'Cl': False, 'Cu': False}

# Create the first spectrum
df = 2285
fwhm = [150, 50]
spec1 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000.0, racah_int=True)

spec1.background = 300
spec1.set_variation(varyDict)

# Generate some random data
xdata = np.linspace(min(spec1.mu) - 1000, max(spec1.mu) + 1000, 1000)
ydata = spec1(xdata)
ydata += 3 * np.sqrt(ydata) * np.random.randn(ydata.shape[0])

# Create a second spectrum
fwhm = [50, 150]
df = 4100.0
spec2 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=3000, racah_int=True)
spec2.background = 400

# Generate some random data for spectrum 2
xdata2 = np.linspace(min(spec2.mu) - 1000, max(spec2.mu) + 1000, 1000)
ydata2 = spec2(xdata2)
ydata2 += 3 * np.sqrt(ydata2) * np.random.randn(ydata2.shape[0])
spec2.df = 4000

# Combine the spectra in a CombinedSpectrum, fit it and report the fit
spec1comb = hs.CombinedSpectrum([spec1, spec2])

spec1comb.chisquare_spectroscopic_fit([xdata, xdata2], [ydata, ydata2])
spec1comb.display_chisquare_fit(show_correl=False)

# Plot the result
eval1, eval2 = spec1comb.seperate_response([xdata, xdata2])

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(xdata, ydata, 'o', markersize=3)
ax[0].plot(xdata, eval1, lw=3.0)
ax[1].plot(xdata2, ydata2, 'o', markersize=3)
ax[1].plot(xdata2, eval2, lw=3.0)

ax[1].set_xlabel('Frequency (MHz)', fontsize=16)
ax[0].set_ylabel('Counts', fontsize=16)

plt.tight_layout()
plt.show()
