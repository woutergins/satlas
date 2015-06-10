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

# Add an isomer to the spectrum
# First, create the spectrum itself
I = 4
ABC = [-45, -591, 0, 0, 0, 0]
df = 2266
spec4 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000, racah_int=True)
spec4.set_variation(varyDict)

# Add the spectra together
first = spec1 + spec4
# Set the parameters to be shared between the spectra.
first.shared = ['FWHM', 'FWHML', 'FWHMG']

# Generate some random data
xdata = np.linspace(-2000, 8000, 1000)
ydata = first(xdata)
ydata += 3 * np.sqrt(ydata) * np.random.randn(ydata.shape[0])

first.chisquare_spectroscopic_fit(xdata, ydata)
first.display_chisquare_fit(show_correl=False)
iso1, iso2 = first.seperate_response(xdata)

# Plot the results
fig, ax = plt.subplots(1, 1)
ax.plot(xdata, ydata, 'o', markersize=3)
ax.plot(xdata, first(xdata), lw=3.0)
ax.plot(xdata, iso1, lw=3.0)
ax.plot(xdata, iso2, lw=3.0)

ax.set_xlabel('Frequency (MHz)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)

plt.tight_layout()
plt.show()
