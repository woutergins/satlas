import satlas.spectrum as hs
import matplotlib.pyplot as plt
import numpy as np
import seaborn


seaborn.set_style('ticks')
seaborn.set_palette('colorblind')

varyDict = {'Bl': False, 'Bu': False, 'Cl': False, 'Cu': False}

# Gather all information
I = 1.0
J = [5.0 / 2, 3.0 / 2]
ABC = [-129.109, -1723.61, 0, 0, 0, 0]

df = 2285.804947
fwhm = [150, 150]
# Create the spectrum
spec1 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000.0, rAmp=True)
# Set a few parameters
spec1.background = 300
# Fix the B and C parameters for spec1
spec1.setVary(varyDict)

# Isomer adding
I = 4
ABC = [-45.3133, -591.94, 0, 0, 0, 0]
df = 2266.436463
spec4 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000, rAmp=True)
spec4.setVary(varyDict)

first = spec1 + spec4
# Set the parameters to be shared between the spectra.
first.shared = ['FWHM', 'FWHML', 'FWHMG']

# Generate the data, add some white noise
xdata = np.linspace(-2000, 8000, 1000)
ydata = first(xdata)
ydata += 3 * np.sqrt(ydata) * np.random.randn(ydata.shape[0])

first.FitToSpectroscopicData(xdata, ydata)
first.DisplayFit(show_correl=False)
iso1, iso2 = first.seperateResponse(xdata)

# Plot the results
fig, ax = plt.subplots(1, 1)
ax.plot(xdata, ydata, 'o', markersize=5)
ax.plot(xdata, first(xdata), lw=2.0)
ax.plot(xdata, iso1, lw=2.0)
ax.plot(xdata, iso2, lw=2.0)

ax.set_xlabel('Frequency (MHz)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)

seaborn.despine(offset=10, trim=True)

plt.tight_layout()
plt.show()
