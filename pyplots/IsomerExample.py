import satlas.spectrum as hs
import matplotlib.pyplot as plt
import numpy as np
import seaborn


seaborn.set()
# Gather all information
I = 1.0
J = [5.0 / 2, 3.0 / 2]
ABC = [-129.109, -1723.61, 0, 0, 0, 0]

df = 2285.804947
fwhm = [150, 150]
varyDict = {'Bl': False, 'Bu': False, 'Cl': False, 'Cu': False}
# Create the spectrum
spec1 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000.0, rAmp=True)
# Set a few parameters
spec1.background = 300
spec1.setVary(varyDict)

# Generate a new spectrum for the isomer
fwhm = [100, 100]
df = 3000.0
spec2 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=3000, rAmp=True)
spec2.background = 400

# Isomer adding
I = 4
ABC = [-45.3133, -591.94, 0, 0, 0, 0]
df = 2266.436463
spec4 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000, rAmp=True)
spec4.setVary(varyDict)

first = spec1 + spec4
first.shared = ['FWHM', 'FWHML', 'FWHMG']

# Generate the data, add some white noise
xdata = np.linspace(-2000, 8000, 1000)
ydata = first(xdata)
ydata += 3 * np.sqrt(ydata) * np.random.randn(ydata.shape[0])

# Do the same for another dataset
df = 3000.0
spec42 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                           scale=3000.0, rAmp=True)
spec42.setVary(varyDict)

second = spec2 + spec42
second.shared = ['FWHM', 'FWHML', 'FWHMG']

xdata2 = np.linspace(-2000, 10000, 1000)
ydata2 = second(xdata2)
ydata2 += 3 * np.sqrt(ydata2) * np.random.randn(ydata2.shape[0])

# Create a combined spectrum, and fit at the same time
spec2comb = hs.CombinedSpectrum([first, second])

spec2comb.FitToSpectroscopicData([xdata, xdata2], [ydata, ydata2])
spec2comb.DisplayFit(show_correl=False)

evaluated = spec2comb([xdata, xdata2])
eval1, eval2 = evaluated[:len(xdata)], evaluated[len(xdata):]

evaluated = spec2comb.seperateResponse([xdata, xdata2])
(sep1, sep2), (sep3, sep4) = evaluated

# Plot the results
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(xdata, ydata, 'ro')
ax[0].plot(xdata, eval1, lw=2.0)
ax[0].plot(xdata, sep1, lw=2.0)
ax[0].plot(xdata, sep2, lw=2.0)
ax[1].plot(xdata2, ydata2, 'ro')
ax[1].plot(xdata2, eval2, lw=2.0)
ax[1].plot(xdata2, sep3, lw=2.0)
ax[1].plot(xdata2, sep4, lw=2.0)

ax[1].set_xlabel('Frequency (MHz)', fontsize=16)
ax[0].set_ylabel('Counts', fontsize=16)

plt.tight_layout()
plt.show()
