import satlas.spectrum as hs
import matplotlib.pyplot as plt
import numpy as np
import seaborn


seaborn.set()
I = 1.0
J = [5.0 / 2, 3.0 / 2]
ABC = [-129.109, -1723.61, 0, 0, 0, 0]

df = 2285.804947
fwhm = [150, 150]
varyDict = {'Bl': False, 'Bu': False, 'Cl': False, 'Cu': False}
spec1 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000.0, rAmp=True)

spec1.background = 300
spec1.setVary(varyDict)

xdata = np.linspace(min(spec1.mu) - 1000, max(spec1.mu) + 1000, 1000)
ydata = spec1(xdata)
ydata += 3 * np.sqrt(ydata) * np.random.randn(ydata.shape[0])

fwhm = [100, 100]
df = 3000.0
spec2 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=3000, rAmp=True)
spec2.background = 400

xdata2 = np.linspace(min(spec2.mu) - 1000, max(spec2.mu) + 1000, 1000)
ydata2 = spec2(xdata2)
ydata2 += 3 * np.sqrt(ydata2) * np.random.randn(ydata2.shape[0])
spec2.df = 2950.0
spec1comb = hs.CombinedSpectrum([spec1, spec2])

spec1comb.FitToSpectroscopicData([xdata, xdata2], [ydata, ydata2])
spec1comb.DisplayFit(show_correl=False)

evaluated = spec1comb([xdata, xdata2])
eval1, eval2 = evaluated[:len(xdata)], evaluated[len(xdata):]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(xdata, ydata, 'ro')
ax[0].plot(xdata, eval1, lw=2.0)
ax[1].plot(xdata2, ydata2, 'ro')
ax[1].plot(xdata2, eval2, lw=2.0)

ax[1].set_xlabel('Frequency (MHz)', fontsize=16)
ax[0].set_ylabel('Counts', fontsize=16)
plt.tight_layout()
plt.show()
