import satlas.spectrum as hs
import matplotlib.pyplot as plt
import numpy as np
import seaborn


seaborn.set_style('ticks')
seaborn.set_palette('colorblind')
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
df = 4100.0
spec2 = hs.SingleSpectrum(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=3000, rAmp=True)
spec2.background = 400

xdata2 = np.linspace(min(spec2.mu) - 1000, max(spec2.mu) + 1000, 1000)
ydata2 = spec2(xdata2)
ydata2 += 3 * np.sqrt(ydata2) * np.random.randn(ydata2.shape[0])
spec2.df = 4000
spec1comb = hs.CombinedSpectrum([spec1, spec2])

spec1comb.FitToSpectroscopicData([xdata, xdata2], [ydata, ydata2])
spec1comb.DisplayFit(show_correl=False)

eval1, eval2 = spec1comb.seperateResponse([xdata, xdata2])

xdata3 = np.linspace(-5000, -4500, 20)
xdata3 = [xdata3, xdata3]
y2 = spec1comb(xdata3)
x, y2, _ = spec1comb.sanitizeFitInput(xdata3, y2, np.sqrt(y2))

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(xdata, ydata, 'o')
ax[0].plot(xdata, eval1, lw=2.0)
ax[1].plot(xdata2, ydata2, 'o')
ax[1].plot(xdata2, eval2, lw=2.0)

ax[1].set_xlabel('Frequency (MHz)', fontsize=16)
ax[0].set_ylabel('Counts', fontsize=16)

seaborn.despine(ax=ax[1], offset=10, trim=True)
seaborn.despine(ax=ax[0], offset=10, trim=True, bottom=True)
plt.tight_layout()
plt.show()
