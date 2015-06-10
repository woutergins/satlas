import matplotlib.pyplot as plt
import numpy as np
import satlas.profiles as p
import seaborn as sns


sns.set_style('ticks', rc={'xtick.direction': 'in', 'ytick.direction': 'in'})
sns.set_palette('colorblind')

x = np.linspace(-6, 6, 1000)
fwhm = 1.0

g = p.Gaussian(fwhm=fwhm)
l = p.Lorentzian(fwhm=fwhm)
b = p.PseudoVoigt(fwhm=fwhm)
e = p.ExtendedVoigt(fwhm=fwhm)
v = p.Voigt(fwhm=fwhm)

prof = [g, l, b, e, v]
names = ['Gaussian', 'Lorentzian',  'PseudoVoigt',
         'ExtendedVoigt', 'Voigt']
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
for lineshape, name in zip(prof, names):
    ax[0].plot(x, lineshape(x), lw=2.0, label=name)

g = p.Gaussian(fwhm=fwhm, ampIsArea=True)
l = p.Lorentzian(fwhm=fwhm, ampIsArea=True)
b = p.PseudoVoigt(fwhm=fwhm, ampIsArea=True)
e = p.ExtendedVoigt(fwhm=fwhm, ampIsArea=True)
v = p.Voigt(fwhm=fwhm, ampIsArea=True)

prof = [g, l, b, e, v]
for lineshape, name in zip(prof, names):
    ax[1].plot(x, lineshape(x), lw=2.0, label=name)
ax[1].legend(loc=0, fontsize=14)

ax[0].set_title('Amplitude is height', fontsize=20)
ax[1].set_title('Amplitude is area', fontsize=20)
plt.tight_layout()
plt.show()
