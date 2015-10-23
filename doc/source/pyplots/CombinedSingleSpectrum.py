import numpy as np
import satlas as hs

np.random.seed(0)

# Set the general parameters
I = 1.0
J = [5.0 / 2, 3.0 / 2]
ABC = [-130, -1710, 0, 0, 0, 0]
varyDict = {'Bl': False, 'Bu': False, 'Cl': False, 'Cu': False}

# Create the first basemodel
df = 2285
fwhm = [150, 50]
spec1 = hs.HFSModel(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=4000.0, racah_int=True)

# spec1.background = 300
spec1.set_variation(varyDict)


# Create a second basemodel
fwhm = [50, 150]
df = 4100.0
spec2 = hs.HFSModel(I, J, ABC, df, shape='voigt', fwhm=fwhm,
                          scale=3000, racah_int=True)
# spec2.background = 400

# Generate some random data
pos = spec1.locations
xdata = np.linspace(min(pos) - 1000, max(pos) + 1000, 100)
ydata = spec1(xdata)
ydata += 3 * np.sqrt(ydata) * np.random.randn(ydata.shape[0])

# Generate some random data for basemodel 2
pos = spec2.locations
xdata2 = np.linspace(min(pos) - 1000, max(pos) + 1000, 100)
ydata2 = spec2(xdata2)
ydata2 += 3 * np.sqrt(ydata2) * np.random.randn(ydata2.shape[0])
# spec2.df = 4000

# Combine the spectra in a CombinedModel, fit it and report the fit
spec1comb = hs.CombinedModel([spec1, spec2])

hs.chisquare_spectroscopic_fit(spec1comb, [xdata, xdata2], [ydata, ydata2], func=np.sqrt)
spec1comb.display_chisquare_fit(show_correl=False)

# Plot the result
spec1comb.plot_spectroscopic(x=[xdata, xdata2], y=[ydata, ydata2])
