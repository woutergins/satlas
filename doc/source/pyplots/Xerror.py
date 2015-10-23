import matplotlib.pyplot as plt
import satlas.hfsmodel as hs
import numpy as np
# import satlas.utilities as utils
# import seaborn as sns
# import timeit
# import scipy.integrate as integrate
# import scipy.stats as stats
# import scipy.signal as signal

# theta = np.linspace(-3, 3, 100)
# s = 1

# array1 = stats.poisson(7-theta).pmf(1)
# array2 = np.exp(-theta**2/2)

# array1 = np.tile(array1, (2, 1))
# array1 = np.array([stats.poisson(9-theta).pmf(1), stats.poisson(7-theta).pmf(1)])
# array2 = np.tile(array2, (2, 1))
# setup = """
# gc.enable()
# import numpy as np
# import scipy.integrate as integrate
# import scipy.stats as stats
# import scipy.signal as signal

# theta = np.linspace(-5, 5, 10000)
# s = 1

# array1 = stats.poisson(7-theta).pmf(1)
# array2 = np.exp(-theta**2/2)"""

# setup2 = setup + """
# array1 = np.array([stats.poisson(9-theta).pmf(1), stats.poisson(7-theta).pmf(1)])
# array2 = np.tile(array2, (2, 1))
# """

# numpy = timeit.Timer("""np.convolve(array1, array2, 'valid') / (len(theta)/(theta.max()-theta.min()))""", setup=setup)
# scipy = timeit.Timer("""signal.fftconvolve(array1, array2, 'valid') / (len(theta)/(theta.max()-theta.min()))""", setup=setup)
# integral = timeit.Timer("""integrate.quad(lambda theta: stats.poisson(7 - theta).pmf(1) * np.exp(-theta**2/2), theta.min(), theta.max())""", setup=setup)
# homemade = timeit.Timer("""np.fft.irfft(np.fft.rfft(array1) * np.fft.rfft(array2))""", setup=setup)
# homemade2 = timeit.Timer("""np.fft.irfft(np.fft.rfft(array1) * np.fft.rfft(array2))""", setup=setup2)

# print('NumPy: ', numpy.timeit(10000))
# print('SciPy: ', numpy.timeit(10000))
# print('Homemade: ', numpy.timeit(10000))
# print('Homemade (double): ', numpy.timeit(10000))
# print('Integral: ', integral.timeit(100))

# print((np.convolve(array1, array2, 'valid') / (len(theta)/(theta.max()-theta.min()))))
# s1 = np.array(array1.shape)
# s2 = np.array(array2.shape)
# shape = s1+s2-1
# print(shape)
# fslice = tuple([slice(0, int(sz)) for sz in shape])
# print(fslice)
# arr = np.fft.irfft(np.fft.rfft(array1) * np.fft.rfft(array2))

# print(arr[:, -1])  #0.06632327, 0.36833716
# print(signal.convolve(array1.T, array2.T, 'valid'))  #[[-3.78941135]]
# print(np.log(integrate.quad(lambda theta: stats.poisson(7 - theta).pmf(1) * np.exp(-theta**2/2), theta.min(), theta.max())[0]))
# print(np.convolve(array1, array2, 'valid') / (len(theta)/(theta.max()-theta.min())))
# print(integrate.quad(lambda theta: stats.poisson(7 - theta).pmf(1) * np.exp(-theta**2/2), theta.min(), theta.max()))

# sns.set_style('ticks', rc={'xtick.direction': 'in', 'ytick.direction': 'in'})
# sns.set_palette('colorblind')
# Gather all information
I = 1.0
J = [0.5, 0.5]
ABC = [500, 200, 0, 0, 0, 0]
df = 5000

np.random.seed(0)

# Create the basemodel
hfs = hs.HFSModel(I, J, ABC, df, scale=3000)
hfs.background = 200

# Say which frequencies are scanned
x = np.linspace(4000, 6000, 100)
sigma_x = (x[1] - x[0]) / 3  #MHz
used_x = x + np.random.randn(100) * sigma_x

# Generate the data, add some noise
y = hfs(x)
used_y = hfs(used_x)
# y += 3 * np.random.randn(x.shape[0]) * np.sqrt(y)
# Fit to the generated data
# hfs.chisquare_spectroscopic_fit(x, y)
# Print the fit report
# hfs.display_chisquare_fit()

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'o', label='Actual data')
ax.plot(x, used_y, 'o', label='Sampled data')
# ax.plot(x, used_y)
# ax.plot(x, y, 'o', markersize=5)
# ax.plot(superx, hfs(superx), lw=3.0, label=r'$\chi^2$')
# ax.set_xlabel('Frequency (MHz)', fontsize=16)
# ax.set_ylabel('Counts', fontsize=16)

# Example of Maximum Likelihood Estimation (MLE) fitting,
# along with error calculation using Monte Carlo walking.
import time
start = time.time()
hfs.likelihood_fit(x, used_y, xerr=0, vary_sigma=False,
                   walking=False,  # Perform the walk
                   walkers=50,  # Number of walkers,
                                # see the emcee documentation for
                                # more on this
                   nsteps=200,  # Number of steps for each walker
                   burnin=10.0,  # Defines the percentage of burnin
                   )
hfs.display_mle_fit()
print(time.time()-start)
hfs.plot(ax=ax, show=False, legend=r'No x-error')
start = time.time()
hfs.likelihood_fit(x, used_y, xerr=sigma_x, vary_sigma=True,
                   walking=False,  # Perform the walk
                   walkers=50,  # Number of walkers,
                                # see the emcee documentation for
                                # more on this
                   nsteps=200,  # Number of steps for each walker
                   burnin=10.0,  # Defines the percentage of burnin
                   )
print(time.time()-start)
hfs.display_mle_fit()
hfs.plot(ax=ax, show=False, legend=r'x-error')
ax.legend(loc=0)
# plt.tight_layout()
# utils.generate_correlation_plot(hfs.mle_data)
plt.show()
