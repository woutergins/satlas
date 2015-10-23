import matplotlib.pyplot as plt
import numpy as np
import satlas as s

# Gather all information
I = 1.0
J = [0.5, 0.5]
ABC = [500, 200, 0, 0, 0, 0]
df = 5000

np.random.seed(0)
# Create the basemodel
hfs = s.HFSModel(I, J, ABC, df, scale=3000, saturation=10)
hfs.background = 200
constraintsDict = {'Au': {'min': None, 'max': None}}
hfs.set_boundaries(constraintsDict)
# Say which frequencies are scanned
x = np.linspace(4000, 6000, 100)
superx = np.linspace(x.min(), x.max(), 100 * len(x))
# Generate the data, add some noise
y = hfs(x)
y += 3 * np.random.randn(x.shape[0]) * np.sqrt(y)

s.chisquare_spectroscopic_fit(hfs, x, y, monitor=False)
# Print the fit report
hfs.display_chisquare_fit()
hfs.plot(x, y)
# s.likelihood_fit(hfs, x, y, walking=True)
# fig, axes, cbar = s.generate_correlation_plot(hfs.mle_data)
# fig.savefig('walk')
# Plot the result
# fig, ax = plt.subplots(1, 1)
# hfs.plot(show=True, x=x, y=y, no_of_points=1000, legend=r'$\chi^2$', data_legend='Data', bayesian=True, colormap='gnuplot2_r')

# # Example of Maximum Likelihood Estimation (MLE) fitting,
# # along with error calculation using Monte Carlo walking.
# hfs.likelihood_fit(x, y,
#                    walking=False,  # Perform the walk
#                    walkers=50,  # Number of walkers,
#                                 # see the emcee documentation for
#                                 # more on this
#                    nsteps=200,  # Number of steps for each walker
#                    burnin=10.0,  # Defines the percentage of burnin
#                    )
# hfs.plot(ax=ax, show=False, bayesian=True, legend=r'MLE')
# ax.legend(loc=0)
# hfs.display_mle_fit()
# # # plt.tight_layout()
# utils.generate_correlation_plot(hfs.mle_data)
# plt.show()
