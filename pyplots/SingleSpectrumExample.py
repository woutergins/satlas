import satlas.spectrum as hs
import matplotlib.pyplot as plt
import numpy as np
import seaborn


seaborn.set_style('ticks')
seaborn.set_palette('colorblind')
# Gather all information
I = 0.5
J = [0.5, 0.5]
ABC = [500, 200, 0, 0, 0, 0]
df = 5000

# Create the spectrum
hfs = hs.SingleSpectrum(I, J, ABC, df, scale=3000)
hfs.background = 200

# Say which frequencies are scanned
x = np.linspace(4000, 6000, 100)
# Generate the data, add some noise
y = hfs(x)
y += 3 * np.random.randn(x.shape[0]) * np.sqrt(y)
# Fit to the generated data
hfs.FitToSpectroscopicData(x, y)
# Print the fit report
hfs.DisplayFit()
e = hfs.seperateResponse(x)

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'o')
ax.plot(x, hfs(x), lw=2.0, label=r'$\chi^2$')
ax.set_xlabel('Frequency (MHz)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)
seaborn.despine(offset=10, trim=True)

# Example of Maximum Likelihood Estimation (MLE) fitting,
# along with error calculation using Monte Carlo walking.
hfs.showAll = True  # Show all triangle plots
hfs.LikelihoodSpectroscopicFit(x, y,
                               walking=True,  # Perform the walk
                               walkers=100,  # Number of walkers,
                                             # see the emcee documentation for
                                             # more on this
                               nsteps=2000,  # Number of steps for each walker
                               burnin=10.0,  # Defines the percentage of burnin
                               showTriangle=True  # Show an awesome result plot
                               )
ax.plot(x, hfs(x), lw=2.0, label='MLE')
ax.legend(loc=0)
plt.show()
