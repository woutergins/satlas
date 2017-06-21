import satlas as sat
import numpy as np
import matplotlib.pyplot as plt

# Parameter and model initialization
I = 0.5
J = [0, 1]
ABC = [0, 500.0, 0, 0, 0, 0]
centroid = 300.0
shape = 'lorentzian'
fwhm = 100.0
background = [0.3]
scale = 100.0
model = sat.HFSModel(I, J, ABC, centroid, fwhm=fwhm, shape=shape, scale=scale, background_params=background)
boundaries = {'Background0': {'min': 1E-16}}
model.set_boundaries(boundaries)
# Loading dataset
data = np.loadtxt('toy_data.txt')
x = data[:, 0]
y = data[:, 1]
# Chisquare analysis
sat.chisquare_spectroscopic_fit(model, x, y)
model.display_chisquare_fit()
fig2, ax2, cbar = sat.generate_correlation_map(model, x, y, method='chisquare_spectroscopic', filter=['Au', 'Centroid'], distance=3.1)
# Likelihood analysis
sat.likelihood_fit(model, x, y, hessian=True, walking=True, walk_kws={'filename': 'random_walk.h5', 'nsteps': 12000})
model.display_mle_fit()
fig3, ax3, cbar2 = sat.generate_correlation_map(model, x, y, method='mle', distance=3.5)
fig4, ax4, cbar = sat.generate_correlation_plot('random_walk.h5', selection=(5, 100))
fig5, ax5 = sat.generate_walk_plot('random_walk.h5', selection=(0, 5))
