import satlas as s
import numpy as np
np.random.seed(0)  #Ensure the same random numbers each time

# Create the first basemodel
I = 1.0
J = [1.0, 2.0]

ABC = [100, 200, 100, 200, 0, 0]
fwhm = [10, 10]
centroid = 500
scale = 100

basemodel = s.HFSModel(I, J, ABC, centroid, fwhm=fwhm, scale=scale, background=10)
basemodel2 = s.HFSModel(I, J, ABC, centroid, fwhm=fwhm, scale=scale, background=10)

frequency_range = (min(basemodel.locations) - 100, max(basemodel.locations) + 100)
frequency_range = np.linspace(frequency_range[0], frequency_range[1], 200)

data = basemodel(frequency_range) + basemodel(frequency_range)**0.5 * np.random.randn(len(frequency_range))

# success, message = s.likelihood_fit(basemodel, frequency_range, data, walking=True, walk_kws={'nsteps': 20})
# frame_chi = basemodel.get_result_frame(method='chisquare')
# frame_mle = basemodel.get_result_frame(method='mle')
# frame = s.concat_results([frame_chi, frame_mle], index=['chisquare', 'poisson'])
# print(frame)
sp = s.CombinedModel([basemodel, basemodel2])
freq = [frequency_range, frequency_range]
dat = [data, data]
success, message = s.chisquare_spectroscopic_fit(sp, freq, dat, pearson=False)
# fig, axes, cbar = s.utilities.generate_correlation_map(sp, freq, dat,
#                                                       filter=['Al', 'Bl', 'Centroid'],
#                                                       resolution_diag=20, resolution_map=10,
#                                                       fit_kws={'pearson': False})
# success, message = s.chisquare_spectroscopic_fit(basemodel, frequency_range, data, pearson=False)
import matplotlib.pyplot as plt
# fig.savefig('combined')
# success, message = s.likelihood_fit(basemodel, frequency_range, data)
success, message = s.chisquare_spectroscopic_fit(basemodel, frequency_range, data, pearson=False)
# basemodel.display_chisquare_fit(show_correl=False)
# fig, axes, cbar = s.utilities.generate_correlation_map(basemodel, frequency_range, data,
#                                                       filter=['Al', 'Au', 'Bl', 'Bu', 'Centroid'],
#                                                       resolution_diag=20, resolution_map=20,
#                                                       fit_kws={'pearson': True})
# fig, axes, cbar = s.utilities.generate_correlation_map(basemodel, frequency_range, data,
#                                                       filter=['Al', 'Au', 'Bl', 'Bu', 'Centroid'],
#                                                       resolution_diag=20, resolution_map=10,
#                                                       fit_kws={'pearson': False})
fig, axes, cbar = s.utilities.generate_correlation_map(basemodel, frequency_range, data,
                                                      method='mle',
                                                      filter=['Al', 'Au', 'Bl', 'Bu', 'Centroid'],
                                                      resolution_diag=20, resolution_map=10,
                                                      fit_kws={'walking': False})
fig.savefig('single')
plt.show()
# fig, axes, cbar = s.utilities._make_axes_grid(4)
# plt.show()
