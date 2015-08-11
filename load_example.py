import satlas.utilities as utils
import pickle
import numpy as np

from analysisIO import load

b = load('test.analysis')
b['no_1'].display_chisquare_fit()
b.plot_spectrum('no_1')

fig = utils.generate_correlation_plot(b['no_1'].mle_data, filter=b['no_1'].selected)
fig.show()