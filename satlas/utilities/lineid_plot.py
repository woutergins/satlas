"""Automatic placement of labels for features in a plot.

Depends on Numpy and Matplotlib.
"""
from __future__ import division, print_function

__version__ = "0.2.1"
__author__ = "Prasanth Nair"




if __name__ == "__main__":
    wave = 1240 + np.arange(300) * 0.1
    flux = np.random.normal(size=300)
    line_wave = [1242.80, 1260.42, 1264.74, 1265.00, 1265.2, 1265.3, 1265.35]
    line_flux = np.interp(line_wave, wave, flux)
    line_label1 = ['N V', 'Si II', 'Si II', 'Si II', 'Si II', 'Si II', 'Si II']
    label1_size = np.array([12, 12, 12, 12, 12, 12, 12])
    plot_line_ids(wave, flux, line_wave, line_label1, label1_size)
    plt.show()
