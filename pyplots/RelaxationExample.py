from satlas.relaxation import KorringaRelaxation
from satlas.utilities import ReleaseCurve
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn


seaborn.set_style('ticks')
seaborn.set_palette('colorblind')
# Needed for interaction temperature
g = 0.4213
Bfield = 1000.0  # Gauss
mu_N = 0.762259372  # kHz/gauss
k_B = 8.6173324 * 0.00001  # eV/K (Boltzmann constant)
h = 4.13567 * (10 ** (-15))  # eV/s

# Parameters for simulator and release curve setup
t1_2 = 1.77
I = 1.5
Tint = h * g * Bfield * mu_N * 1000 / k_B  # interaction temparture
T = 20.0
pulsedelay = 2.0

# Create a release curve
RC = ReleaseCurve(delay=pulsedelay)
# Create the simulator
simulator = KorringaRelaxation(I, t1_2, T, Tint, 1.0, implant=RC)

# Select the time to simulate
t = np.linspace(0, 20 * t1_2, 2000)

# Select a range of spin-lattice relaxation times
T1 = np.linspace(0.1, 3.4, 21)
# Preallocate the solutions array
sol = np.zeros((T1.shape[0], t.shape[0]))
# Simulate the system for each value of the spin-lattice relaxation time.
# Save the polarization and activity in function of time.
for i, entry in enumerate(T1):
    simulator.T1 = entry
    sol[i, :], pops = simulator.Simulate(t)
# Trick to give a range of colors with a colorbar instead of seperate legend entries.
t_temp, T1_temp = np.meshgrid(t, T1)
ticks = np.linspace(T1.min(), T1.max(), 21)
fig = plt.figure()
ax = fig.add_subplot(2, 1, 2)
cf = ax.contourf(t_temp, sol, T1_temp, ticks, cmap=cm.spectral)
ax.cla()

# Plot different contour lines, corresponding with the ticks of the colorbar.
ax.contour(t_temp, sol, T1_temp, ticks, cmap=cm.spectral, linewidths=2.0)
cbar = fig.colorbar(cf, ticks=ticks[::4], orientation='vertical')
cbar.set_label('T$_1$ [s]', fontsize=16)
ax.set_xlabel('Time [s]', fontsize=16)
ax.set_ylabel('Polarization [%]', fontsize=16)
ax.set_title('Pulse delay {:.3g} s'.format(pulsedelay), fontsize=18)
ax.tick_params(labelsize=14)
ax.grid(True, alpha=0.4)

# Add in the expected activity.
ax2 = fig.add_subplot(2, 1, 1)
ax2.semilogy(t, pops, lw=2)
ax2.set_xlabel('Time [s]', fontsize=16)
ax2.set_ylabel('Activity [Hz]', fontsize=16)
ax2.set_title('Expected activity', fontsize=18)
ax2.tick_params(labelsize=14)
ax2.grid(True, alpha=0.4)

seaborn.despine(ax=ax, offset=10, trim=False)
seaborn.despine(ax=ax2, offset=10, trim=False)
plt.tight_layout()
plt.show()
