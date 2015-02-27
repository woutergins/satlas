import satlas.spectrum as hs
import matplotlib.pyplot as plt
import numpy as np
import seaborn  # Makes the plots prettier, not really necessary


seaborn.set()
# Gather all information
I = 0
J = [0.5, 0.5]
ABC = [0, 0, 0, 0, 0, 0]
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
ax.plot(x, y, 'ro')
ax.plot(x, hfs(x), lw=2.0)
ax.set_xlabel('Frequency (MHz)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)
plt.show()
