from analysis import Analysis
import pylab as pl
import numpy as np

a = Analysis('test')
a.addSingleSpectrum(name='no_1', I=1.5, J=[0.5,1.5], ABC=[1000,100,2,2,0,0], df=0)
a['no_1'].scale = 10

x = np.linspace(-1500,1500,1000)
y = a['no_1'](x) 
y += np.random.normal(np.sqrt(y))

fig,ax = a['no_1'].plot_spectroscopic(x=x,y=y)
fig.show()

a['no_1'].fit_spectroscopic(x=x,y=y)

import pickle

dump = pickle.dumps(a['no_1'])

b = pickle.loads(dump)
b.plot()










##### stuff to do with the analysis class
## have a save function
## have overview plotting function
## have fitting function for chisqr and mle that uses the data from the file
## support for multiple files -> one spectrum and multiple files -> combinedSpectrum