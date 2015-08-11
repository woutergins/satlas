from analysis import Analysis
from singlespectrum import SingleSpectrum
from combinedspectrum import CombinedSpectrum
import pylab as pl
import numpy as np

Is=[1.5,1.5]
Js=[[0.5,1.5],[0.5,1.5]]
ABCs=[[1000,100,2,2,0,0],[1000,100,2,2,0,0]]
dfs=[-600,0]
spectra = []
for I,J,ABC,df in zip(Is,Js,ABCs,dfs):
    one = SingleSpectrum(I,J,ABC,df)
    two = SingleSpectrum(I+1,J,[i+100 for i in ABC],df)
    one.scale = 10
    one.background = 5
    two.scale = 10
    two.background = 5

    spectra.append(one+two)

spectrum = CombinedSpectrum(spectra)

a = Analysis(['data1.csv', 'data2.csv'])
a['no_1'] = spectrum


a.plot_spectrum('no_1')
result = a.analyse_chisq('no_1')
a.analyse_mle('no_1')

a.plot_spectrum('no_1')

from analysisIO import save

save(a,'test')

