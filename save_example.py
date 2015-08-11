from analysis import Analysis
from satlas.singlespectrum import SingleSpectrum
from satlas.combinedspectrum import CombinedSpectrum
import pylab as pl
import numpy as np

Is=[1.5,1.5]
Js=[[0.5,1.5],[0.5,1.5]]
ABCs=[[1000,100,2,2,0,0],[900,100,2,2,0,0]]
dfs=[-600,0]

spectra = []
for I,J,ABC,df in zip(Is,Js,ABCs,dfs):
    one = SingleSpectrum(I,J,ABC,df)
    two = SingleSpectrum(I+1,J,[i+100 for i in ABC],df)
    one.scale = 50
    one.background = 5
    two.scale = 50
    two.background = 5

    spectra.append(one+two)

# generate the data and save on disk
x1 = np.linspace(-4000,4000,1000)
y1 = spectra[0](x1) 
y1 += np.abs(np.random.randn(x1.shape[0]) * np.sqrt(y1))

x2 = np.linspace(-4000,4000,1000)
y2 = spectra[1](x2) 
y2 += np.abs(np.random.randn(x2.shape[0]) * np.sqrt(y2))

x = [x1,x2]
y = [y1,y2]

np.savetxt('data1.csv', np.column_stack((x1,y1)))
np.savetxt('data2.csv', np.column_stack((x2,y2)))


a = Analysis(['data1.csv', 'data2.csv'])
spectrum = CombinedSpectrum(spectra)
a.add_spectrum('no_1', spectrum)


## plot and analyse
a.plot_spectrum('no_1')

## second analysis approach
a.add_spectrum('no_2', copy_name = 'no_1')
a['no_2'].shared = ['Bl','Bu','Cl','Cu','Offset']

result = a.analyse_chisq('no_1')
a['no_1'].display_chisquare_fit()
a.plot_spectrum('no_1',show = False)

result = a.analyse_chisq('no_2')
a['no_2'].display_chisquare_fit()
a.plot_spectrum('no_2',show = True)

from analysisIO import save
save(a,'test')
