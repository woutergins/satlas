from .analysis import Analysis
from .unit import DataUnit
from .singlespectrum import SingleSpectrum
from .combinedspectrum import CombinedSpectrum
from .isomerspectrum import IsomerSpectrum
from .spectrum import Spectrum


## Example approach function
## Simple case: if there is only one file in the unit, then
## make a SingleSpectrum with some parameters. If there are 
## multiple (e.g. data spread accross files), make a 
## CombinedSpectrum
def my_approach(data_unit):
    I = 1.5
    J = [0.5,1.5]
    ABC = [1000,100,2,2,0,0]
    df = 600

    if len(data_unit.files) == 1:
        spectrum = SingleSpectrum(I,J,ABC,df)
    else:
        spectra = []
        for i in range(len(data_unit.files)):
            spectra.append(SingleSpectrum(I,J,ABC,df))
        spectrum = CombinedSpectrum(spectra)

## Another example
def my_2nd_approach(data_unit):
    if data_unit.name == 'this':
        I = 1.5
        J = [0.5,1.5]
        ABC = [1000,100,2,2,0,0]
        df = 600
    else:
        I = 2.5
        J = [0.5,1.5]
        ABC = [1000,100,2,2,0,0]
        df = 600

    return SingleSpectrum(I,J,ABC,df)


## Fancier way of making an approach function
def generate_approach(**kwargs):
    """
    Returns an approach function. This function takes a DataUnit
    as input and returns an Spectrum object appropriate for it's 
    analysis.
    """
    def approach(data_unit):

        if len(data_unit.files) == 1:
            spectrum = SingleSpectrum(**kwargs)
        else:
            spectra = []
            for i in range(len(data_unit.files)):
                spectra.append(SingleSpectrum(**kwargs))
            spectrum = CombinedSpectrum(spectra)

        return spectrum

    return approach


a = Analysis()

for i in range(4):
    a.add_data_unit(DataUnit(name='unit1',files=['test1']))

function = generate_approach(I = 1.5,J = [0.5,1.5],
        ABC = [1000,100,2,2,0,0],df = 600)

a.approaches['first'] = function
a.analyse_chisq(approach='first')