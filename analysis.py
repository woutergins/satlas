from satlas.singlespectrum import SingleSpectrum
from satlas.combinedspectrum import CombinedSpectrum
from copy import deepcopy

class Analysis(object):
    """
    Class that contains references to (several) spectra that can 
    be tied to data for later analysis. Notes can also be added
    at will.

    parameters: 
    None

    Attributes:
    _dataPath: string
        Location of the data
    data_loaded: boolean
        Signifies if the data is loaded
    notes: list (of lists)
        List of notes for each spectrum - an empty
        list is appended to this list whenever a new
        spectrum is added
    """
    def __init__(self,path):
        super(Analysis,self).__init__()
        self._dataPath = ''

        self.data_loaded = False
        self.spectra = {}
        self.spectra_params = {}
        self.notes = {}

        self.dataPath = path

    ### Properties and setters
    @property
    def dataPath(self):
        return self._dataPath

    @dataPath.setter
    def dataPath(self,path):
        self._dataPath = path
        self.loadData()

    ### Methods
    def loadData(self):
        ## load from path
        # load

        self.data_loaded = True

    def analyse(self):
        pass

    def addSingleSpectrum(self,name,to_copy = None,
            isomers=0,**kwargs):

        if to_copy is not None:
            if not to_copy in self.keys():
                raise IndexError('Spectrum {} does not exist.'.format(to_copy))
            else:
                new_spectrum = deepcopy(self.spectra[to_copy]) 
                
        else:
            new_spectrum = SingleSpectrum(**kwargs)

        self.addSpectrum(name,new_spectrum)

    def addCombinedSpectrum(self,name,to_copy = True,
            isomers=0,Is=[],Js=[],ABCs=[],dfs=[],**kwargs):

        if to_copy:
            try:
                new_spectrum = deepcopy(self.spectra[to_copy]) 
            except:
                raise 

        else:
            spectra = []
            for I,J,ABC,df in zip(Is,Js,ABCs,dfs):
                spectra.append(
                    SingleSpectrum(I=I,J=J,ABC=ABC,df=df,**kwargs)
                    )
            new_spectrum = CombinedSpectrum(spectra)

        self.addSpectrum(name,new_spectrum)

    def addSpectrum(self,name,new_spectrum):
        self.spectra[name] = new_spectrum
        self.notes[name] = []

    def plot_spectrum(self,index=-1):
        spectrum = self.getSpectrum(index)
        if not self.data_loaded:
            self.loadData()
        spectrum.plot_spectroscopic(self._x,self._y)

    def __str__(self):
        ret = '' 
        ret += 'Data path:\n'
        ret += '\t' + str(self.dataPath) + '\n'
        ret += 'Analysis history:\n'
        for name,spec in self.items():
            ret += '\t {}\n'.format(name)
            for n,par in spec.params_from_var().items():
                ret += '\t\t{}:{}+-{}\n'.format(n,par.value,par.stderr)
        return ret


def save(analysis):
    import pickle
    analysis.spectra_params = {n:v.params_from_var() for n,v in analysis.spectra.items()}
    toSave = {n:v for n,v in analysis.__dict__.items() if not n == 'spectra'}

    return pickle.dumps(toSave)

def load(p_dump):
    import pickle
    d = pickle.loads(p_dump)

    a = Analysis(path = d['_dataPath'])
    a.data_loaded = d['data_loaded']
    a.spectra_params = d['spectra_params']
    for n,pars in a.spectra_params.items():
        a.addSingleSpectrum(n,I=0,J=[0,0],ABC=[0,0,0,0,0,0],df=0)
        a.spectra[n].var_from_params(pars)

    a.notes = d['notes']

    return a