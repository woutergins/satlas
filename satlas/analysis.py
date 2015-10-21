from .unit import DataUnit
import pickle

class Analysis(object):
    def __init__(self):
        super(Analysis,self).__init__()
        self.data_units = []
        self.approaches = {}

    def define_approach(self,name,approach):
        self.approaches[name] = [approach(u) for u in self.data_units]

    def add_data_unit(self,data_unit):
        self.data_units.append(data_unit)

    def choose_approach(self,approach):
        zipper = zip(self.data_units,self.approaches[approach])
        for unit,spectrum in zipper:
            unit.spectrum = spectrum

    def analyse_chisq(self):
        for unit in self.data_units:
            unit.analyse_chisq()