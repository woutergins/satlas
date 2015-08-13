import configparser
import glob
import fractions
from itertools import cycle
import os

import numpy as np
from .polar import Polar
from .utilities import Level
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import pyqtgraph.functions as fn

EV_TO_J = 1.60217657 * (10 ** (-19))


class FormattedSpinbox(pg.SpinBox):

    """Extension of pg.SpinBox that correctly implements the
    precision given to it.
    Code by Ruben de Groote."""

    def updateText(self, prev=None):
        self.skipValidate = True
        if self.opts['siPrefix']:
            if self.val == 0 and prev is not None:
                (s, p) = fn.siScale(prev)
                txt = "0.0 %s%s" % (p, self.opts['suffix'])
            else:
                txt = fn.siFormat(float(self.val),
                                  precision=self.opts['decimals'],
                                  suffix=self.opts['suffix'])
        else:
            fmt = "%." + str(self.opts['decimals']) + "g%s"
            txt = fmt % (self.val, self.opts['suffix'])

        self.lineEdit().setText(txt)
        self.lastText = txt
        self.skipValidate = False


class EnergyDock(Dock):

    """Ease-of-use class to create a dock representing a level.
    Spinboxes contain the energy, hyperfine parameters, population
    and quantum numbers of the level, and allow for easy extraction
    of these values."""

    def __init__(self, *args, **kwargs):
        super(EnergyDock, self).__init__(*args, **kwargs)
        self.EnergyLabel = QtGui.QLabel('Energy')
        self.Energy = FormattedSpinbox(value=0,
                                       bounds=[0, None],
                                       suffix='eV', siPrefix=True,
                                       decimals=10)
        self.PopulationLabel = QtGui.QLabel('Initial population')
        self.Population = FormattedSpinbox(value=1,
                                           bounds=[0, None],
                                           decimals=10)
        self.ALabel = QtGui.QLabel('Hyperfine A')
        self.A = FormattedSpinbox(value=0,
                                  suffix=' MHz',
                                  decimals=10)
        self.BLabel = QtGui.QLabel('Hyperfine B')
        self.B = FormattedSpinbox(value=0,
                                  suffix=' MHz',
                                  decimals=10)
        self.LLabel = QtGui.QLabel('Spin L')
        self.L = FormattedSpinbox(value=0,
                                  bounds=[0, None], step=0.5)
        self.SLabel = QtGui.QLabel('Spin S')
        self.S = FormattedSpinbox(value=0.5,
                                  bounds=[0, None], step=0.5)
        self.JLabel = QtGui.QLabel('Spin J')
        self.J = FormattedSpinbox(value=0.5,
                                  bounds=[0, None], step=0.5)

        self.externalLayout = pg.LayoutWidget()
        self.externalLayout.addWidget(self.EnergyLabel)
        self.externalLayout.addWidget(self.Energy)
        self.externalLayout.nextRow()
        self.externalLayout.addWidget(self.PopulationLabel)
        self.externalLayout.addWidget(self.Population)
        self.externalLayout.nextRow()
        self.externalLayout.addWidget(self.ALabel)
        self.externalLayout.addWidget(self.A)
        self.externalLayout.nextRow()
        self.externalLayout.addWidget(self.BLabel)
        self.externalLayout.addWidget(self.B)
        self.externalLayout.nextRow()
        self.externalLayout.addWidget(self.LLabel)
        self.externalLayout.addWidget(self.L)
        self.externalLayout.nextRow()
        self.externalLayout.addWidget(self.SLabel)
        self.externalLayout.addWidget(self.S)
        self.externalLayout.nextRow()
        self.externalLayout.addWidget(self.JLabel)
        self.externalLayout.addWidget(self.J)
        self.addWidget(self.externalLayout)

        self.spinWidgets = [self.L, self.S, self.J]
        for sp in self.spinWidgets:
            sp.sigValueChanging.connect(self.checkSpins)

    def checkSpins(self):
        # Validate the input of the SpinBoxes for the quantum numbers
        for sp in self.spinWidgets:
            valCheck = sp.value()
            if int(valCheck) == valCheck or int(2 * valCheck) == 2 * valCheck:
                pass
            else:
                sp.setValue(int(valCheck))

    def setValues(self, dic):
        # Extract the useable parameters from a dictionary.
        valKeys = {'energy': self.Energy,
                   'a': self.A,
                   'b': self.B,
                   'l': self.L,
                   's': self.S,
                   'j': self.J,
                   'population': self.Population}
        for k in dic.keys():
            if k.lower() in valKeys.keys():
                valKeys[k.lower()].setValue(eval(dic[k]))


class LifetimesDock(Dock):

    """Ease-of-use class to create a dock
    containing the lifetimes for each possible decay
    of each state. The layout is such that the input is
    intuitive.
    Negative values are treated as infinite lifetimes, excluding
    the decay path from happening."""

    def __init__(self, levelCount, *args, **kwargs):
        super(LifetimesDock, self).__init__(*args, **kwargs)
        self.n = levelCount
        self.s = QtGui.QScrollArea()
        self.externalLayout = pg.LayoutWidget()
        self.createWidgets()
        self.addWidget(self.s)
        self.s.setWidgetResizable(True)
        self.s.setWidget(self.externalLayout)

    def createWidgets(self):
        self.labels = []
        self.spinboxes = []
        self.trans = []
        for i in range(1, self.n + 1):
            if i is not 1:
                lab = QtGui.QLabel("%i" % i)
                self.labels.append(lab)
                self.externalLayout.addWidget(self.labels[-1], 0, i)
            lab = QtGui.QLabel("%i" % i)
            self.labels.append(lab)
            self.externalLayout.addWidget(self.labels[-1], i, 0)
            for j in range(i + 1, self.n + 1):
                self.trans.append('%i <-> %i' % (i, j))
                spb = FormattedSpinbox(value=1 * 10 ** -9,
                                       suffix='s', siPrefix=True,
                                       decimals=10)
                self.spinboxes.append(spb)
                self.externalLayout.addWidget(self.spinboxes[-1], i, j)
        lab = QtGui.QLabel("->")
        self.labels.append(lab)
        self.externalLayout.addWidget(self.labels[-1], 0, 1)

    def removeWidgets(self):
        for i in reversed(range(self.externalLayout.layout.count())):
            widget = self.externalLayout.layout.itemAt(i).widget()
            self.externalLayout.layout.removeWidget(widget)
            widget.setParent(None)

    def changeN(self, N):
        self.n = N
        self.removeWidgets()
        self.createWidgets()

    def setValues(self, vals):
        n = self.n
        labels = ['%i <-> %i' % (i, j)
                  for i in range(1, n + 1)
                  for j in range(i + 1, n + 1)]
        for key in vals.keys():
            if key.lower() in labels:
                ind = labels.index(key.lower())
                if vals[key] == 'inf':
                    self.spinboxes[ind].setValue(-1)
                else:
                    self.spinboxes[ind].setValue(eval(vals[key]))

    def giveArray(self):
        # Extract the information in the appropriate 2D NumPy array.
        arr = np.zeros((self.n, self.n))
        arr[:] = np.inf
        for loc, spb in zip(self.trans, self.spinboxes):
            x, y = loc.translate(dict.fromkeys(map(ord, ' <>lifetime'), None)).split('-')
            x, y = int(x) - 1, int(y) - 1
            arr[x, y] = spb.value() if spb.value() >= 0 else np.inf
        return arr


class LasersDock(Dock):

    """Creates a dock that controls all the settings for the laser
    in the simulation. In the case of multiple scanning lasers,
    the input parameters are appropriately extended."""

    def __init__(self, laserCount, laserScans, *args, **kwargs):
        super(LasersDock, self).__init__(*args, **kwargs)
        self.n = laserCount
        self.scannedLasers = laserScans
        self.s = QtGui.QScrollArea()
        self.externalLayout = pg.LayoutWidget()
        self.createWidgets()
        self.addWidget(self.s)
        self.s.setWidgetResizable(True)
        self.s.setWidget(self.externalLayout)

    def createWidgets(self):
        self.intensities = []
        self.frequencies = []
        self.modes = []
        self.left = []
        self.right = []
        self.num = []
        for i in range(1, self.n + 1):
            lab = QtGui.QLabel("Laser %i intensity" % i)
            self.externalLayout.addWidget(lab)
            spb = FormattedSpinbox(value=80 + i,
                                   suffix='W/m^2', siPrefix=True,
                                   decimals=10)
            self.intensities.append(spb)
            self.externalLayout.addWidget(self.intensities[-1])
            self.externalLayout.nextRow()

            lab = QtGui.QLabel("Laser %i frequency" % i)
            self.externalLayout.addWidget(lab)
            spb = FormattedSpinbox(value=761904630.0,
                                   suffix=' MHz', siPrefix=False,
                                   decimals=10)
            self.frequencies.append(spb)
            self.externalLayout.addWidget(self.frequencies[-1])
            self.externalLayout.nextRow()

            lab = QtGui.QLabel("Laser %i polarization" % i)
            self.externalLayout.addWidget(lab)
            spb = pg.SpinBox(value=1, step=1, bounds=[-1, 1],
                             int=True)
            self.modes.append(spb)
            self.externalLayout.addWidget(self.modes[-1])
            self.externalLayout.nextRow()
            self.modes[-1].sigValueChanging.connect(self.checkMode)

            if i <= self.scannedLasers:
                lab = QtGui.QLabel('Left boundary of frequencies')
                self.externalLayout.addWidget(lab)
                spb = FormattedSpinbox(value=-3000,
                                       bounds=[None, 0], suffix=' MHz',
                                       decimals=10)
                self.left.append(spb)
                self.externalLayout.addWidget(self.left[-1])
                self.externalLayout.nextRow()

                lab = QtGui.QLabel('Right boundary of frequencies')
                self.externalLayout.addWidget(lab)
                spb = FormattedSpinbox(value=2000,
                                       bounds=[0, None], suffix=' MHz',
                                       decimals=10)
                self.right.append(spb)
                self.externalLayout.addWidget(self.right[-1])
                self.externalLayout.nextRow()

                lab = QtGui.QLabel('Number of frequency points')
                self.externalLayout.addWidget(lab)
                spb = pg.SpinBox(value=400, step=100,
                                 bounds=[50, 2000], int=True)
                self.num.append(spb)
                self.externalLayout.addWidget(self.num[-1])
                self.externalLayout.nextRow()

    def checkMode(self):
        for i in range(self.n):
            valCheck = self.modes[i].value()
            if valCheck not in [-1, 0, 1]:
                self.modes[i].setValue(1)

    def removeWidgets(self):
        for i in reversed(range(self.externalLayout.layout.count())):
            widget = self.externalLayout.layout.itemAt(i).widget()
            self.externalLayout.layout.removeWidget(widget)
            widget.setParent(None)

    def changeN(self, N):
        self.n = N
        self.removeWidgets()
        self.createWidgets()

    def setScans(self, N):
        self.scannedLasers = N

    def setValues(self, vals):
        mappingLow = {'mode': self.modes, 'leftfreq': self.left,
                      'rightfreq': self.right, 'num': self.num,
                      'freq': self.frequencies, 'intensity': self.intensities}
        mappingHigh = {'mode': self.modes, 'freq': self.frequencies,
                       'intensity': self.intensities}
        keys = sorted(vals.keys())
        for i, las in enumerate(keys):
            if i > (self.scannedLasers - 1):
                mapping = mappingHigh
            else:
                mapping = mappingLow
            for opt in vals[las].keys():
                if opt.lower() in mapping.keys():
                    mapping[opt.lower()][i].setValue(eval(vals[las][opt]))

    def giveValues(self):
        lasers = []
        for i in range(self.n):
            intensity = self.intensities[i].value()
            frequency = self.frequencies[i].value()
            mode = self.modes[i].value()
            if i <= (self.scannedLasers - 1):
                left = self.left[i].value()
                right = self.right[i].value()
                num = self.num[i].value()
                frequency = np.linspace(left + frequency,
                                        right + frequency,
                                        num)
            lasers.append([intensity, frequency, mode])
        return lasers


class OpticalPumpingSimulator(object):

    def __init__(self):
        super(OpticalPumpingSimulator, self).__init__()

        pg.setConfigOptions(antialias=True, foreground='k', background='w')
        dV = {'nuclearSpin': 0.5,
              'leftBound': -3000,
              'rightBound': 2000,
              'freqnum': 400,
              'magField': 600 * (10 ** (-6)),
              'laserInt': 80,
              'laserMode': -1}

        self.axisstyle = {'font-size': '16pt'}
        self.titlestyle = {'size': '18pt'}
        self.tickstyle = {}
        self.app = QtGui.QApplication([])

        # Create the main window for simulations, resize it.
        self.mainWin = QtGui.QMainWindow()
        self.mainWin.resize(1400, 700)
        self.mainWin.setWindowTitle('OPS: Simulator')

        # Create the window for the level docks.
        self.levelWin = QtGui.QMainWindow()
        self.levelWin.setWindowTitle('OPS: Levels')

        # Add the dockareas for both windows.
        self.mainArea = DockArea()
        self.mainWin.setCentralWidget(self.mainArea)
        self.levelArea = DockArea()
        self.scrollLevels = QtGui.QScrollArea()
        self.levelWin.setCentralWidget(self.scrollLevels)
        self.scrollLevels.setWidget(self.levelArea)
        self.scrollLevels.setWidgetResizable(True)

        # Create the different docks, except the lifetimes.
        self.Information = Dock("Beam/Nuclear/Atomic information", size=(1, 1))
        self.LaFiSettings = Dock("Laser and Field settings", size=(1, 1))
        self.Pol = Dock('Polarization', size=(9, 24))
        self.Pop = Dock('Population', size=(9, 24))
        self.Simulate = Dock('Simulate', size=(10, 1))

        # Hide the titlebar, this one doesn't need it.
        self.Simulate.hideTitleBar()

        # Add the docks in the right place to the main window.
        self.mainArea.addDock(self.Information, 'right')
        self.mainArea.addDock(self.LaFiSettings, 'below', self.Information)
        self.mainArea.addDock(self.Pop, 'right')
        self.mainArea.addDock(self.Simulate, 'bottom')
        self.mainArea.addDock(self.Pol, 'above', self.Pop)

        ############################
        # BEAM-INFORMATION WIDGETS #
        ############################
        self.nuclearSpinLabel = QtGui.QLabel('Nuclear spin')
        self.nuclearSpin = pg.SpinBox(value=dV['nuclearSpin'],
                                      bounds=[0, None], step=0.5)
        self.pathLabel = QtGui.QLabel('Flight path time')
        self.path = pg.SpinBox(value=4*10**-6, suffix='s', siPrefix=True)
        self.rPathLabel = QtGui.QLabel('Relaxation path time')
        self.rPath = pg.SpinBox(value=0, suffix='s', siPrefix=True)
        self.levelCountLabel = QtGui.QLabel('Levels')
        self.levelCount = pg.SpinBox(value=2, step=1,
                                     bounds=[2, None], int=True)

        # Add the widgets to a layout, add the layout to the dock.
        self.BeamInformation = pg.LayoutWidget()
        self.BeamInformation.addWidget(self.nuclearSpinLabel)
        self.BeamInformation.addWidget(self.nuclearSpin)
        self.BeamInformation.nextRow()
        self.BeamInformation.addWidget(self.pathLabel)
        self.BeamInformation.addWidget(self.path)
        self.BeamInformation.nextRow()
        self.BeamInformation.addWidget(self.rPathLabel)
        self.BeamInformation.addWidget(self.rPath)
        self.BeamInformation.nextRow()
        self.BeamInformation.addWidget(self.levelCountLabel)
        self.BeamInformation.addWidget(self.levelCount)
        self.levelCount.sigValueChanged.connect(self.checkLevels)
        self.Information.addWidget(self.BeamInformation)

        ###########################
        # LASER AND FIELD WIDGETS #
        ###########################
        self.magFieldLabel = QtGui.QLabel('Magnetic field')
        self.magField = pg.SpinBox(value=dV['magField'],
                                   suffix='T', siPrefix=True)
        self.laserCountLabel = QtGui.QLabel('Lasers')
        self.laserCount = pg.SpinBox(value=1, step=1,
                                     bounds=[1, None], int=True)

        # Add the widgets to a layout, add the layout to the dock
        self.LFSettings = pg.LayoutWidget()
        self.LFSettings.addWidget(self.magFieldLabel)
        self.LFSettings.addWidget(self.magField)
        self.LFSettings.nextRow()
        self.LFSettings.addWidget(self.laserCountLabel)
        self.LFSettings.addWidget(self.laserCount)
        self.LFSettings.nextRow()
        self.laserCount.sigValueChanged.connect(self.checkLasers)
        self.laserValue = self.laserCount.value()

        self.laserButtons = QtGui.QGroupBox("Lasers to scan")
        self.laserButtonsLayout = QtGui.QVBoxLayout()
        self.buttonOne = QtGui.QRadioButton("1")
        self.laserButtonsLayout.addWidget(self.buttonOne)
        self.buttonTwo = QtGui.QRadioButton("2")
        self.laserButtonsLayout.addWidget(self.buttonTwo)
        self.laserButtons.setLayout(self.laserButtonsLayout)
        self.LFSettings.addWidget(self.laserButtons)
        self.buttonOne.clicked.connect(self.setToOneLaser)
        self.buttonTwo.clicked.connect(self.setToTwoLasers)
        self.buttonOne.setChecked(True)

        self.LaFiSettings.addWidget(self.LFSettings)

        #####################################
        # POLARIZATION AND POPULATION PLOTS #
        #####################################
        b = QtGui.QFont()
        b.setPixelSize(14)
        self.polPlot = pg.PlotWidget()
        self.polPlot.setTitle("Polarization from optical pumping", **self.titlestyle)
        self.polPlot.setLabel('left', "Polarization (%)", **self.axisstyle)
        self.polPlot.getAxis('left').tickFont = b
        self.polPlot.setLabel('bottom', "Frequency deviation (MHz)", **self.axisstyle)
        self.polPlot.getAxis('bottom').tickFont = b
        self.polCurve = self.polPlot.plot([0], [0], pen='k',
                                          fillLevel=0.0,
                                          brush=(100, 100, 200, 100))
        self.polPlot.showGrid(x=True, y=True)

        self.popPlot = pg.PlotWidget()
        self.popPlot.setTitle("Population in fine-structure levels", **self.titlestyle)
        self.popPlot.setLabel('left', "Population (%)", **self.axisstyle)
        self.popPlot.getAxis('left').tickFont = b
        self.popPlot.setLabel('bottom', "Frequency deviation (MHz)", **self.axisstyle)
        self.popPlot.getAxis('bottom').tickFont = b
        self.popPlot.setXLink(self.polPlot)
        self.popPlot.showGrid(x=True, y=True)
        self.popPlot.addLegend()
        self.popCurves = []
        self.popCurves.append(self.popPlot.plot([0], [0], pen='k',
                                                fillLevel=0.0,
                                                brush=(100, 100, 200, 100)))

        self.Pol.addWidget(self.polPlot)
        self.Pop.addWidget(self.popPlot)

        ####################
        # SETTINGS WIDGETS #
        ####################
        self.simulateButton = QtGui.QPushButton('Simulate')
        self.simulateButton.clicked.connect(self.simulatePushed)
        self.configMenu = pg.ComboBox()
        items = glob.glob('*.ini')
        for item in items:
            self.configMenu.addItem(item)
        if 'default.ini' not in items:
            self.configMenu.addItem('')
            self.configMenu.setValue('')
        else:
            self.configMenu.setValue('default.ini')
        self.configMenu.currentIndexChanged.connect(self.readConfig)
        self.writeConfigButton = QtGui.QPushButton('Save')
        self.writeConfigButton.clicked.connect(self.createConfig)

        self.Simulate.addWidget(self.simulateButton)
        self.Simulate.addWidget(self.writeConfigButton)
        self.Simulate.addWidget(self.configMenu)

        # Add the lifetimes dock, at the right place.
        self.lifetimes = LifetimesDock(self.levelCount.value(),
                                       "Lifetimes", size=(1, 1))
        self.mainArea.addDock(self.lifetimes, 'bottom', self.Information)

        # Add the lasers dock
        self.lasers = LasersDock(self.laserCount.value(), 1, "Lasers",
                                 size=(1, 1))
        self.mainArea.addDock(self.lasers, 'below', self.lifetimes)

        # Generate the level docks
        self.levels = []
        self.addLevels()

    def removeLevels(self):
        splitter = self.levelArea.topContainer
        for i in reversed(range(splitter.count())):
            widget = splitter.widget(i)
            widget.deleteLater()
            widget.setParent(None)

    def checkLevels(self):
        self.removeLevels()
        self.addLevels()

    def addLevels(self):
        n = self.levelCount.value()
        self.levels = []
        self.lifetimes.changeN(n)
        for i in range(1, n + 1):
            self.levels.append(EnergyDock("Level " + str(i), size=(1, 10)))
            if i == 1:
                self.levelArea.addDock(self.levels[-1], 'top')
            else:
                self.levelArea.addDock(self.levels[-1], 'right',
                                       self.levels[-2])
            self.levels[-1].Population.setValue(0)
        self.levels[-1].Population.setValue(1)

    def checkLasers(self):
        n = self.laserCount.value()

        if n == 1 and not isinstance(self.polPlot, pg.PlotWidget):
            self.setOneScanView()
        elif n > 1:
            if self.laserValue == 1:
                if self.buttonTwo.isChecked():
                    self.setTwoScanView()
        else:
            pass
        if not self.laserValue == n:
            # self.lasers.setScans(1 if self.buttonOne.isChecked() else 2)
            self.lasers.changeN(n)
            self.laserValue = n

    def setOneScanView(self):
        self.polPlot.deleteLater()
        self.polPlot.setParent(None)
        self.popPlot.deleteLater()
        self.popPlot.setParent(None)

        b = QtGui.QFont()
        b.setPixelSize(14)
        self.polPlot = pg.PlotWidget()
        self.polPlot.setTitle("Polarization from optical pumping", **self.titlestyle)
        self.polPlot.setLabel('left', "Polarization (%)", **self.axisstyle)
        self.polPlot.getAxis('left').tickFont = b
        self.polPlot.setLabel('bottom', "Frequency deviation (MHz)", **self.axisstyle)
        self.polPlot.getAxis('bottom').tickFont = b
        self.polCurve = self.polPlot.plot([0], [0], pen='k',
                                          fillLevel=0.0,
                                          brush=(100, 100, 200, 100))
        self.polPlot.showGrid(x=True, y=True)

        self.popPlot = pg.PlotWidget()
        self.popPlot.setTitle("Population in fine-structure levels", **self.titlestyle)
        self.popPlot.setLabel('left', "Population (%)", **self.axisstyle)
        self.polPlot.getAxis('left').tickFont = b
        self.popPlot.setLabel('bottom', "Frequency deviation (MHz)", **self.axisstyle)
        self.polPlot.getAxis('bottom').tickFont = b
        self.popPlot.setXLink(self.polPlot)
        self.popPlot.showGrid(x=True, y=True)
        self.popPlot.addLegend()
        self.popCurves = []
        self.popCurves.append(self.popPlot.plot([0], [0], pen='k',
                                                fillLevel=0.0,
                                                brush=(100, 100, 200, 100)))
        self.Pol.addWidget(self.polPlot)
        self.Pop.addWidget(self.popPlot)
        self.lasers.setScans(1)

    def setTwoScanView(self):
        self.polPlot.deleteLater()
        self.polPlot.setParent(None)
        self.popPlot.deleteLater()
        self.popPlot.setParent(None)
        pl_view = pg.PlotItem(labels={'bottom':
                                      ("Laser 1 - Frequency deviation", "Hz"),
                                      'left':
                                      ("Laser 2 - Frequency deviation", "Hz")})
        self.polPlot = pg.ImageView(view=pl_view)

        pp_view = pg.PlotItem(labels={'bottom':
                                      ("Laser 1 - Frequency deviation", "Hz"),
                                      'left':
                                      ("Laser 2 - Frequency deviation", "Hz")})
        self.popPlot = pg.ImageView(view=pp_view)
        self.polPlot.setLevels(-100, 100)
        self.popPlot.setLevels(0, 100)
        self.polPlot.ui.histogram.gradient.loadPreset('bipolar')
        self.popPlot.ui.histogram.gradient.loadPreset('bipolar')
        self.Pol.addWidget(self.polPlot)
        self.Pop.addWidget(self.popPlot)
        self.lasers.setScans(2)

    def setToOneLaser(self):
        n = self.laserCount.value()
        if n > 1:
            self.setOneScanView()
        else:
            pass
        self.lasers.changeN(n)

    def setToTwoLasers(self):
        n = self.laserCount.value()
        if n == 1:
            pass
        elif n > 1:
            self.setTwoScanView()
        self.lasers.changeN(n)

    def checkMode(self):
        valCheck = self.laserMode.value()
        if valCheck not in [-1, 0, 1]:
            self.laserMode.setValue(1)

    def simulatePushed(self):
        # Gather values
        levels = []
        for lev in self.levels:
            levels.append(Level(
                          lev.Energy.value(),
                          (lev.A.value(),
                           lev.B.value()),
                          lev.L.value(),
                          lev.S.value(),
                          lev.J.value())
                          )
        tof = self.path.value()

        lasers = self.lasers.giveValues()
        laser_intensity = []
        laser_frequency = []
        laser_mode = []
        for l in lasers:
            laser_intensity.append(l[0])
            laser_frequency.append(l[1])
            laser_mode.append(l[2])

        field = self.magField.value()
        lifetime = self.lifetimes.giveArray()
        spin = self.nuclearSpin.value()

        rPath = max(0, self.rPath.value())

        # Create the calculation object
        p = Polar(levels, laser_intensity,
                  laser_mode, spin, field,
                  lifetime, tof,
                  relaxationtime=rPath)
        print('levels=', levels)
        print('laser_intensity=', laser_intensity)
        print('laser_mode=', laser_mode)
        print('spin=', spin)
        print('field=', field)
        print('lifetime=', lifetime)
        print('tof=', tof)
        print('rPath=', rPath)

        # Set the population
        initialPopulaion = [lev.Population.value() for lev in self.levels]
        p.changeInitialPopulation(initialPopulaion)

        # Calculate
        print(laser_frequency)
        y = p(laser_frequency)

        # Process and plot the results
        if self.buttonOne.isChecked():
            laser_frequency = (laser_frequency[0] -
                               self.lasers.frequencies[0].value())
            self.labels = []
            self.polPlot.clear()
            self.polCurve = self.polPlot.plot(laser_frequency, y[:, 0], pen='k',
                                              fillLevel=0.0,
                                              brush=(100, 100, 200, 100))
            curves = y.shape[1]
            self.popPlot.clear()
            self.popPlot.plotItem.legend.items = []
            _colors = cycle('rbkgcmy')
            for i, pen in zip(range(1, curves), _colors):
                d = self.popPlot.plot(laser_frequency, y[:, i],
                                      pen=pen,
                                      name='Level %i' % (i))
                d.setPen(pen, width=2.0)
            used_labels = []
            pos = 1
            for fg, fe, position in sorted(p.pos, key=lambda x: x[-1]):
                if -1400 < position - self.lasers.frequencies[0].value() < 400:
                    text = '{fg}->{fe}'.format(fe=fractions.Fraction(fe), fg=fractions.Fraction(fg))
                    if text not in used_labels:
                        used_labels.append(text)
                        label = pg.TextItem(text=text,
                                            anchor=(0.5, 0.5), color='k')
                        label.setPos(position - self.lasers.frequencies[0].value(), pos)
                        pos = -pos
                        self.labels.append(label)
                        self.polPlot.addItem(self.labels[-1])
        else:
            # Laser settings
            x0, y0 = (laser_frequency[0][0], laser_frequency[1][0])
            x1, y1 = (laser_frequency[0][-1], laser_frequency[1][-1])
            x0 -= self.lasers.frequencies[0].value()
            x1 -= self.lasers.frequencies[0].value()
            y0 -= self.lasers.frequencies[1].value()
            y1 -= self.lasers.frequencies[1].value()
            x0, y0, x1, y1 = (x0 * 10 ** 6, y0 * 10 ** 6,
                              x1 * 10 ** 6, y1 * 10 ** 6)
            xnum, ynum = (len(laser_frequency[0]), len(laser_frequency[1]))
            xscale, yscale = ((x1 - x0) / xnum, (y1 - y0) / ynum)

            # Set the polarization image
            polImg = np.transpose(y[:, :, 0])
            self.polPlot.setImage(polImg, pos=[x0, y0], scale=[xscale, yscale],
                                  autoLevels=False, autoHistogramRange=False)

            # Set the population image
            popImg = y[:, :, 1:]
            pImg = []
            xvals = np.array([i + 1 for i in range(self.levelCount.value())])
            for x in xvals:
                pImg.append(np.transpose(popImg[:, :, x - 1]))
            pImg = np.array(pImg)
            self.popPlot.setImage(pImg, xvals=xvals,
                                  pos=[x0, y0], scale=[xscale, yscale],
                                  axes={'x': 0, 'y': 1, 't': 2, 'c': 3},
                                  autoLevels=False, autoHistogramRange=False)
            self.popPlot.timeLine.setPen(
                QtGui.QPen(QtGui.QColor(0, 0, 0, 200)))

    def show(self):
        self.mainWin.show()
        self.levelWin.show()

    def createConfig(self):
        n = self.levelCount.value()
        filename, filter = QtGui.QFileDialog.getSaveFileNameAndFilter(
            None, 'Save config', '', '*.ini')
        filename = str(filename)
        filename, extension = os.path.splitext(filename)
        extension = '.ini'
        filename += extension

        with open(filename, 'w') as cfgfile:
            config = configparser.ConfigParser()
            for i in range(n):
                config.add_section('Level %i' % (i + 1))
                config.set("Level %i" % (i + 1),
                           'Energy', self.levels[i].Energy.value())
                config.set("Level %i" % (i + 1),
                           'Population', self.levels[i].Population.value())
                config.set("Level %i" % (i + 1),
                           'A', self.levels[i].A.value())
                config.set("Level %i" % (i + 1),
                           'B', self.levels[i].B.value())
                config.set("Level %i" % (i + 1),
                           'L', self.levels[i].L.value())
                config.set("Level %i" % (i + 1),
                           'S', self.levels[i].S.value())
                config.set("Level %i" % (i + 1),
                           'J', self.levels[i].J.value())
            config.add_section('Lifetimes')
            lifetimes = self.lifetimes.giveArray()
            for i in range(lifetimes.shape[0]):
                for j in range(i + 1, lifetimes.shape[1]):
                    config.set('Lifetimes', '%i <-> %i' % (i + 1, j + 1),
                               lifetimes[i, j])

            config.add_section('Properties')
            config.set('Properties', 'Nucleus', str(self.nucleus.text()))
            config.set('Properties', 'NuclearSpin', self.nuclearSpin.value())
            config.set('Properties', 'KineticEnergy',
                       self.kineticenergy.value())
            config.set('Properties', 'Path', self.path.value())

            n = self.laserCount.value()
            for i in range(n):
                config.add_section('Laser %i' % (i + 1))
                config.set('Laser %i' % (i + 1),
                           'Intensity', self.lasers.intensities[i].value())
                config.set('Laser %i' % (i + 1),
                           'Mode', self.lasers.modes[i].value())
                config.set('Laser %i' % (i + 1),
                           'Freq', self.lasers.frequencies[i].value())
                if i <= self.lasers.scannedLasers - 1:
                    config.set('Laser %i' % (i + 1),
                               'LeftFreq', self.lasers.left[i].value())
                    config.set('Laser %i' % (i + 1),
                               'RightFreq', self.lasers.right[i].value())
                    config.set('Laser %i' % (i + 1),
                               'Num', self.lasers.num[i].value())
            config.add_section('Field Settings')
            config.set('Field Settings',
                       'magField', self.magField.value())
            config.write(cfgfile)

    def readConfig(self, index):
        configFile = str(self.configMenu.itemText(index))
        config = configparser.ConfigParser()
        config.read(configFile)

        values = {}
        sect = config.sections()
        n = []
        for s in sect:
            if 'level' in s.lower():
                n.append(s)
        n = len(n)

        self.levelCount.setValue(n)

        for i, lev in enumerate(self.levels):
            sec = 'Level ' + str(i + 1)
            for opt in config.options(sec):
                values[opt] = config.get(sec, opt)
            lev.setValues(values)

        values = {}
        sec = 'Lifetimes'
        for opt in config.options(sec):
            values[opt] = config.get(sec, opt)
        self.lifetimes.setValues(values)

        values = {}
        sect = config.sections()
        n = []
        for s in sect:
            if 'laser' in s.lower():
                n.append(s)
        n = len(n)

        self.laserCount.setValue(n)

        for i in range(n):
            sec = 'Laser ' + str(i + 1)
            values[sec] = {}
            for opt in config.options(sec):
                values[sec][opt] = config.get(sec, opt)
        self.lasers.setValues(values)

        values = {}
        for sect in config.sections():
            if 'level' not in sect.lower() and 'laser' not in sect.lower():
                for opt in config.options(sect):
                    values[opt] = config.get(sect, opt)
        self.setValues(values)

    def setValues(self, values):
        keys = {'nuclearspin': self.nuclearSpin,
                'nucleus': self.nucleus,
                'kineticenergy': self.kineticenergy,
                'path': self.path,
                'magfield': self.magField}

        for k in keys.keys():
            if k in values.keys() and k is not 'nucleus':
                keys[k].setValue(eval(values[k]))
            elif k is 'nucleus':
                keys[k].setText(values[k])
            else:
                keys[k].setValue(None)

if __name__ == '__main__':
    import sys
    Simple = OpticalPumpingSimulator()
    Simple.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
