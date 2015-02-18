"""
.. module:: collaps
    :platform: Windows
    :synopsis: Implementation of classes and functions for easy interfacing and
     analysis of COLLAPS data

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""

from datetime import datetime
import re
import sys
from satlas.spectrum import Spectrum
import satlas.utilities as utilities
import lmfit as lm
import numpy as np
from scipy.optimize import root

if sys.version_info >= (3, 0):
    string_class = str
else:
    import string as string_class

normal = 'normal'

###############
# READING MCP #
###############


class Measurement(object):

    """Representation of a measurement as saved in an MCP-file.

    Attributes
    ----------
    x : NumPy array
        Line voltage.
    y : NumPy array
        Counts of the scalar with the highest attribute.
    ySeperate : dictionary of NumPy arrays
        The counts for the different scalars are saved in this
        dictionary, with the channel number as the key.
    premaVoltage : NumPy array.
        An array of all the recorded premavoltages for the run.
    tracks : List of Track-objects.
        The different tracks are saved in this list.
    timeStamp : Datetime object
        Contains the timestamp of when the run was taken."""

    def __init__(self):
        super(Measurement, self).__init__()
        self.tracks = []
        self.timeStamp = None

    def _updateProperties(self):
        newFiles = hasattr(self.tracks[-1], 'cecVoltage')
        if self.tracks[-1].measurement:
            self.ySeperate = {}
            for key in sorted(self.tracks[-1].spectrum.keys()):
                dinkie = np.array([])
                self.x = np.array([])
                self.premaVoltage = np.array([])
                if newFiles:
                    self.cecVoltage = np.array([])
                    self.coolerVoltage = np.array([])
                for track in self.tracks:
                    self.x = np.append(self.x, track.voltage)
                    self.premaVoltage = np.append(self.premaVoltage,
                                                  track.premaVoltage)
                    if newFiles:
                        self.cecVoltage = np.append(self.cecVoltage,
                                                    track.cecVoltage)
                        self.coolerVoltage = np.append(self.coolerVoltage,
                                                       track.coolerVoltage)
                    dinkie = np.append(dinkie, track.spectrum[key].data)
                self.ySeperate[key] = dinkie
            self.channelNumbers = sorted(self.ySeperate.keys())
            self.y = self.ySeperate[self.channelNumbers[-1]]
        else:
            self.x = self.tracks[-1].voltage
            self.y = self.tracks[-1].kepcoMeasurement if (
                self.tracks[-1].kepcoUsed) else self.tracks[-1].siclMeasurement
            self.ySeperate = {}
            for track in self.tracks:
                self.ySeperate[track.fluke] = track.kepcoMeasurement if (
                    track.kepcoUsed) else track.siclMeasurement

    def addTrack(self, trackObj):
        try:
            for track in trackObj:
                self.tracks.append(track)
        except TypeError:
            self.tracks.append(trackObj)
        self._updateProperties()


class Track(object):

    """A Track object contains all the information
    that is displayed in the 'MCP for NT' program for
    each track.
    Note: If an attribute is not applicable for the track, i.e. it could
    not be retrieved from the file, the value defaults to :class:`None`.

    Attributes
    ----------
    isotope : string
        Isotope investigated in the run.
    scans : int
        Number of scans.
    cycles : int
        Number of cycles per scan.
    spectrum : dictionary
        Dictionary of :class:`PM_spectrumObj`.
    premaVoltage : NumPy array
        Array of recorded voltages for the Prema.
    kepcoMeasurement : NumPy array
        Voltages as recorded during the Kepco calibration measurements.
    fluke : int
        Number of the used Fluke.
    voltage : NumPy array
        Applied linevoltages.
    trigger : float
        Measuring step of the trigger, in seconds.
    a, astderr : floats
        In the case of a Kepco calibration measurement, this is the fitted
        value and standard deviation for the slope.
    b, bstderr : floats
        In the case of a Kepco calibration measurement, this is the fitted
        value and standard deviation for the intercept."""

    def __init__(self, *arg):
        super(Track, self).__init__()
        self.isotope = arg[0]
        self.scans = arg[2]
        self.cycles = arg[1]
        self.measurement = False
        self.kepco = False
        self.sicl = False
        self._kepcoMeasurement = None
        self._siclMeasurement = None
        self.fluke = None
        self.voltage = None
        self._voltageObj = None
        self._bins = None
        self.trigger = None
        self.premaVoltage = np.array([])

    def __str__(self):
        s = 'Track (' + str(self.isotope) + ')'
        return s

    def addObject(self, obj):
        if isinstance(obj, PM_SpectrumObj):
            if not self.measurement:
                self.measurement = True
                self.spectrum = {}
            if obj.number in self.spectrum.keys():
                obj.number = obj.number + 1
            self.spectrum[obj.number] = obj
            self.bins = obj.bins
        elif isinstance(obj, PremaVoltageObj):
            self.premaVoltage = np.append(self.premaVoltage, obj.voltageList)
        elif isinstance(obj, KepcoEichungVoltageObj):
            self.kepco = True
            self.kepcoUsed = False
            self.bins = len(obj.voltageList)
            self.kepcoMeasurement = obj.voltageList
        elif isinstance(obj, TriggerObj):
            self.trigger = obj.measuringStep
        elif isinstance(obj, FlukeSwitchObj):
            self.fluke = obj.number
        elif isinstance(obj, LineVoltageSweepObj):
            self.voltageObj = obj
        elif isinstance(obj, SiclStepObj):
            self.sicl = True
            self.siclUsed = False
            self.siclMeasurement = obj.voltageList[0:self.bins]
        elif isinstance(obj, SiclReaderObj):
            if obj.measured == 'CEC':
                if not hasattr(self, 'cecVoltage'):
                    self.cecVoltage = np.array([])
                self.cecVoltage = np.append(self.cecVoltage, obj.cecvoltage)
            elif obj.measured == 'Cooler':
                self.coolerVoltage = obj.coolervoltage
            else:
                pass

    @property
    def voltageObj(self):
        return self._voltageObj

    @voltageObj.setter
    def voltageObj(self, obj):
        self._voltageObj = obj
        if self.bins is not None:
            self.voltage = self._voltageObj(self.bins)
            if self.kepco:
                self.kepcoMeasurement = self.kepcoMeasurement
            if self.sicl:
                self.siclMeasurement = self.siclMeasurement

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, value):
        self._bins = value
        if self.voltageObj is not None:
            self.voltage = self.voltageObj(self._bins)
            if self.kepco:
                self.kepcoMeasurement = self.kepcoMeasurement
            if self.sicl:
                self.siclMeasurement = self.siclMeasurement

    @property
    def kepcoMeasurement(self):
        return self._kepcoMeasurement

    @kepcoMeasurement.setter
    def kepcoMeasurement(self, value):
        self._kepcoMeasurement = value
        if self.voltage is not None and self._kepcoMeasurement is not None:
            model = lambda params: self._kepcoMeasurement - (params['a'].value
                                                             * self.voltage +
                                                             params['b'].value)
            estA = (self._kepcoMeasurement[1] -
                    self._kepcoMeasurement[0]) / (self.voltage[1] -
                                                  self.voltage[0])
            estB = self._kepcoMeasurement[0] - estA * self.voltage[0]

            params = lm.Parameters()
            params.add('a', value=estA)
            params.add('b', value=estB)

            result = lm.minimize(model, params)
            self.aKepco = result.params['a'].value
            self.astderrKepco = result.params['a'].stderr
            self.bKepco = result.params['b'].value
            self.bstderrKepco = result.params['b'].stderr
            kepcoA = (self.aKepco, self.astderrKepco)
            kepcoB = (self.bKepco, self.bstderrKepco)
            if hasattr(self, 'astderrSicl'):
                siclA = (self.aSicl, self.astderrSicl)
                siclB = (self.bSicl, self.bstderrSicl)
                selectKepco = self.astderrKepco < self.astderrSicl
                self.a, self.astderr = kepcoA if selectKepco else siclA
                self.b, self.bstderr = kepcoB if selectKepco else siclB
                self.kepcoUsed = selectKepco
                self.siclUsed = not self.kepcoUsed
            else:
                self.kepcoUsed = True
                self.a, self.astderr = kepcoA
                self.b, self.bstderr = kepcoB

    @property
    def siclMeasurement(self):
        return self._siclMeasurement

    @siclMeasurement.setter
    def siclMeasurement(self, value):
        self._siclMeasurement = value[0:self.bins]
        if self.voltage is not None and self._siclMeasurement is not None:
            model = lambda params: self._siclMeasurement - (params['a'].value
                                                            * self.voltage +
                                                            params['b'].value)
            estA = (self._siclMeasurement[1] -
                    self._siclMeasurement[0]) / (self.voltage[1] -
                                                 self.voltage[0])
            estB = self._siclMeasurement[0] - estA * self.voltage[0]

            params = lm.Parameters()
            params.add('a', value=estA)
            params.add('b', value=estB)

            result = lm.minimize(model, params)
            self.aSicl = result.params['a'].value
            self.astderrSicl = result.params['a'].stderr
            self.bSicl = result.params['b'].value
            self.bstderrSicl = result.params['b'].stderr
            siclA = (self.aSicl, self.astderrSicl)
            siclB = (self.bSicl, self.bstderrSicl)
            if hasattr(self, 'astderrKepco'):
                kepcoA = (self.aKepco, self.astderrKepco)
                kepcoB = (self.bKepco, self.bstderrKepco)
                selectKepco = self.astderrKepco < self.astderrSicl
                self.a, self.astderr = kepcoA if selectKepco else siclA
                self.b, self.bstderr = kepcoB if selectKepco else siclB
                self.kepcoUsed = selectKepco
                self.siclUsed = not self.kepcoUsed
            else:
                self.siclUsed = True
                self.a, self.astderr = siclA
                self.b, self.bstderr = siclB


class KepcoEichungVoltageObj(object):

    def __init__(self, *arg):
        super(KepcoEichungVoltageObj, self).__init__()
        # self.arg = arg
        self.voltageList = np.array(arg[-1]) * 10.0 ** 3

    def __str__(self):
        s = "KepcoEichungVoltageObj(" + str(self.voltageList) + ')'
        return s


class PremaVoltageObj(object):

    def __init__(self, *arg):
        super(PremaVoltageObj, self).__init__()
        self.arg = arg
        self.voltageList = np.array(arg[-1]) * 1000.0

    def __str__(self):
        s = "PremaVoltageObj(" + str(self.voltageList) + ')'
        return s


class LineVoltageSweepObj(object):

    def __init__(self, begin, end, *args):
        super(LineVoltageSweepObj, self).__init__()
        self.begin = begin
        self.end = end
        self.args = args

    def __call__(self, number):
        return np.linspace(self.begin, self.end, number)

    def __str__(self):
        s = "LineVoltageSweepObj(" + str(self.begin) + ','
        s += str(self.end) + ','
        s += str(self.args) + ')'
        return s


class FlukeSwitchObj(object):

    def __init__(self, *arg):
        super(FlukeSwitchObj, self).__init__()
        self.arg = arg
        self.number = arg[0]

    def __str__(self):
        s = "FlukeSwitchObj(" + str(self.arg) + ')'
        return s


class TriggerObj(object):

    def __init__(self, *arg):
        super(TriggerObj, self).__init__()
        self.arg = arg
        self.measuringStep = arg[-1] * 10 ** -3

    def __str__(self):
        s = "TriggerObj(" + str(self.measuringStep) + ')'
        return s


class PM_SpectrumObj(object):

    # """Represents a spectrum object.

    # Attributes
    # ----------

    # number : int
    #     Channel number of the scalar. Typically, 0-3 represent the ungated photomultiplier tubes (L1, L2, R1, R2), 4-7 represent the gated photomultiplier tubes (L1, L2, R1, R2), the highest channel number represents the sum of the gated signals
    # data : NumPy array
    #     Data of the spectrum
    # bins : int
    #     Amount of channels in the spectrum
    # kind : 'normal' or 'composite'.
    #     If kind is 'normal', the scalar is the gated or ungated signal. If kind is 'composite', the scalar is a processed signal
    # formula : string
    #     If kind is 'composite', this string represents the formula used to obtain the spectrum in function of the other channels and tracks"""

    def __init__(self, *arg):
        super(PM_SpectrumObj, self).__init__()
        self.number = arg[0]
        self.data = np.array(arg[-1])
        self.bins = arg[-2]
        self.kind = arg[4] if not arg[4] == '0' else 'composite'
        self.formula = arg[5] if self.kind == 'composite' else None

    def __str__(self):
        s = "PM_SpectrumObj(" + str(self.number) + ')'
        return s


class SiclReaderObj(object):

    def __init__(self, *arg):
        super(SiclReaderObj, self).__init__()
        if arg[0] == 'lan[A-34461A-06287]:inst0':
            self.measured = 'CEC'
            self.cecvoltage = np.array(arg[-1]) * 10.0 ** 3
            for i, val in enumerate(self.cecvoltage):
                dig = 7 if str(val)[0] == '1' else 6
                self.cecvoltage[i] = utilities.round2SignifFigs([val], dig)
        else:
            self.measured = 'Cooler'
            self.coolervoltage = np.array(arg[-1]) * 10.0 ** 4
            for i, val in enumerate(self.coolervoltage):
                dig = 7 if str(val)[0] == '1' else 6
                self.coolervoltage[i] = utilities.round2SignifFigs([val], dig)


class SiclStepObj(object):

    def __init__(self, *arg):
        super(SiclStepObj, self).__init__()
        self.voltageList = np.array(arg[-1]) * 10.0 ** 3
        for i, val in enumerate(self.voltageList):
            dig = 7 if str(val)[0] == '1' else 6
            self.voltageList[i] = utilities.round2SignifFigs([val], dig)


class MCPParser(object):

    """Parser for MCP files as recorded by the COLLAPS experiment
    at ISOLDE, CERN."""

    def __init__(self):
        super(MCPParser, self).__init__()
        self.startDataSymbol = '<'
        self.stopDataSymbol = '>'
        self.startObjectSymbol = '['
        self.endObjectSymbol = ']'
        self.table = string_class.maketrans('<>', '[]')

    def _startObject(self, dataString):
        orig = dataString
        dataString = dataString[1:-1]
        dataString = re.sub('[\n\r]', '', dataString)
        dataString = dataString.split(',')
        obj = dataString[0].strip('"')
        arguments = dataString[1:]
        args = ','.join(arguments)
        args = args.replace('""', '')
        args = args.replace(',,', ',')
        args = args.translate(self.table)
        args = args.replace(')[', '),[')
        charRepl = []
        for i, char in enumerate(args):
            if char == '(':
                if not args[i - 1] in [',', 'c']:
                    charRepl.append(args[i - 1])
        for char in charRepl:
            args = args.replace(char + '(', char + ',(')
        input = obj + '(' + args + ')'
        try:
            return eval(input)
        except:
            print(orig)
            raise

    def feed(self, dataString):
        """Parse the input string and convert to different objects for use in
        the creation of a :class:`Measurement` object.

        Parameters
        ----------
        dataString: string
            String to be parsed.

        Returns
        -------
        Measurement
            Object representing the measurement as described by the input
            string."""
        checkData = False
        measure = Measurement()
        tracks = []
        for i, char in enumerate(dataString):
            try:
                if (dataString[i] == dataString[i + 1] == '<') or \
                   (dataString[i] == '>' and dataString[i + 1] == ','
                        and dataString[i + 2] == '<'):
                    for j, val in enumerate(dataString[i:]):
                        if val == '\n':
                            pos = j
                            break
                    args = re.sub('[\n]', '', dataString[i:i + pos + 1])
                    args = args.strip('<,>')
                    tracks.append(eval('Track(' + args + ')'))
                if dataString[i] == '@' and dataString[i + 1] == '<':
                    date = dataString[i + 3:i + 2 + 25]
                    Time = datetime.strptime(date, '%a %b %d %H:%M:%S %Y')
            except IndexError:
                pass
            if char == self.startObjectSymbol and not dataString[i - 1] == 'n':
                checkData = True
            if char == self.startDataSymbol and checkData:
                closings = 0
                for j, val in enumerate(dataString[i:]):
                    if val == self.startDataSymbol:
                        closings += 1
                    if val == self.stopDataSymbol:
                        pos = j
                        closings -= 1
                    if closings == 0:
                        break
                dinkie = self._startObject(dataString[i:i + pos + 1])
                tracks[-1].addObject(dinkie)
                checkData = False
        for track in tracks:
            measure.addTrack(track)
        measure.timeStamp = Time
        return measure

    def feedFile(self, filename):
        """Read the entire file as a single string, then call :meth:`feed`
        with this string.

        Parameters
        ----------
        filename: string
            Name of the file to be read, including extension.

        Returns
        -------
        Measurement
            Measurement object representing the scan."""
        input = ''
        with open(filename, 'r') as f:
            for line in f:
                input += line
        return self.feed(input)


#################################
# ANALYSIS AND PREDICTION TOOLS #
#################################
def beta(mass, V):
    """Calculates the beta-factor for a mass in amu
    and applied voltage in Volt.

    Parameters
    ----------
    mass : float
        Mass in amu.
    V : float
        voltage in volt.

    Returns
    -------
    float
        Relativistic beta-factor.
    """
    c = 299792458.0
    q = 1.60217657 * (10 ** (-19))
    AMU2KG = 1.66053892 * 10 ** (-27)
    mass = mass * AMU2KG
    top = mass ** 2 * c ** 4
    bottom = (mass * c ** 2 + q * V) ** 2
    beta = np.sqrt(1 - top / bottom)
    return beta


def dopplerfactor(mass, V):
    """Calculates the Doppler shift of the laser frequency for a
    given mass in amu and voltage in V.

    Parameters
    ----------
    mass : float
        Mass in amu.
    V : float
        Voltage in volt.

    Returns
    -------
    float
        Doppler factor.
    """
    betaFactor = beta(mass, V)
    dopplerFactor = np.sqrt((1.0 - betaFactor) / (1.0 + betaFactor))
    return dopplerFactor


def dopplerwidth(mass, dE, V, freq):
    """Calculates the estimated Doppler width of the transition, given
    the mass, acceleration voltage and initial energy spread.

    Parameters
    ----------
    mass : float
        Mass in amu.
    dE : float
        Energy spread in eV.
    V : float
        Acceleration voltage.
    freq : float
        Optical transition frequency.

    Returns
    -------
    float
        Estimated Doppler width.

    Note
    ----
    Formula found in the master thesis of Hanne Heylen :cite:`Heylen2012`."""
    c = 299792458.0
    q = 1.60217657 * (10 ** (-19))
    AMU2KG = 1.66053892 * 10 ** (-27)
    mass = mass * AMU2KG
    width = freq * dE * np.sqrt(q / (2.0 * V * mass * c * c))
    return width


def voltageshift(mass, transitionFreq, laserFreq,
                 origVoltage=30000.0, isotopeshift=0.0):
    """Calculates what voltage has to be applied in order to get the
    resonance.

    Parameters
    ----------
    mass : float
        Mass of the isotope in amu.
    transitionFreq : float or :class:`Energy` instance
        Frequency of the transition. Can be a value (in MHz), or an
        instance of the :class:`Energy` class.
    laserFreq : float or :class:`Energy` instance
        Frequency that the laser is tuned to. Can be a value (in MHz), or an
        instance of the :class:`Energy` class.

    Other parameters
    ----------------
    origVoltage : float, optional
        Original applied voltage in volt. Defaults to 30kV.
    isotopeshift : float or :class:`Energy` instance
        Value of the isotope shift. Can be a value (in MHz), or an
        instance of the :class:`Energy` class. Defaults to 0.

    Returns
    -------
    tuple
        The first value is the new value of the total applied voltage needed,
        the second is the shift compared to the original voltage. Note that a
        negative value means an increase in kinetic energy is needed. The value
        of :attr:`shift` is what the Fluke should be set to, including
        polarity!
    """
    if isinstance(isotopeshift, utilities.Energy):
        isotopeshift = isotopeshift('MHz')
    if isinstance(transitionFreq, utilities.Energy):
        transitionFreq = transitionFreq('MHz')
    if isinstance(laserFreq, utilities.Energy):
        laserFreq = laserFreq('MHz')
    transitionFreq += isotopeshift
    dopplerFactor = transitionFreq / laserFreq

    def rootFinding(V):
        dopp = dopplerfactor(mass, V[0])
        return [dopplerFactor - dopp]

    res = root(rootFinding, [origVoltage])
    totalVoltage = res.x[0]
    shift = origVoltage - totalVoltage
    return totalVoltage, shift


def predictspectrum(mass, I, J, ABC, isotopeshift, lifetime, transitionFreq,
                    laserFreq, fixedVoltage, kepcoFactor=50.0, channels=10000,
                    plot=False, show=True, name=None, useMatplotLib=False):
    """Based on the parameters given, return the guess for what the spectrum
    will look like. Warns if the needed line voltage exceeds the capabilities.

    Parameters
    ----------
    mass : float
        Mass of the isotope.
    I : integer or half-integer
        Nuclear spin.
    J : list of integers or half-integers
        Electronic spin, ordered as [J :sub:`lower`, J :sub:`upper`].
    ABC : list of floats
        Values for the hyperfine constants, ordered as [A :sub:`lower`,
        A :sub:`upper`, B :sub:`lower`, B :sub:`upper`, C :sub:`lower`,
        C :sub:`upper`].
    isotopeshift : float or :class:`Energy` instance
        Isotope shift of the transition frequency, in MHz or
        :class:`Energy` class.
    lifetime : float
        Lifetime of the excited state, in seconds.
    transitionFreq : float or `Energy` instance
        Frequency of the fine structure transition, in MHz or
        :class:`Energy` class.
    laserFreq : float or `Energy` instance
        Frequency that the laser is set to, in MHz or
        :class:`Energy` class.
    fixedVoltage : float
        Applied voltage in volt.

    Other parameters
    ----------------
    kepcoFactor : float, optional
        Kepco amplification factor, defaults to 50.0.
    channels : int, optional
        Number of points on x-axis for evaluation, defaults to 10000.
    plot: boolean, optional
        If True, a plot of the simulated spectrum will be shown. If the
        voltage excedes the limits of the possible linevoltages
        (meaning -10 < voltages < 10 evaluates as :class:`False` somewhere),
        the plot is drawn in red, otherwise in black. Defaults to
        :class:`False`.
    show: boolean, optional
        If :class:`True`, immediately show a plot of the simulated spectrum.
        Has no effect is :attr:`plot` is :class:`False`. Defaults to
        :class:`True`.
    name : string, optional
        Isotope name. Displayed in the plot title. Defaults to :class:`None`.
    useMatplotLib: boolean, optional
        Use MatplotLib as the graphical engine, otherwise use PyQtGraph.
        Defaults to :class:`False`.

    Returns
    -------
    Linevoltage : NumPy array of linevoltages.
    Response : NumPy array of response to Linevoltage.
    """
    # Check the types of the given frequencies. Convert to MHz if not a float.
    if isinstance(isotopeshift, utilities.Energy):
        isotopeshift = isotopeshift('MHz')
    if isinstance(transitionFreq, utilities.Energy):
        transitionFreq = transitionFreq('MHz')
    if isinstance(laserFreq, utilities.Energy):
        laserFreq = laserFreq('MHz')
    # With the supplied nuclear and atomic information, estimate a spectrum.
    # For fancyness, use the extended Voigt profile.
    SimulatedSpectrum = Spectrum(I, J, ABC, transitionFreq + isotopeshift,
                                 shape='extendedvoigt')
    # Awesome aspect: Lorentz width can be calculated from the lifetime of the
    # excited state!
    loreW = 1.0 / (2 * np.pi * lifetime) * 10 ** -6
    # Assuming some initial energy spread (1 eV is the estimate taken
    # from the master thesis of Hanne Heylen, see the 'dopplerwidth' function),
    # the width of the Gaussian part of the profile can be calculated.
    # This formula is not completely exact; scanning over a voltage range
    # will change the acceleration voltage, which has an effect on the Gaussian
    # width. However, on COLLAPS this scanning range is 1000 Volt, and the
    # acceleration voltage is on the order of 30 to 40 kV. This change of,
    # at most, 3% does not have a strong influence on the calculated width.
    dE = 1.0
    doppW = dopplerwidth(mass, dE, fixedVoltage, transitionFreq + isotopeshift)
    # Adjust the widths of the extended Voigt profile to match the calculated
    # values.
    SimulatedSpectrum.fwhm = [doppW, loreW]

    # Calculate the left-most and right-most transition position in frequency
    # space. Extend this range by 20 times the FWHM of the extended Voigt
    # profile.
    peaks = SimulatedSpectrum.mu
    leftPeak, rightPeak = min(peaks), max(peaks)
    fwhm = SimulatedSpectrum.parts[0].fwhmV
    distance = 20.0 * fwhm
    leftEdge, rightEdge = leftPeak - distance, rightPeak + distance

    # Calculate the additional voltage needed to achieve these specific
    # frequencies, and convert these values to linevoltages.
    _, leftVoltage = voltageshift(mass, leftEdge, laserFreq, fixedVoltage)
    _, rightVoltage = voltageshift(mass, rightEdge, laserFreq, fixedVoltage)
    leftVoltage, rightVoltage = leftVoltage / \
        kepcoFactor, rightVoltage / kepcoFactor
    color = 'k' if (leftVoltage > -10.0 and rightVoltage < 10.0) else 'r'
    if color == 'r':
        print('!!!WARNING!!! The needed line voltage is problematic!')

    # With the calculated values for the line voltage, divide this region
    # in as many channels as given. Given the setup at COLLAPS,
    # these values have to be multiplied by the KEPCO-factor and subtracted
    # from the supplied voltage.
    lineVoltage = np.linspace(leftVoltage, rightVoltage, channels)
    totalVoltage = fixedVoltage - kepcoFactor * lineVoltage

    # The Doppler shift for each applied voltage is different; calculating
    # all shifts and multiplying by the laser frequency allows for 'scanning'
    # of the frequency range. The reaction is calculated by feeding these
    # frequencies into the Spectrum-object.
    dopplerFactor = dopplerfactor(mass, totalVoltage)
    freq = laserFreq * dopplerFactor

    x = lineVoltage
    y = SimulatedSpectrum(freq)

    if plot:
        titleString = 'Simulated spectrum'
        try:
            # Prefer the use of PyQTGraph, which is faster. For publication-
            # like plots, matplotlib is a better choice.
            if useMatplotLib:
                raise ImportError
            from pyqtgraph.Qt import QtGui
            import pyqtgraph as pg

            # My preferred settings.
            pg.setConfigOptions(antialias=True, foreground='k', background='w')

            app = QtGui.QApplication([])
            titleString = 'COLLAPS Simulation' if name is None else name
            win = pg.GraphicsWindow(title=titleString)
            win.resize(1000, 600)
            if name is not None:
                # Add some nice formatting to the name of the isotope.
                name = '<sup>' + name[:-2] + '</sup>' + name[-2:]
                titleString += ' for {name}'.format(name=name)
            p = win.addPlot(title=titleString)
            plotx = np.append(x, x[-1] + x[1] - x[0]) - 0.5 * (x[1] - x[0])
            datacurve = pg.PlotCurveItem(plotx, y, stepMode=True, pen=color)
            # datacurve = pg.PlotCurveItem(x, y, stepMode=False, brush=(50, 50, 200, 100), pen=color)
            p.addItem(datacurve)
            # p.plot(x, y, fillLevel=0.0, brush=(50, 50, 200, 100), pen=color)
            p.setLabel('left', "Intensity", units='arb.')
            p.setLabel('bottom', "Line voltage", units='V')
            p.showGrid(x=True, y=True)
            win.show()
            app.exec_()
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                try:
                    import seaborn
                    seaborn.set()
                except ImportError:
                    pass

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.step(x, y, color)
                if name is not None:
                    # Add some nice formatting to the name of the isotope.
                    name = '$^{' + name[:-2] + '}$' + name[-2:]
                    titleString += ' for {name}'.format(name=name)
                ax.set_xlabel('Line voltage [V]', fontsize=14)
                ax.set_ylabel('Intensity [arb.]', fontsize=14)
                ax.set_title(titleString, fontsize=20)
                ax.grid(True)
                ax.set_ylim(-0.01, 1.0)
                plt.tight_layout()
                if show:
                    plt.show()
            except ImportError:
                pass
    return x, y


def fullprediction(filename):
    """Parses the file given, making an estimate of how the spectrum will
    look like based on the parameters given. The format of the file should look
    like this::

        [General]
        TransitionFrequency : float
        TransitionFrequencyUnit : string
        LaserFrequency : float
        LaserFrequencyUnit : string
        Lifetime : float
        ISOLDEVoltage : float

        [IsotopeName1]
        I : integer or half-integer
        mass : mass in amu
        Jlower : integer or half-integer
        Jupper : integer or half-integer
        Alower : float
        Aupper : float
        Blower : float
        Bupper : float
        Isotopeshift : float
        IsotopeshiftUnit : string

    For each section formatted such as `[IsotopeName1]`, a spectrum is estimated.
    Also given will be the premavoltage needed for each isotope.

    Parameters
    ----------
    filename : string
        Name of the file which contains all the data."""
    if sys.version_info >= (3, 0):
        import configparser
    else:
        import ConfigParser as configparser

    Config = configparser.ConfigParser()
    Config.read(filename)

    trans = eval(Config.get('General', 'TransitionFrequency'))
    laser = eval(Config.get('General', 'LaserFrequency'))
    transUnit = Config.get('General', 'TransitionFrequencyUnit')
    laserUnit = Config.get('General', 'LaserFrequencyUnit')

    lifetime = eval(Config.get('General', 'Lifetime'))
    ISOLDEVoltage = eval(Config.get('General', 'ISOLDEVoltage'))

    trans = utilities.Energy(trans, unit=transUnit)
    laser = utilities.Energy(laser, unit=laserUnit)

    printString = '{name}: Premavoltage {voltageShift} V'
    for section in Config.sections():
        if section == 'General':
            pass
        else:
            name = section
            mass = eval(Config.get(section, 'mass'))
            I = eval(Config.get(section, 'I'))
            J = [eval(Config.get(section, 'Jlower')),
                 eval(Config.get(section, 'Jupper'))]
            AB = [eval(Config.get(section, 'Alower')),
                  eval(Config.get(section, 'Aupper')),
                  eval(Config.get(section, 'Blower')),
                  eval(Config.get(section, 'Bupper')),
                  eval(Config.get(section, 'Clower')),
                  eval(Config.get(section, 'Cupper'))]
            isotopeshift = eval(Config.get(section, 'Isotopeshift'))
            try:
                isotopeshiftUnit = Config.get(section, 'IsotopeshiftUnit')
                isotopeshift = utilities.Energy(isotopeshift,
                                                unit=isotopeshiftUnit)
            except:
                pass
            try:
                channels = eval(Config.get(section, 'Channels'))
            except:
                channels = 400
            try:
                k = eval(Config.get(section, 'Kepco'))
            except:
                k = 50.0
            fixedVoltage, shift = voltageshift(mass,
                                               origVoltage=ISOLDEVoltage,
                                               transitionFreq=trans,
                                               laserFreq=laser)
            predictspectrum(mass, I, J, AB, isotopeshift,
                            lifetime, trans,
                            laser, fixedVoltage,
                            plot=True, name=name, show=False,
                            useMatplotLib=True, channels=channels,
                            kepcoFactor=k)
            print(printString.format(name=name,
                                     voltageShift=shift))
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except ImportError:
        pass
    return None


def interactiveMode(I, J, ABC, f, filename=None, mass=None,
                    transitionFreq=None, laserFreq=None, scalar=46, kepco=50.0):
    """Given the information contained in I, J, AB and f (see the documentation
    of Spectrum for these parameters), a GUI is built with sliders for the most
    important parameters.

    If the optional values filename, mass, transitionFreq and laserFreq are
    given, the .mcp file is loaded, converted to frequency, the laser
    frequency is subtracted and is displayed in the back of the spectrum. If
    this is so, a fit can be performed with the currently selected values.

    Uses PyQtGraph to build the GUI.

    Parameters
    ----------
    I : integer or half-integer
        Nuclear spin.
    J : list of integers or half-integers
        Electronic spin, ordered as [J :sub:`lower`, J :sub:`upper`].
    ABC : list of floats
        Values for the hyperfine constants, ordered as [A :sub:`lower`,
        A :sub:`upper`, B :sub:`lower`, B :sub:`upper`, C :sub:`lower`,
        C :sub:`upper`].
    f : float
        Central frequency of the spectrum.

    Other parameters
    ----------------
    filename : string, optional
        If this is different from :class:`None`, the file is loaded and
        displayed in the background.
    transitionFreq : float or :class:`Energy` instance, optional
        Frequency of the fine structure transition, in MHz or
        :class:`Energy` class. Used to convert the voltage in the
        file to frequency.
    laserFreq : float or :class:`Energy` instance, optional
        Frequency that the laser is set to, in MHz or
        :class:`Energy` class. Used to convert the voltage in the
        file to frequency.
    scalar : float, optional
        Select the scalar in the file. Defaults to 46.
    kepco : float, optional
        Kepco amplification factor, defaults to 50.0."""
    if not filename is None:
        inputs = np.array([mass, transitionFreq, laserFreq])
        inputs_string = ['Mass', 'Transition frequency', 'Laser frequency']
        checks = [val is None for val in inputs]
        if any(checks):
            input = [inputs_string[i] for i, val in enumerate(inputs) if val is None]
            messagestring = '{input!r} not given!'.format(input="', '".join(input))
            raise NameError(messagestring)
        parser = MCPParser()
        measure = parser.feedFile(filename)

        x = measure.x
        y = measure.ySeperate[scalar]

        k = kepco
        voltage = measure.coolerVoltage.mean() - measure.cecVoltage.mean() - k * x
        dopp = dopplerfactor(mass, voltage)

        x = laserFreq * dopp - transitionFreq

        test = Spectrum(I, J, ABC, f, shape='lorentzian')
        test.scale = max(y)
    else:
        test = Spectrum(I, J, ABC, f, shape='lorentzian')
        left, right = min(test.mu), max(test.mu)
        x = np.linspace(left - 600, right + 600, 1000)
    test.fwhm = 100

    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, foreground='k', background='w')

    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.resize(1000, 600)

    l = pg.LayoutWidget()
    win.setCentralWidget(l)

    pw = pg.PlotWidget()
    l.addWidget(pw, colspan=3)
    l.nextRow()
    l.addWidget(QtGui.QLabel(' '))
    l.nextRow()
    if not filename is None:
        datax = x
        x = np.linspace(x[0], x[-1], 1000)
        plotx = np.append(datax, datax[-1] + datax[1] - datax[0]) - 0.5 * (datax[1] - datax[0])
        datacurve = pg.PlotCurveItem(plotx, y, stepMode=True, pen=0.8)
        pw.addItem(datacurve)

    spectrumcurve = pw.plot(x, test(x), pen='r')
    pw.setLabel('left', "Counts (arb.)")
    pw.setLabel('bottom', "Frequency deviation (MHz)")
    pw.showGrid(x=True, y=True, alpha=0.1)

    orient = QtCore.Qt.Horizontal

    label = QtGui.QLabel('Spin')
    spin = pg.SpinBox(value=I, bounds=[0, None], step=0.5)
    l.addWidget(label)
    l.addWidget(spin, colspan=2)
    l.nextRow()

    label = QtGui.QLabel('Amplitude')
    amp = QtGui.QSlider(orient)
    amp.setMinimum(0)
    try:
        amp.setMaximum(2 * max(y))
    except:
        amp.setMaximum(2)
    amp.setValue(int(test.scale))
    ampLabel = QtGui.QLabel(str(amp.value()))
    if not filename is None:
        l.addWidget(label)
        l.addWidget(amp)
        l.addWidget(ampLabel)
        l.nextRow()

    label = QtGui.QLabel('A lower')
    l.addWidget(label)
    Al = QtGui.QSlider(orient)
    Al.setMinimum(-2000)
    Al.setMaximum(2000)
    Al.setValue(int(test.ABC[0]))
    l.addWidget(Al)
    AlLabel = QtGui.QLabel(str(Al.value()))
    l.addWidget(AlLabel)
    l.nextRow()

    label = QtGui.QLabel('A upper')
    l.addWidget(label)
    Au = QtGui.QSlider(orient)
    Au.setMinimum(-2000)
    Au.setMaximum(2000)
    Au.setValue(int(test.ABC[1]))
    l.addWidget(Au)
    AuLabel = QtGui.QLabel(str(Au.value()))
    l.addWidget(AuLabel)
    l.nextRow()

    label = QtGui.QLabel('B lower')
    l.addWidget(label)
    Bl = QtGui.QSlider(orient)
    Bl.setMinimum(-2000)
    Bl.setMaximum(2000)
    Bl.setValue(int(test.ABC[2]))
    l.addWidget(Bl)
    BlLabel = QtGui.QLabel(str(Bl.value()))
    l.addWidget(BlLabel)
    l.nextRow()

    label = QtGui.QLabel('B upper')
    l.addWidget(label)
    Bu = QtGui.QSlider(orient)
    Bu.setMinimum(-2000)
    Bu.setMaximum(2000)
    Bu.setValue(int(test.ABC[3]))
    l.addWidget(Bu)
    BuLabel = QtGui.QLabel(str(Bu.value()))
    l.addWidget(BuLabel)
    l.nextRow()

    label = QtGui.QLabel('FWHM')
    l.addWidget(label)
    fwhm = QtGui.QSlider(orient)
    fwhm.setMinimum(0)
    fwhm.setMaximum(200)
    fwhm.setValue(int(test.fwhm))
    l.addWidget(fwhm)
    fwhmLabel = QtGui.QLabel(str(fwhm.value()))
    l.addWidget(fwhmLabel)
    l.nextRow()

    label = QtGui.QLabel('CoG')
    l.addWidget(label)
    f = QtGui.QSlider(orient)
    f.setMinimum(-10000)
    f.setMaximum(10000)
    f.setValue(int(test.df))
    l.addWidget(f)
    fLabel = QtGui.QLabel(str(f.value()))
    l.addWidget(fLabel)
    l.nextRow()

    fitButton = QtGui.QPushButton('Fit it!')
    if not filename is None:
        l.addWidget(fitButton, colspan=3)
        l.nextRow()

    printButton = QtGui.QPushButton('Print it!')
    l.addWidget(printButton, colspan=3)

    def updateSpectrum():
        test.ABC = [Al.value(), Au.value(),
                    Bl.value(), Bu.value(),
                    0, 0]
        test.fwhm = fwhm.value()
        test.df = f.value()
        test.scale = amp.value()
        # if filename is None:
        #     left, right = min(test.mu), max(test.mu)
        #     x = np.linspace(left - 600, right + 600, 1000)
        spectrumcurve.setData(x, test(x))

        ampLabel.setText(str(amp.value()))
        AlLabel.setText(str(Al.value()))
        AuLabel.setText(str(Au.value()))
        BlLabel.setText(str(Bl.value()))
        BuLabel.setText(str(Bu.value()))
        fwhmLabel.setText(str(fwhm.value()))
        fLabel.setText(str(f.value()))

    def fit():
        test.FitToSpectroscopicData(datax, y)

        Al.setValue(test.ABC[0])
        Au.setValue(test.ABC[1])
        Bl.setValue(test.ABC[2])
        Bu.setValue(test.ABC[3])

        amp.setValue(test.scale)
        fwhm.setValue(test.fwhm)
        f.setValue(test.df)

        updateSpectrum()

    def printIt():
        names = ['Amplitude', 'Al', 'Au', 'Bl', 'Bu', 'FWHM', 'df']
        widg = [amp, Al, Au, Bl, Bu, fwhm, f]
        maxlen = 0
        for i in names:
            maxlen = max(maxlen, len(i))
        message = '{:<' + str(maxlen) + '} :\t{: .5f}'
        for n, w in zip(names, widg):
            print(message.format(n, w.value()))

    def checkSpins(self):
        valCheck = spin.value()
        if int(valCheck) == valCheck or int(2 * valCheck) == 2 * valCheck:
            pass
        else:
            spin.setValue(int(valCheck))
        test.I = spin.value()
        updateSpectrum()

    spin.sigValueChanged.connect(checkSpins)

    amp.sliderMoved.connect(updateSpectrum)
    Al.sliderMoved.connect(updateSpectrum)
    Au.sliderMoved.connect(updateSpectrum)
    Bl.sliderMoved.connect(updateSpectrum)
    Bu.sliderMoved.connect(updateSpectrum)
    fwhm.sliderMoved.connect(updateSpectrum)
    f.sliderMoved.connect(updateSpectrum)

    fitButton.clicked.connect(fit)
    printButton.clicked.connect(printIt)

    win.show()
    app.exec_()
