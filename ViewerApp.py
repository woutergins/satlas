from PyQt4 import QtCore,QtGui
from satlas.analysisIO import load
from PyQt4 import QtCore,QtGui
import sys

class Viewer(QtGui.QMainWindow):
    def __init__(self):
        super(Viewer, self).__init__()

        menubar = self.menuBar()

        self.openAction = QtGui.QAction('&Open',self)
        self.openAction.setShortcut('Ctrl+O')
        self.openAction.setStatusTip('Choose analysis file to open')
        self.openAction.triggered.connect(self.open)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.openAction)

        self.tabs = QtGui.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.setGeometry(QtCore.QRect(128, 128, 800, 600))

        self.show()

        self.open()

    def open(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                            filter = "Analysis (*.analysis)")

        self.analysis = load(fname)

        self.make_analysis_view()

    def make_analysis_view(self):
        self.tabs.clear()
        for n,s in sorted(self.analysis.items()):
            fig, ax = self.analysis.plot_spectrum(n,show=False)
            self.tabs.addTab(ApproachOverview(s,fig),n)

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

class ApproachOverview(QtGui.QWidget):
    def __init__(self,spectrum,fig):
        super(ApproachOverview,self).__init__()
        
        self.spectrum = spectrum

        self.layout = QtGui.QGridLayout(self)
        self.parWidget = QtGui.QTableWidget()
        self.layout.addWidget(self.parWidget,0,0)
        self.populateTable()

        figureWidget = QtGui.QWidget()
        figureLayout = QtGui.QVBoxLayout()
        
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        figureLayout.addWidget(canvas)
        figureLayout.addWidget(toolbar)

        self.layout.addLayout(figureLayout,0,1)

    def populateTable(self):
        names,values,errors = self.spectrum.vars()
        self.parWidget.setRowCount(len(names))
        self.parWidget.setColumnCount(3)
        for i,(n,v,ve) in enumerate(zip(names,values,errors)):
            self.parWidget.setItem(i,0,QtGui.QTableWidgetItem(n))
            self.parWidget.setItem(i,1,QtGui.QTableWidgetItem(str(v)))
            self.parWidget.setItem(i,2,QtGui.QTableWidgetItem(str(ve)))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    m = Viewer()
    sys.exit(app.exec_())
