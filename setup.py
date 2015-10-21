from distutils.core import setup
setup(
  name='satlas',
  packages=['satlas'], # this must be the same as the name above
  version='0.1.0b1',
  description='This Python package has been created with the goal of creating an easier interface for the analysis of data gathered from laser spectroscopy experiments. Support for fitting the spectra, using both chi2-fitting and Maximum Likelihood Estimation routines, are present.',
  author='Wouter Gins',
  author_email='woutergins@gmail.com',
  url='https://woutergins.github.io/satlas/',
  download_url='https://github.com/woutergins/satlas/tarball/0.1', # I'll explain this in a second
  keywords=['physics', 'hyperfine structure', 'fitting'], # arbitrary keywords
  install_requires=['numpy',
                    'scipy',
                    'sympy',
                    'matplotlib',
                    'pandas',
                    'emcee',
                    'lmfit',
                    'progressbar2'],
  classifiers=['Classifier: Development Status :: 4 - Beta',
               'Classifier: Intended Audience :: Science/Research',
               'Classifier: License :: OSI Approved :: MIT License',
               'Classifier: Operating System :: Microsoft :: Windows',
               'Classifier: Programming Language :: Python :: 2',
               'Classifier: Programming Language :: Python :: 3',
               'Classifier: Topic :: Scientific/Engineering :: Physics'],
)