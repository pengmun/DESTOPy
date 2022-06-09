# DESTOPy - Atmospheric Density Estimation Toolbox in Python
The Atmospheric Density Estimation Toolbox in python (DESTOPy) is a python toolbox for estimation of the global thermospheric mass density based on the Matlab [Density Estimation Toolbox (DESTO)](https://github.com/davidgondelach/DESTO) by David Gondelach and Richard Linares. 

Currently, only a single reduced-order density model (JB2008 reduced-order model) is available and the python toolbox only supports assimilation of Two-Line Element (TLE) sets. 

Copyright &copy; 2021 by Peng Mun Siew and Richard Linares

## License
The code is licensed under GNU General Public License version 3 - see the [LICENSE](https://github.com/pengmun/DESTOPy/blob/main/LICENSE) file for details.

## Acknowledgments
The code is developed based upon work supported by the National Science Foundation under award NSF-PHY-2028125.

The python scripts are coded based on work taken from the Matlab [Density Estimation Toolbox (DESTO)](https://github.com/davidgondelach/DESTO) by David Gondelach and Richard Linares.

The Jacchia-Bowman 2008 model was downloaded from the [ATMOS](https://github.com/lcx366/ATMOS) python package by Chunxiao Li.

The solar radiation pressure and third-body perturbations models are adapted from the Matlab code by Meysam Mahooti (copyright 2018) and were downloaded from [Matlab File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/55167-high-precision-orbit-propagator).

The SGP4 model used was based on the [python-sgp4](https://github.com/brandon-rhodes/python-sgp4) module by Brandon Rhodes.

Several time and reference frame routines are adapted from the Matlab code developed by David Vallado (and others) and were downloaded from https://celestrak.com/software/vallado-sw.php.

The Earth Gravitational Model 2008 (EGM2008) coefficients were obtained from the NGA's Office of Geomatics: https://earth-info.nga.mil.

The toolbox also makes use of NASA's [SPICE toolkit (N66)](https://naif.jpl.nasa.gov/naif/toolkit.html) via the SpiceyPy module by Annex et al. available [here](https://github.com/AndrewAnnex/SpiceyPy).

Some of the scripts also take advantage of [Numba](https://numba.pydata.org/numba-doc/dev/index.html) for faster computations.

## References
The thermospheric density modeling and estimation techniques using Two-Line Element (TLE) sets are described in:
```
@article{gondelach2020tle,
  author = {Gondelach, David J. and Linares, Richard},
  title = {Real-Time Thermospheric Density Estimation Via Two-Line-Element Data Assimilation},
  journal = {Space Weather},
  volume = {18},
  number = {2},
  pages = {e2019SW002356},
  doi = {10.1029/2019SW002356},
  url = {https://doi.org/10.1029/2019SW002356}
}
```
see https://doi.org/10.1029/2019SW002356.

### Dependencies (partial list)
* python
* numpy
* scipy
* matplotlib
* spiceypy
* datetime
* sgp4
* numba
* h5py

## Installation instructions
1. Download the DESTOPy python code.
2. Install the necessary dependencies.
3. Download SPICE kernels (i.e. ephemeris files) from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/ and put them in the folder Data. See **Ephemeris files** section below.
4. Download space weather file from Celestrak and put in folder Data: https://www.celestrak.com/SpaceData/SW-All.txt
5. Download Earth orientation data file from Celestrak and put in folder Data: https://www.celestrak.com/SpaceData/EOP-All.txt
6. Download 2 space weather files needed for the JB2008 model and put in folder Data: http://sol.spacenvironment.net/jb2008/indices/SOLFSMY.TXT and http://sol.spacenvironment.net/jb2008/indices/DTCFILE.TXT
7. (Optional) Download new TLE data and put them in the folder TLEdata.
8. For each space object used for estimation, ensure that their ballistic coefficient (BC) are specified in the text file: Data/BCdata.txt

## Run instructions
1. Open the DensityEstimationTLE.ipynb or SO_propagation.ipynb using jupyter notebook.
2. (Optional) Under section **1.0.1 Settings**, select the estimation window (start year, start month, start day, number of days, and number of hours).
4. (Optional) Under section **1.0.1 Settings**, select the reduced-order model dimensions. 
5. (Optional) Under section **1.0.1 Settings**, select the space objects to use for density estimation.
6. Run all cells.
  
## Ephemeris files
Download the following ephemeris files and put them in the Data folder:
* latest_leapseconds.tls: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/
* de430.bsp: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
* earthstns_itrf93_201023.bsp: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/stations/
* pck00010.tpc, earth_fixed.tf, earth_200101_990628_predict.bpc, earth_000101_210530_210308.bpc, earth_720101_070426.bpc, earth_latest_high_prec.bpc: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/

Python 3.7 was used to develop the code.

Peng Mun Siew, Jun 2021 email: siewpm@mit.edu
