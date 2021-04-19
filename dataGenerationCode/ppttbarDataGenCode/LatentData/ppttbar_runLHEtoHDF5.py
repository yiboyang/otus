import sys
import numpy as np
import h5py
import LHEtoHDF5

#-- Run main program --#
print('Running LHE to HDF5 on file ', sys.argv[1])
LHEfileName = sys.argv[1]
LHEtoHDF5.main(LHEfileName)
