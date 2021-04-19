# Make sure to use proper version of python (virtual env messed this up)
python --version
source /data/apps/anaconda/3.6-5.0.1/etc/profile.d/conda.sh
conda activate MLS

#-- run_10 --#
# 10k events
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip ../MGData/ppzee/Events/run_10/unweighted_events.lhe.gz
python runLHEtoHDF5_ppzee.py '../MGData/ppzee/Events/run_10/unweighted_events.lhe'
mv file_out.hdf5 file_out10.hdf5

#-- run_11 --#
# 30k events
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip ../MGData/ppzee/Events/run_11/unweighted_events.lhe.gz
python runLHEtoHDF5_ppzee.py '../MGData/ppzee/Events/run_11/unweighted_events.lhe'
mv file_out.hdf5 file_out11.hdf5

#-- run_12 --#
# 90k events
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip ../MGData/ppzee/Events/run_12/unweighted_events.lhe.gz
python runLHEtoHDF5_ppzee.py '../MGData/ppzee/Events/run_12/unweighted_events.lhe'
mv file_out.hdf5 file_out12.hdf5

#-- run_13 --#
# 160k events
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip ../MGData/ppzee/Events/run_13/unweighted_events.lhe.gz
python runLHEtoHDF5_ppzee.py '../MGData/ppzee/Events/run_13/unweighted_events.lhe'
mv file_out.hdf5 file_out13.hdf5

#-- run_14 --#                  
# 600k events
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip ../MGData/ppzee/Events/run_14/unweighted_events.lhe.gz
python runLHEtoHDF5_ppzee.py '../MGData/ppzee/Events/run_14/unweighted_events.lhe'
mv file_out.hdf5 file_out14.hdf5

#-- run_15 --#
# 600k events 
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip ../MGData/ppzee/Events/run_15/unweighted_events.lhe.gz
python runLHEtoHDF5_ppzee.py '../MGData/ppzee/Events/run_15/unweighted_events.lhe'
mv file_out.hdf5 file_out15.hdf5

#-- run_16 --#
# 160k events 
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip ../MGData/ppzee/Events/run_16/unweighted_events.lhe.gz
python runLHEtoHDF5_ppzee.py '../MGData/ppzee/Events/run_16/unweighted_events.lhe'
mv file_out.hdf5 file_out16.hdf5

mkdir -p ./HDF5Data
mv *.hdf5 ./HDF5Data
