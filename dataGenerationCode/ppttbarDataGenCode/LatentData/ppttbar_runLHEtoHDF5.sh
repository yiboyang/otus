
# Make sure to use proper version of python (virtual env messed this up)
python --version
source /data/apps/anaconda/3.6-5.0.1/etc/profile.d/conda.sh
conda activate MLS

#-- run_05 --#
# 10k events
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip /pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_05/unweighted_events.lhe.gz
python runLHEtoHDF5_ppttbar.py '/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_05/unweighted_events.lhe'
mv file_out.hdf5 file_out5.hdf5

#-- run_06 --#                
# 600k events
# Unzip file, run LHE to HDF5, move resulting output file
#gunzip /pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_06/unweighted_events.lhe.gz
python runLHEtoHDF5_ppttbar.py '/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_06/unweighted_events.lhe'
mv file_out.hdf5 file_out6.hdf5

#-- run_07 --#
# 600k events
# Unzip file, run LHE to HDF5, move resulting output file
gunzip /pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_07/unweighted_events.lhe.gz
python runLHEtoHDF5_ppttbar.py '/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_07/unweighted_events.lhe'
mv file_out.hdf5 file_out7.hdf5

#-- run_08 --#
# 600k events                                                           
# Unzip file, run LHE to HDF5, move resulting output file               
gunzip /pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_08/unweighted_events.lhe.gz
python runLHEtoHDF5_ppttbar.py '/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_08/unweighted_events.lhe'
mv file_out.hdf5 file_out8.hdf5

#-- run_09 --#                                                          
# 600k events                                                           
# Unzip file, run LHE to HDF5, move resulting output file               
gunzip /pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_09/unweighted_events.lhe.gz
python runLHEtoHDF5_ppttbar.py '/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_09/unweighted_events.lhe'
mv file_out.hdf5 file_out9.hdf5



mkdir -p ./HDF5Data
mv *.hdf5 ./HDF5Data