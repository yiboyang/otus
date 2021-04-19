
# Make sure to use proper version of python (virtual env messed this up)
python --version
source /data/apps/anaconda/3.6-5.0.1/etc/profile.d/conda.sh
conda activate MLS

python FinalData_ppzee.py

mkdir -p ./FinalData
mv *.hdf5 ./FinalData