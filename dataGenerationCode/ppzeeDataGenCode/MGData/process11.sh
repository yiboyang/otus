# Load Delphes File Locations
source init.sh
# Make sure to use proper version of python (virtual env messed this up)
python --version
source /data/apps/anaconda/3.6-5.0.1/etc/profile.d/conda.sh
conda deactivate
python --version
# Generate data iseed = 11
~/Software/MG5_aMC_v2_6_3_2/bin/mg5_aMC 11_10k_ppzee.txt
