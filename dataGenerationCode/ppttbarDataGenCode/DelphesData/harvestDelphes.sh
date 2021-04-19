source init.sh

# Make sure to use proper version of python (virtual env messed this up)
python --version
source /data/apps/anaconda/3.6-5.0.1/etc/profile.d/conda.sh
conda deactivate

# Event 5
root -l -b -q ./harvestDelphes.C'("/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_05/tag_1_delphes_events.root")'
mv out_file.csv out_file5.csv 
mv numberperevt_out_file.csv numberperevt_out_file5.csv

# Event 6
root -l -b -q ./harvestDelphes.C'("/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_06/tag_1_delphes_events.root")'
mv out_file.csv out_file6.csv
mv numberperevt_out_file.csv numberperevt_out_file6.csv

# Event 7
root -l -b -q ./harvestDelphes.C'("/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_07/tag_1_delphes_events.root")'
mv out_file.csv out_file7.csv
mv numberperevt_out_file.csv numberperevt_out_file7.csv

# Event 8
root -l -b -q ./harvestDelphes.C'("/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_08/tag_1_delphes_events.root")'
mv out_file.csv out_file8.csv
mv numberperevt_out_file.csv numberperevt_out_file8.csv

# Event 9
root -l -b -q ./harvestDelphes.C'("/pub/jnhoward/FastSim/ppttbarData/MGData2/ppttbar_2/Events/run_09/tag_1_delphes_events.root")'
mv out_file.csv out_file9.csv
mv numberperevt_out_file.csv numberperevt_out_file9.csv



mkdir -p ./csv_ppttbar
mv *.csv ./csv_ppttbar