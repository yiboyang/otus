source init.sh

root -l -b -q ./harvestDelphes.C'("../MGData/ppzee/Events/run_10/tag_2_delphes_events.root")'
mv out_file.csv out_file10.csv
mv numberperevt_out_file.csv numberperevt_out_file10.csv

root -l -b -q ./harvestDelphes.C'("../MGData/ppzee/Events/run_11/tag_1_delphes_events.root")'
mv out_file.csv out_file11.csv
mv numberperevt_out_file.csv numberperevt_out_file11.csv

root -l -b -q ./harvestDelphes.C'("../MGData/ppzee/Events/run_12/tag_1_delphes_events.root")'
mv out_file.csv out_file12.csv
mv numberperevt_out_file.csv numberperevt_out_file12.csv

root -l -b -q ./harvestDelphes.C'("../MGData/ppzee/Events/run_13/tag_1_delphes_events.root")'
mv out_file.csv out_file13.csv
mv numberperevt_out_file.csv numberperevt_out_file13.csv

root -l -b -q ./harvestDelphes.C'("../MGData/ppzee/Events/run_14/tag_1_delphes_events.root")'
mv out_file.csv out_file14.csv
mv numberperevt_out_file.csv numberperevt_out_file14.csv

root -l -b -q ./harvestDelphes.C'("../MGData/ppzee/Events/run_15/tag_1_delphes_events.root")'
mv out_file.csv out_file15.csv
mv numberperevt_out_file.csv numberperevt_out_file15.csv

root -l -b -q ./harvestDelphes.C'("../MGData/ppzee/Events/run_16/tag_1_delphes_events.root")'
mv out_file.csv out_file16.csv
mv numberperevt_out_file.csv numberperevt_out_file16.csv

mkdir -p ./csv_ppzee
mv *.csv ./csv_ppzee
