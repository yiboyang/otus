#!/bin/bash
#$ -N ppttbar2_DelphesData
#$ -q free64
#$ -m beas

# Replace "date" with your program. Output goes to out
./harvestDelphes.sh  > log.txt

#sleep 60
