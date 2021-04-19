#!/bin/bash
#$ -N ppttbar2_LatentData
#$ -q free64
#$ -m beas

# Replace "date" with your program. Output goes to out
./ppttbar_runLHEtoHDF5.sh  > log.txt

#sleep 60
