#!/bin/bash
#$ -N ppzee_LatentData
#$ -q free64
#$ -m beas

# Replace "date" with your program. Output goes to out
./ppzee_runLHEtoHDF5.sh  > log.txt

#sleep 60
