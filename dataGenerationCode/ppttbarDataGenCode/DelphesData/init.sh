# source init.sh

#export DELPHES=/c/Users/tdhttt/workspace/hep/delphes/Delphes-3.4.1
export DELPHES=~/Software/MG5_aMC_v2_6_3_2/Delphes/
export LD_LIBRARY_PATH=$DELPHES:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=./:$DELPHES:$DELPHES/external:$ROOT_INCLUDE_PATH