###############################  
#-- Load Required Libraries --# 
###############################  

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import h5py

################# 
#-- Functions --# 
################# 
def getMGData(f):

    #-- Get the keys --#
    gkey = list(f.keys())[0]
    pkey = list(f.keys())[1]

    #-- Get the data --#
    ggp = f.get(gkey)
    gdata = ggp.get('global_event_data')

    pgp = f.get(pkey)
    pdata = pgp.get('particle_event_data')
    
    #-- Reshape particle data into an intuitive format --#
    pdata_flat = [item for sublist in pdata for item in sublist]
    
    newpdata = np.array(pdata_flat).reshape((-1,15))
    # Note on Structure:
    # newpdata[0],   [1],      [2],    [3],     [4],     [5],    [6],    [7], [8], [9], [10], [11], [12],  [13],  [14]
    #     event #, row #, PDG_code, status, parent1, parent2, color1, color2,  px,  py,  pz,     E, mass, spin1, spin2 
    
    #-- DEBUG -- Print out 1 event to make sure everything makes sense --#
    #print(np.array(pdata[0]).reshape((-1,15)))
    
    #-- Select out latent information --#
    # Note on Structure:                
    #     [0], [1], [2], [3], [4],  [5] 
    # event #,  px,  py,  pz,   E, mass 
    # PID for electron/positron is +-11
    eleMsk    = np.ma.masked_where(newpdata[:,2]==11, newpdata[:,2]).mask
    posMsk    = np.ma.masked_where(newpdata[:,2]==-11, newpdata[:,2]).mask
 
    fnalStMsk = np.ma.masked_where(newpdata[:,3]==1, newpdata[:,3]).mask
    
    fEleMsk = np.logical_and(eleMsk, fnalStMsk)
    fPosMsk = np.logical_and(posMsk, fnalStMsk)
    
    MGeleInfo = np.concatenate([newpdata[fEleMsk,0].reshape(-1,1), newpdata[fEleMsk,8:13]],axis=1)
    MGposInfo = np.concatenate([newpdata[fPosMsk,0].reshape(-1,1), newpdata[fPosMsk,8:13]],axis=1)
    
    return newpdata, MGeleInfo, MGposInfo


def processMGData(MGfList, NfList):
    
    assert(len(MGfList)==len(NfList))
    
    selLatentInfoList = []
    
    for i in range(len(NfList)):
        
        #-- Get MG data --#
        newpdata, MGeleInfo, MGposInfo = getMGData(MGfList[i])
        
        #-- Select madgraph data such that we only have the events which passed the above requirement --#
      
        # Note on Structure:
        #Nf: evt# Particle number per event: e mu photon jet met => Nf1.values[:,0] = array of passing event numbers
        #MGeleInfo: evt#, px,  py,  pz, E, mass
        selMGeleInfo = MGeleInfo[NfList[i].values[:,0],:]
        selMGposInfo = MGposInfo[NfList[i].values[:,0],:]
        
        #-- DBUG --#
        #print('i =',i)
        #print(NfList[i].values[0:10,0]) 
        #print(selMGbInfo[0:10,0]) 
        #print('')  

        #-- Concatenate these together event by event --#
        selLatentInfo = np.concatenate([selMGeleInfo[:,1:5],selMGposInfo[:,1:5]],axis=1)
        
        #-- Add result to final list --#
        selLatentInfoList.append(selLatentInfo)
        
        #-- Add progress print statement --#
        print('Finished event ',i+1,' out of ',len(NfList))
    
    zData = np.concatenate(selLatentInfoList,axis=0)
    
    return zData

def make4Vector(l, m):
    #l = [PT, Eta, Phi] shape (N,3)
    #m = mass (0 for electrons and met)
    
    # px = PT*cos(Phi)
    # py = PT*sin(Phi)
    # pz = PT*sinh(Eta)
    # E  = sqrt(m**2 + (PT*cosh(Eta))**2)
    
    PT  = l[:,0]
    Eta = l[:,1]
    Phi = l[:,2]
    
    N = PT.shape[0]
    
    px = PT*np.cos(Phi)
    py = PT*np.sin(Phi)
    pz = PT*np.sinh(Eta)
    E  = np.sqrt(m**2 + (PT*np.cosh(Eta))**2)
    
    return np.concatenate([px.reshape(-1,1),py.reshape(-1,1),pz.reshape(-1,1),E.reshape(-1,1)],axis=1)

def processDelphesData(DfList):
    selDataInfoList = []
    
    eleInfoList = []
    
    for i in range(len(DfList)):
        
        #-- Take Delphes and rearrange so that it looks like e- info, e+ info, met info --#
        
        # Select out each type and turn into 4 vector
        a = DfList[i].values[:,1]
        eleMsk = np.ma.masked_where(a==11, a).mask
        selEleData = make4Vector(DfList[i].values[eleMsk,2:5],0) 
        #print(selEleData.shape)
        
        posMsk = np.ma.masked_where(a==-11, a).mask
        selPosData = make4Vector(DfList[i].values[posMsk,2:5],0)
        #print(selPosData.shape)
        
        metMsk = np.ma.masked_where(a==99, a).mask
        selMetData = make4Vector(DfList[i].values[metMsk,2:5],0)
        #print(selMetData.shape)
        
        #-- Concatenate them together --#
        selData = np.concatenate([selEleData,selPosData,selMetData],axis=1)
        
        #-- Add result to running list --#
        selDataInfoList.append(selData)
        
        #-- Add progress print statement --#
        print('Finished event ',i+1,' out of ',len(NfList))
    
    xData = np.concatenate(selDataInfoList,axis=0)
    
    return xData

#-----------------------------------------------------------------------------------------------  
#-----------------------------------------------------------------------------------------------  

##########################
#-- Load Initial Files --#
##########################

MGfile_string = '../LatentData/HDF5Data/'
Dfile_string  = '../DelphesData/csv_ppzee/'

#-- Load h5py files --#
MGf10 = h5py.File(MGfile_string + 'file_out10.hdf5', 'r')
MGf11 = h5py.File(MGfile_string + 'file_out11.hdf5', 'r')
MGf12 = h5py.File(MGfile_string + 'file_out12.hdf5', 'r')
MGf13 = h5py.File(MGfile_string + 'file_out13.hdf5', 'r')
MGf14 = h5py.File(MGfile_string + 'file_out14.hdf5', 'r')
MGf15 = h5py.File(MGfile_string + 'file_out15.hdf5', 'r')
MGf16 = h5py.File(MGfile_string + 'file_out16.hdf5', 'r')


MGfList = [MGf10, MGf11, MGf12, MGf13, MGf14, MGf15, MGf16]

#-- Load in Delphes data files --#
#Df: # evt# PID PT Eta Phi
Df10 = pd.read_csv(Dfile_string + 'out_file10.csv', header=None)
Df11 = pd.read_csv(Dfile_string + 'out_file11.csv', header=None)
Df12 = pd.read_csv(Dfile_string + 'out_file12.csv', header=None)
Df13 = pd.read_csv(Dfile_string + 'out_file13.csv', header=None)
Df14 = pd.read_csv(Dfile_string + 'out_file14.csv', header=None)
Df15 = pd.read_csv(Dfile_string + 'out_file15.csv', header=None)
Df16 = pd.read_csv(Dfile_string + 'out_file16.csv', header=None)

DfList = [Df10, Df11, Df12, Df13, Df14, Df15, Df16]

#-- Load in Delphes number files --#
#Nf: # evt# Particle number per event: e mu photon jet met 
Nf10 = pd.read_csv(Dfile_string + 'numberperevt_out_file10.csv', header=None)
Nf11 = pd.read_csv(Dfile_string + 'numberperevt_out_file11.csv', header=None)
Nf12 = pd.read_csv(Dfile_string + 'numberperevt_out_file12.csv', header=None)
Nf13 = pd.read_csv(Dfile_string + 'numberperevt_out_file13.csv', header=None)
Nf14 = pd.read_csv(Dfile_string + 'numberperevt_out_file14.csv', header=None)
Nf15 = pd.read_csv(Dfile_string + 'numberperevt_out_file15.csv', header=None)
Nf16 = pd.read_csv(Dfile_string + 'numberperevt_out_file16.csv', header=None)

NfList = [Nf10, Nf11, Nf12, Nf13, Nf14, Nf15, Nf16]

######################## 
#-- Process FDL Data --#
######################## 
print('Processing zData')
zData = processMGData(MGfList, NfList)

######################## 
#-- Process ROL Data --#
######################## 
print('Processing xData')
xData = processDelphesData(DfList)

#################
#-- Save Data --#
#################

#-- Create output file with structure --#
print('Creating output file: FinalData_ppzee.hdf5')
file_out = h5py.File('FinalData_ppzee.hdf5', 'w')   #write only mode for output
 
#-- Write data to output file --#
print('Writing data to output file')
zgrp = file_out.create_group("FDL")
xgrp = file_out.create_group("ROL") 

#-- Create datasets --#
# Note in this case all xData events have the same dimension (much easier to add to dataset)
# This won't be the case in pp>ttbar, must do a trick
zdset = zgrp.create_dataset("zData", data=zData)
xdset = xgrp.create_dataset("xData", data=xData)

#-- Close file --#
print('Closing file!')
file_out.close()
