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

    
    #-- DEBUG: Print out 1 event to make sure everything makes sense --#
    #print(np.array(pdata[0]).reshape((-1,15)))
    
    #-- Select out latent information --#
    # Note on Structure:
    #     [0], [1], [2], [3], [4],  [5]
    # event #,  px,  py,  pz,   E, mass
    eleMsk    = np.ma.masked_where(newpdata[:,2]==11, newpdata[:,2]).mask
    nubareMsk = np.ma.masked_where(newpdata[:,2]==-12, newpdata[:,2]).mask
    bMsk      = np.ma.masked_where(newpdata[:,2]==5, newpdata[:,2]).mask
    bbarMsk   = np.ma.masked_where(newpdata[:,2]==-5, newpdata[:,2]).mask
    uMsk      = np.ma.masked_where(newpdata[:,2]==2, newpdata[:,2]).mask
    dbarpMsk  = np.ma.masked_where(newpdata[:,2]==-1, newpdata[:,2]).mask

    fnalStMsk = np.ma.masked_where(newpdata[:,3]==1, newpdata[:,3]).mask
    
    feleMsk    = np.logical_and(eleMsk, fnalStMsk)
    fnubareMsk = np.logical_and(nubareMsk, fnalStMsk)
    fbMsk      = np.logical_and(bMsk, fnalStMsk)
    fbbarMsk   = np.logical_and(bbarMsk, fnalStMsk)
    fuMsk      = np.logical_and(uMsk, fnalStMsk)
    fdbarMsk   = np.logical_and(dbarpMsk, fnalStMsk)

    MGeleInfo = np.concatenate([newpdata[feleMsk,0].reshape(-1,1), newpdata[feleMsk,8:13]],axis=1)
    MGnubareInfo   = np.concatenate([newpdata[fnubareMsk,0].reshape(-1,1), newpdata[fnubareMsk,8:13]],axis=1)
    MGbInfo   = np.concatenate([newpdata[fbMsk,0].reshape(-1,1), newpdata[fbMsk,8:13]],axis=1)
    MGbbarInfo   = np.concatenate([newpdata[fbbarMsk,0].reshape(-1,1), newpdata[fbbarMsk,8:13]],axis=1)
    MGuInfo   = np.concatenate([newpdata[fuMsk,0].reshape(-1,1), newpdata[fuMsk,8:13]],axis=1)
    MGdbarInfo   = np.concatenate([newpdata[fdbarMsk,0].reshape(-1,1), newpdata[fdbarMsk,8:13]],axis=1)

    return newpdata, MGeleInfo, MGnubareInfo, MGbInfo, MGbbarInfo, MGuInfo, MGdbarInfo

def processMGData(MGfList, NfList):
    
    assert(len(MGfList)==len(NfList))
    
    selLatentInfoList = []
    
    for i in range(len(NfList)):
        
        #-- Get MG data --#
        newpdata, MGeleInfo, MGnubareInfo, MGbInfo, MGbbarInfo, MGuInfo, MGdbarInfo = getMGData(MGfList[i])
        
        #-- Select madgraph data such that we only have the events which passed the above requirement --#

        # Note on Structure:
        #Nf: evt# Particle number per event: e mu photon jet met => Nf1.values[:,0] = array of passing event numbers
        #MGeleInfo: evt#, px,  py,  pz, E, mass

        selMGeleInfo    = MGeleInfo[NfList[i].values[:,0],:]
        selMGnubareInfo = MGnubareInfo[NfList[i].values[:,0],:]
        selMGbInfo      = MGbInfo[NfList[i].values[:,0],:]
        selMGbbarInfo   = MGbbarInfo[NfList[i].values[:,0],:]
        selMGuInfo      = MGuInfo[NfList[i].values[:,0],:]
        selMGdbarInfo   = MGdbarInfo[NfList[i].values[:,0],:]
              
        #-- DBUG --#
        #print('i =',i)
        #print(NfList[i].values[0:10,0])
        #print(selMGbInfo[0:10,0])
        #print('')

        #-- Concatenate these together event by event --#
        selLatentInfo = np.concatenate([selMGeleInfo[:,1:5], 
                                        selMGnubareInfo[:,1:5], 
                                        selMGbInfo[:,1:5], 
                                        selMGbbarInfo[:,1:5],
                                        selMGuInfo[:,1:5], 
                                        selMGdbarInfo[:,1:5]], axis=1)
        
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
        
        #-- Take Delphes and rearrange so that it looks like e- info, met info, jet info --#
        
        # Select out each type and turn into 4 vector
        a = DfList[i].values[:,1]
        eleMsk = np.ma.masked_where(a==11, a).mask
        selEleData = make4Vector(DfList[i].values[eleMsk,2:5],0) 
        #print(selEleData.shape)
        
        metMsk = np.ma.masked_where(a==99, a).mask
        selMetData = make4Vector(DfList[i].values[metMsk,2:5],0)
        #print(selMetData.shape)

        jetMsk = np.ma.masked_where(a==88, a).mask
        selJetData = make4Vector(DfList[i].values[jetMsk,2:5],0)
        #print(selJetData.shape)
        #print(selJetData.reshape(-1,4*4).shape)

        #-- Concatenate them together --#
        selData = np.concatenate([selEleData,selMetData,selJetData.reshape(-1,4*4)],axis=1)
        
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
Dfile_string  = '../DelphesData/csv_ppttbar/'

#-- Load h5py files --#
MGf5 = h5py.File(MGfile_string + 'file_out5.hdf5', 'r')
MGf6 = h5py.File(MGfile_string + 'file_out6.hdf5', 'r')
MGf7 = h5py.File(MGfile_string + 'file_out7.hdf5', 'r')
MGf8 = h5py.File(MGfile_string + 'file_out8.hdf5', 'r')
MGf9 = h5py.File(MGfile_string + 'file_out9.hdf5', 'r')

MGfList = [MGf5, MGf6, MGf7, MGf8, MGf9]

#-- Load csv data files --#
#Df: # evt# PID PT Eta Phi
Df5 = pd.read_csv(Dfile_string + 'out_file5.csv', header=None)
Df6 = pd.read_csv(Dfile_string + 'out_file6.csv', header=None)
Df7 = pd.read_csv(Dfile_string + 'out_file7.csv', header=None)
Df8 = pd.read_csv(Dfile_string + 'out_file8.csv', header=None)
Df9 = pd.read_csv(Dfile_string + 'out_file9.csv', header=None)

DfList = [Df5, Df6, Df7, Df8, Df9]

#-- Load csv number files --#
#Nf: # evt# Particle number per event: e mu photon jet met 
Nf5 = pd.read_csv(Dfile_string + 'numberperevt_out_file5.csv', header=None)
Nf6 = pd.read_csv(Dfile_string + 'numberperevt_out_file6.csv', header=None)
Nf7 = pd.read_csv(Dfile_string + 'numberperevt_out_file7.csv', header=None)
Nf8 = pd.read_csv(Dfile_string + 'numberperevt_out_file8.csv', header=None)
Nf9 = pd.read_csv(Dfile_string + 'numberperevt_out_file9.csv', header=None)

NfList = [Nf5, Nf6, Nf7, Nf8, Nf9]

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
print('Creating output file: FinalData_ppttbar.hdf5')
file_out = h5py.File('FinalData_ppttbar.hdf5', 'w')   #write only mode for output
 
#-- Write data to output file --#
print('Writing data to output file')
zgrp = file_out.create_group("FDL")
xgrp = file_out.create_group("ROL") 

#-- Create datasets --#
# Note in this case all xData events have the same dimension (much easier to add to dataset)
# This won't be the case in general pp>ttbar, must do a trick
zdset = zgrp.create_dataset("zData", data=zData)
xdset = xgrp.create_dataset("xData", data=xData)

#-- Close file --#
print('Closing file!')
file_out.close()
