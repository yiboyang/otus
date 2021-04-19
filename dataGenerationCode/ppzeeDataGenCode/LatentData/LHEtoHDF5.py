import numpy as np
import h5py

############################################################
##
## The following code is modified by Jessica N. Howard from: 
## https://github.com/rebecca-riley/muon_collider/blob/master/cuts.py
##
## -- Note on data structure: --
##   In the output file there are two groups each with its own dataset
## 
##   global_event/global_event_data
##      event # (added), nparticles, idprup, event_weight, scale, alpha_em, alpha_qcd
##      shape (nBlocks,1+6)
##      dType floats (store everything as floats, convert appropriate things to ints when in use)
##
##   particle_event/particle_event_data
##      event # (added), row # (added), PDG_code, status, parent1, parent2, color1, color2, px, py, pz, E, mass, spin1, spin2 
##      shape (nBlocks,nRows*(2+13)), note that nRows varies event block to event block
##      dType floats (store everything as floats, convert appropriate things to ints before use)
##
## -- Resources on LHE file structure: --
##   https://arxiv.org/pdf/hep-ph/0609017.pdf
##   https://www.physics.uci.edu/~tanedo/files/notes/ColliderMadgraph.pdf
##
## -- Resource on HDF5 file structure: --
#    https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
##
## -- Planned additions/modifications, coming soon: --
##   * Include rest of meta event data within LHE file
##   * Retrieve function to auto convert int-like objects that were stored as 
##     floats back to ints (i.e. a GetEventNumber function)
##   * Turn into a nicer class structure
##   * Faster data assignment method for variable length particle data
##
## -- List of functions: --
##   def processEvents(event_file):
##      Takes in LHE file                                  
##      Returns event blocks from LHE file in a list of strings
##   def trimEventData(event):
##      Takes in an event block                                  
##      Returns trimmed event block  
##   def extractEventData(event_block, ievent):
##      Takes in a trimmed event block and which event it belongs to                                 
##      Returns global and particle information in event block
##   def main(filename):
##      Takes in LHE file path                             
##      Returns HDF5 file with information contained
##
############################################################

##----------------------------------------------------------
## Process events
## Takes in LHE file                                  
## Returns event blocks from LHE file in a list of strings 
##----------------------------------------------------------
def processEvents(event_file):
    in_event = False                    #flag to indicate whether in event block
    event = ''                          #string to store event block data
    event_list = []
    #contnu = True                      #DEBUG -- ALLOWS PROGRAM TO RUN OVER ONE EVENT ONLY

    #-- Line processing loop --#
    for line in event_file:
        #if contnu == True:             #DEBUG -- AFTER ONE EVENT, STOP EXECUTION
            if line == '<event>\n':     #search file for start of event block
                in_event = True
            if in_event == True:        #if in event, collect info in event string
                event += line
            if line == '</event>\n':    #once event ends, process the data in the event,
                in_event = False        #reset the in_event flag, and clear storage str
                event_list.append(event)
                event = ''
                #contnu = False         #DEBUG -- AFTER ONE EVENT, STOP EXECUTION
    #print('Raw Event List', event_list) #DEBUG -- PRINT OUT EVENT_LIST
    return event_list

##----------------------------------------------------------
## Trim events
## Takes in an event block                                  
## Returns trimmed event block  
##----------------------------------------------------------  
def trimEventData(event):
    event_block = event.splitlines()  #store event line by line
    i = 1                             #data[0] = <event>; data[1] = event info
    while(i<len(event_block)):        #search for line starting in '<' (end of event block)
        if event_block[i][0] == '<':
            break                     
        else:
            i += 1
    
    #print('Trimmed Event List', event_block[1:i]) #DEBUG -- PRINT OUT TRIMMED EVENT BLOCK
    return event_block[1:i]           #throw away irrelevant lines at beginning/end

##----------------------------------------------------------
## Extract events
## Takes in a trimmed event block and which event it belongs to                                 
## Returns global and particle information in event block
##
## An event block has n+1 lines. The first line is global information 
## about the event. The next n lines are about the particles in the event.
##----------------------------------------------------------   
def extractEventData(event_block, ievent):
    particle_extract = []
    global_extract = []
    k=0
  
    # Loop over each row in event block
    for row in event_block:                       
        srow = row.split()
    
        # If first line in event block, store as global
        if k == 0:
            global_extract.append(ievent)             #keep track of which event this is
      
            for i in range(len(srow)):
                global_extract.append(float(srow[i]))   #conversion to float
    
        # Otherwise store as particle
        else:
            particle_extract.append(ievent)           #keep track of which event this is
            particle_extract.append(k)                #keep track of which row in the event this is
      
            for i in range(len(srow)):
                particle_extract.append(float(srow[i])) #conversion to float
  
        k+=1
  
    # Shape of global particle information (1, 1+6)
    # Shape of particle information (1, number_of_particles*(2+13))
    return global_extract, particle_extract           

##----------------------------------------------------------
## Main
## Takes in LHE file path                             
## Returns HDF5 file with information contained
##----------------------------------------------------------   
def main(filename):
    #-- Attempt to open file produce error if it doesn't work --#
    try:                                
        event_file = open(filename,'r')
    except IOError:                              #give error message, exit if file not found
        print(filename + ' not found. Try again with corrected input.')
        return
  
    #-- Create output file with structure --#
    print('Creating output file: file_out.hdf5')
    file_out = h5py.File('file_out.hdf5', 'w')   #write only mode for output
 
    #-- Process LHE file --#
    print('Processing LHE file')
    gdata = []                                   # list to store global data for all events
    pdata = []                                   # list to store particle data for all events
  
    # Extract all event blocks from input file as list of lists
    event_list = processEvents(event_file) 
  
    # Loop over all event blocks in event_list
    k=0
    for event_block in event_list:
    
        #if k<2:                             #DEBUG -- Only run for 2 events
            # Trim this block
            trimmed_event_block = trimEventData(event_block)  
    
            # Extract the data
            global_extracted_data, particle_extracted_data = extractEventData(trimmed_event_block, k)
  
            #DEBUG -- print the above to confirm shape
            #print("Global: ", global_extracted_data)
            #print("Particle: ", particle_extracted_data)

            # Add data to running list
            gdata.append(global_extracted_data)
            pdata.append(particle_extracted_data)
    
            # Increment iterator
            k+=1

    #-- Write data to output file --#
    print('Writing event information to output file')
    ggrp = file_out.create_group("global_event")
    pgrp = file_out.create_group("particle_event") 
  
    #DEBUG -- PRINT OUT GLOBAL AND PARTICLE DATA
    #print('Global Data') 
    #print(np.array(gdata[0]))
    #print(np.array(gdata).shape)
    
    #print('Particle Data')
    #print(np.array(pdata[0]))   
    #print(np.array(pdata).shape)

    # Create dataset, for pdata have to do things differently because nParticles in event block may vary
    gdset = ggrp.create_dataset("global_event_data", data=gdata)     
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    pdset = pgrp.create_dataset("particle_event_data", (len(pdata),), dtype=dt)
    
    for i in range(len(pdata)):
        pdset[i] = pdata[i]                 #more elegant way?
    
    file_out.close()
  
    print('Program completed successfully!')
