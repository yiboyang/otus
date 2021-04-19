#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#endif

/************************************************************************************/
/** This code counts the number of 4-jet events in the resulting ttbar data.       **/
/** It also plots a histogram of the number of jets in the events.                 **/
/**                                                                                **/
/** To run:                                                                        **/
/** $source init.sh                                                                **/
/** $root -l countNumJets.C'("PATHtoROOTfile")'                                    **/
/**                                                                                **/
/** For example:                                                                   **/
/** $root -l countNumJets.C'("ppttbar_2/Events/run_05/tag_1_delphes_events.root")' **/
/************************************************************************************/ 

void countNumJets(const char *inputFile)
{
  // Load Delphes functions
  gSystem->Load("libDelphes");

  // Create chain of root trees from files
  TChain chain("Delphes");
  chain.Add(inputFile);

  // Create object of class ExRootTreeReader
  ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);

  // Get number of entries in tree of whole file chain (number of total events in all files combined) 
  Long64_t numberOfEntries = treeReader->GetEntries();

  // Get pointers to branches used in this analysis
  TClonesArray *branchJet = treeReader->UseBranch("Jet");

  // Define object classes used in this analysis
  Jet *jet;

  Int_t num4JetEvts = 0;
  Int_t jetNu = 0;  

  // Histogram of jetNu
  TH1F *h = new TH1F("h", "Number of Jets per Event", 15, 0.5, 15.5);

  /**************************/
  /** Loop over all events **/
  /**************************/
  for(Int_t entry = 0; entry < numberOfEntries; ++entry)
    {

      /** Read entry **/
      treeReader->ReadEntry(entry);
      //std::cout << "Event#: " << entry << std::endl;

      /** Get number of jets per event **/
      jetNu = branchJet->GetEntriesFast();

      /** Fill histogram of number of jets **/
      h->Fill(jetNu);

      /** Count number of 4-jet events **/
      if(jetNu==4){
	num4JetEvts ++;
      }

    }

  /** Draw histogram **/
  h->Draw();

  /** Print number of 4-jet events **/
  cout<<"Number of 4 jet events: "<<num4JetEvts<<" out of "<<numberOfEntries<<" total events"<<endl;
}
