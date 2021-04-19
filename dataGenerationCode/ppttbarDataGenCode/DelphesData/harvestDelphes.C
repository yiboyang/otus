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

void harvestDelphes(const char *inputFile)
{

  /******************************************/
  /** Create csv files to store image data **/
  /******************************************/

  std::ofstream fout;
  fout.open("out_file.csv");
  fout << std::fixed << setprecision(8);

  std::ofstream fout1;
  fout1.open("numberperevt_out_file.csv");

  /*********************************/
  /** Delphes preliminary actions **/
  /*********************************/

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
  TClonesArray *branchElectron = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon = treeReader->UseBranch("Muon");
  TClonesArray *branchPhoton = treeReader->UseBranch("Photon");
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchMET = treeReader->UseBranch("MissingET");

  // Define object classes used in this analysis
  Electron *electron;
  Muon *muon;
  Photon *photon;
  Jet *jet;
  MissingET *met;

  // Define object (abs val) PIDs
  Int_t pidEle = 11;
  Int_t pidMu = 13;
  Int_t pidPh = 22;
  Int_t pidJet = 88; //Made up
  Int_t pidMET = 99; //Made up


  /***********************************************************/
  /** Define all other objects to be used in the event loop **/
  /***********************************************************/
  Int_t eleNu = -999;
  Int_t nEle = 0;
  Int_t muNu = -999;
  Int_t nMu = 0;
  Int_t phNu = -999;
  Int_t nPh = 0;
  Int_t jetNu = -999;
  Int_t nJet = 0;
  Int_t metNu = -999;
  Int_t nMET = 0;

  Int_t nTestE = 0;
  Int_t nTestP = 0;

  /**************************/
  /** Loop over all events **/
  /**************************/
  for(Int_t entry = 0; entry < numberOfEntries; ++entry)
    {

      treeReader->ReadEntry(entry);
      //std::cout << "Event#: " << entry << std::endl;

      //Get number of particles per event
      eleNu = branchElectron->GetEntriesFast();
      muNu = branchMuon->GetEntriesFast();
      phNu = branchPhoton->GetEntriesFast();
      jetNu = branchJet->GetEntriesFast();
      metNu = branchMET->GetEntriesFast();

      /*----------------*/
      /*-- Apply cuts --*/
      /*----------------*/

      //Skip any event that has eleNu != 1
      if(eleNu != 1)
	{
	  continue;
	}

      //Skip any event with non-zero muNu, or phNu
      if(muNu != 0 || phNu != 0)
	{
	  continue;
	}

      //Skip any event that has jetNu != 4
      if(jetNu != 4)
        {
          continue;
        }

      //Make sure the event has one electron (as opposed to a positron) 
      nTestE = 0;
      for(Int_t i = 0; i < eleNu; ++i)
        {
          electron = (Electron*) branchElectron->At(i);

	  if(electron->Charge<0){
	    nTestE++;
	  }

	}
      
      if(nTestE != 1)
	{
	  continue;
	}


      // Write information to csv file
      //evt#,e,mu,photon,jet,met
      fout1<<entry<<","<<eleNu<<","<<muNu<<","<<phNu<<","<<jetNu<<","<<metNu<<"\n";

      /*-------------------*/
      /*-- Electron loop --*/
      /*-------------------*/
      //std::cout << "  e#: " << eleNu << std::endl;
      for(Int_t i = 0; i < eleNu; ++i)
	{
	  electron = (Electron*) branchElectron->At(i);
	  
	  // Write information to csv file
	  // evt#,PID,PT,Eta,Phi
	  if(electron->Charge<0){
	    fout<<entry<<","<<pidEle<<","<<electron->PT<<","<<electron->Eta<<","<<electron->Phi<<"\n";
	  }
	  else{
	    fout<<entry<<","<<-pidEle<<","<<electron->PT<<","<<electron->Eta<<","<<electron->Phi<<"\n";
	  }
	  
	  nEle++;
	}

      /*-------------------*/
      /*-- Muon loop --*/
      /*-------------------*/
      for(Int_t i = 0; i < muNu; ++i)
        {
          muon = (Muon*) branchMuon->At(i);

          // Write information to csv file
          // evt#,PID,PT,Eta,Phi          
          if(muon->Charge<0){
            fout<<entry<<","<<pidMu<<","<<muon->PT<<","<<muon->Eta<<","<<muon->Phi<<"\n";
          }
          else{
            fout<<entry<<","<<-pidMu<<","<<muon->PT<<","<<muon->Eta<<","<<muon->Phi<<"\n";
          }

          nMu++;
        }

      /*-----------------*/
      /*-- Photon loop --*/
      /*-----------------*/
      for(Int_t i = 0; i < phNu; ++i)
        {
          photon = (Photon*) branchPhoton->At(i);

          // Write information to csv file
          // evt#,PID,PT,Eta,Phi       
	  fout<<entry<<","<<pidPh<<","<<photon->PT<<","<<photon->Eta<<","<<photon->Phi<<"\n";
	  nPh++;
        }

      /*--------------*/
      /*-- Jet loop --*/
      /*--------------*/
      for(Int_t i = 0; i < jetNu; ++i)
	{
          jet = (Jet*) branchJet->At(i);

          // Write information to csv file
          // evt#,PID,PT,Eta,Phi          
          fout<<entry<<","<<pidJet<<","<<jet->PT<<","<<jet->Eta<<","<<jet->Phi<<"\n";
          nJet++;
        }

      /*--------------*/
      /*-- MET loop --*/
      /*--------------*/
      for(Int_t i = 0; i < metNu; ++i)
	{
          met = (MissingET*) branchMET->At(i);

          // Write information to csv file
          // evt#,PID,PT,Eta,Phi          
          fout<<entry<<","<<pidMET<<","<<met->MET<<","<<met->Eta<<","<<met->Phi<<"\n";
          nMET++;
        }


    }

  fout.close();
  fout1.close();

  cout<<"Number of electron lines in csv file = "<<nEle<<endl;
  cout<<"Number of muon lines in csv file     = "<<nMu<<endl;
  cout<<"Number of photon lines in csv file   = "<<nPh<<endl;
  cout<<"Number of jet lines in csv file      = "<<nJet<<endl;
  cout<<"Number of MET lines in csv file      = "<<nMET<<endl;
}
