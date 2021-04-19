from LorentzVector import LorentzVector
from math import sqrt

# Constants in GeV
massW = 80.4
masstop = 172.5


def ttbar_masses(x):
    # --------------------------------------------------------------------------------------
    # Calculate the W and top masses for event x
    #
    # Inputs:  x =  24-length vector, px,py,pz,E for e, met, jet1, jet2, jet3, jet4
    # 
    # Outputs: ttbar mass, leptonic W mass, hadronic W mass, leptonic top mass, hadronic top mass
    # --------------------------------------------------------------------------------------
    
    # Index of start of lepton, met, jet information
    lepton_i = 0
    met_i    = 4
    jets_i   = 8
    
    # Order of data
    ipx = 0
    ipy = 1
    ipz = 2
    iE  = 3

    lepton = LorentzVector()
    lepton.SetPxPyPzE(x[lepton_i + ipx],x[lepton_i + ipy],x[lepton_i + ipz],x[lepton_i + iE])

    met = LorentzVector()
    met.SetPxPyPzE(x[met_i + ipx],x[met_i + ipy],x[met_i + ipz],x[met_i + iE])

    jets = []
    for i in range(4):
        jet = LorentzVector()
        jet_i = jets_i + i*4
        jet.SetPxPyPzE(x[jet_i + ipx],x[jet_i + ipy],x[jet_i + ipz],x[jet_i + iE])
        jets.append(jet)


    lepW,hadW,leptop,hadtop = ttbar_reco(lepton,met,jets)
    
    return (leptop+hadtop).M(),lepW.M(),hadW.M(),leptop.M(),hadtop.M()


def ttbar_reco(lepton, met, jets):
    # --------------------------------------------------------------------------------------
    # Reconstruct the top and W four-vectors
    # 
    # Inputs:  lepton, met, jets = 4-vectors of lepton, met and 4 jets respectively
    #
    # Outputs: 4-vectors of the leptonic W, hadronic W, leptonic top, hadronic top respectively
    # --------------------------------------------------------------------------------------
    
    # Neutrino pz: solution of quadratic
    k = ((massW**2 - lepton.M()*lepton.M()) / 2) + (lepton.Px()*met.Px() + lepton.Py()*met.Py())
    
    a = lepton.E()*lepton.E() - lepton.Pz()*lepton.Pz()
    b = -2 * k * lepton.Pz()
    c = lepton.E()*lepton.E()*met.Pt()*met.Pt() - k*k

    disc = b*b - 4*a*c
    
    if disc < 0:
        nu_pz = -b/(2*a)
    else:
        nu_pz_1 = (-b + sqrt(disc)) / (2 * a)
        nu_pz_2 = (-b - sqrt(disc)) / (2 * a)

        if abs(nu_pz_1) < abs(nu_pz_2):
            nu_pz = nu_pz_1
        else:
            nu_pz = nu_pz_2

    neutrino = LorentzVector()
    neutrino.SetPxPyPzE(met.Px(), met.Py(), nu_pz, sqrt(met.Pt()*met.Pt() + nu_pz*nu_pz))

    lepW = neutrino+lepton

    # bjet assignment
    best_chi2 = 1e9
    best_bhad = -1
    best_blep = -1
    best_iw1 = -1
    best_iw2 = -1
    for bhad in range(4):
        for blep in range(4):
            if (bhad != blep):
                chi2=0
            
                #hadronic W
                iW1 = -1
                iW2 = -1
                for i in range(4):
                    if (i != blep and i != bhad):
                        if iW1 == -1:
                            iW1 = i
                        else:
                            iW2 = i

                hadW = jets[iW1] + jets[iW2]

                # hadronic top
                hadtop = hadW + jets[bhad]
                # leptonic top
                leptop = lepW + jets[blep]
                


                chi2 = pow( (hadW.M()-massW)/10,2) + pow( (leptop.M()-masstop)/5,2) + pow( (hadtop.M()-masstop)/15,2)

                if (chi2<best_chi2):
                    best_chi2 = chi2
                    best_bhad = bhad
                    best_blep = blep
                    best_iw1 = iW1
                    best_iw2 = iW2

    # Make best one
    hadW = jets[best_iw1] + jets[best_iw2]
 
    # hadronic top
    hadtop = hadW + jets[best_bhad]
    # leptonic top
    leptop = lepW + jets[best_blep]

    return lepW,hadW,leptop,hadtop

