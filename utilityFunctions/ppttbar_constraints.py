import torch
import numpy as np

def pt_eta(px,py,pz, eps=1e-7):
    # --------------------------------------------------------------------------------------
    # Calculates transverse momentum and pseudorapidity 
    #
    # Inputs:  px, py, pz = Particle's 3-momenta 
    #          eps        = Small number to avoid troubles in this calculation
    #          
    # Outputs: transverse momentum (pt) and pseudo rapidity 
    # --------------------------------------------------------------------------------------
    if isinstance(px, torch.Tensor):
        backend = torch
    else:
        backend = np
    
    # Calculates pt and pseudorapidity eta
    pt  = (px**2+py**2) ** 0.5
    mag = (px**2+py**2+pz**2) ** 0.5
    costheta = pz/mag
    # eta = -0.5* backend.log( (1.0-costheta)/(1.0+costheta) )
    if backend is np:
        costheta = backend.clip(costheta, -1+eps, 1-eps)  # To avoid nan with atanh; numpy starts giving nan when eps <= 1e-17
        eta = backend.arctanh(costheta)
    elif backend is torch:
        # eta = backend.atanh(costheta)  # Need pytorch 1.5
        # eps = 1e-5
        costheta = backend.clamp(costheta, -1+eps, 1-eps)  # To avoid nan with atanh; torch starts giving inf when eps <= 1e-8
        eta = backend.log1p(2*costheta/(1-costheta)) / 2   # More stable: https://github.com/pytorch/pytorch/issues/10324

    return pt, eta


def threshold_check(X):
    # --------------------------------------------------------------------------------------
    # Checks the pt threshold constraint
    #
    # Inputs:  X = Matrix of raw x-space data
    #
    # Outputs: Boolean array of length len(X) indicating if each observation in X passes detector thresholds
    # --------------------------------------------------------------------------------------
    
    if isinstance(X, torch.Tensor):
        backend = torch
    else:
        backend = np
    if backend is torch:
        good_mask = torch.tensor([True] * len(X), device=X.device)
    else:
        good_mask = np.array([True] * len(X))

    # Lepton cuts
    min_lep_pt = 10
    max_lep_eta = 2.5

    # Jet cuts
    min_jet_pt = 20
    max_jet_eta = 4.5

    # Index of start of lepton,met, jet information
    lepton_i = 0
    met_i = 4
    jets_i = 8

    # Order of data
    ipx = 0
    ipy = 1
    ipz = 2
    iE = 3

    # For lepton
    x_idx, y_idx, z_idx = lepton_i + ipx, lepton_i + ipy, lepton_i + ipz
    px, py, pz = X[:, x_idx], X[:, y_idx], X[:, z_idx]
    min_pt, max_eta = min_lep_pt, max_lep_eta
    pt, eta = pt_eta(px, py, pz)
    good_mask[pt < min_pt] = False
    good_mask[backend.abs(eta) > max_eta] = False

    # No constraints on MET

    # For jets
    for i in range(4):
        jet_i = jets_i + i * 4
        x_idx, y_idx, z_idx = jet_i + ipx, jet_i + ipy, jet_i + ipz
        px, py, pz = X[:, x_idx], X[:, y_idx], X[:, z_idx]
        min_pt, max_eta = min_jet_pt, max_jet_eta
        pt, eta = pt_eta(px, py, pz)
        good_mask[pt < min_pt] = False
        good_mask[backend.abs(eta) > max_eta] = False

    return good_mask


def threshold_loss(X):
    # --------------------------------------------------------------------------------------
    # Compute the amount of threshold violation
    #
    # Inputs:  X = Matrix of raw x-space data
    #
    # Outputs: Boolean array of length len(X) indicating if each observation in X passes detector thresholds
    # --------------------------------------------------------------------------------------
    
    if isinstance(X, torch.Tensor):
        backend = torch
    else:
        backend = np
    # Lepton cuts
    min_lep_pt = 10
    max_lep_eta = 2.5

    # Jet cuts
    min_jet_pt = 20
    max_jet_eta = 4.5

    # Index of start of lepton, met, jet information
    lepton_i = 0
    met_i    = 4
    jets_i   = 8

    # Order of data
    ipx = 0
    ipy = 1
    ipz = 2
    iE  = 3

    lepton_px, lepton_py, lepton_pz = X[:, lepton_i + ipx], X[:, lepton_i + ipy], X[:, lepton_i + ipz]
    lepton_pt, lepton_eta = pt_eta(lepton_px,lepton_py, lepton_pz)
    loss = 0
    penalty_fun = backend.relu
    # penalty_fun = lambda x: backend.relu(x)**2
    loss += penalty_fun(min_lep_pt - lepton_pt)
    loss += penalty_fun(backend.abs(lepton_eta) - max_lep_eta)

    for i in range(4):
        jet_i = jets_i + i*4
        jet_px,jet_py,jet_pz = X[:, jet_i + ipx], X[:, jet_i + ipy], X[:, jet_i + ipz]
        jet_pt, jet_eta = pt_eta(jet_px,jet_py, jet_pz)

        loss += penalty_fun(min_jet_pt - jet_pt)
        loss += penalty_fun(backend.abs(jet_eta) - max_jet_eta)

    return loss.mean()


