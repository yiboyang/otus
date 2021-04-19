import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#----------------------------------------#
#-- Training Related Utility Functions --#
#----------------------------------------#

def generateTheta(L, ldim):
    # --------------------------------------------------------------------------------------
    # Generate random slices of the unit sphere in R^ldim.
    #
    # Inputs:  L    = number of slices
    #          ldim = dimension of the space
    #
    # Outputs: L random slices of the unit sphere in R^ldim, numpy array with shape (L,ldim)
    #
    # Original from: https://github.com/skolouri/swae
    # --------------------------------------------------------------------------------------

    # Original version:
    # theta = [w / np.sqrt((w ** 2).sum()) for w in np.random.normal(size=(L, ldim))]
    # return np.asarray(theta)  # Shape (L,ldim)

    theta = np.random.normal(size=(L, ldim))
    return theta / np.sqrt(np.sum(theta ** 2, axis=1))[:, np.newaxis]  # Shape (L,ldim)


def sliced_wd(X: torch.Tensor, Y: torch.Tensor, nSlices: int, p=2):
    # --------------------------------------------------------------------------------------
    # Compute (approximately) the sliced Wasserstein distance between two distributions p_x
    # and p_y, using samples X and Y from the two distributions.
    #
    # Inputs:  X       = Sample from distribution 1, torch Tensor of size batch_size x ldim
    #          Y       = Sample from distribution 2, torch Tensor of size batch_size x ldim
    #          nSlices = Number of projections (slices) for calculating the Sliced
    #                    Wasserstein distance (SWD)
    #          p       = L_p norm used in defining Sliced-p-Wasserstein distance; default p=2
    #
    # Outputs: Average Sliced-p-Wasserstein distance 
    #
    # Original from: https://github.com/skolouri/swae
    # --------------------------------------------------------------------------------------
    batch_size = X.size(0)

    # Make sure same batchsize
    assert (batch_size == Y.size(0))

    # dimensionality of the distribution
    ldim = X.size(1)

    # -- Generate random slices --#
    # shape is (nSlices, ldim)
    theta = generateTheta(nSlices, ldim)  # np array
    thetaT = torch.from_numpy(theta.transpose())  # shape (ldim, nSlices)
    thetaT = thetaT.to(dtype=X.dtype, device=X.device)

    # -- Project X onto theta, X dot thetaT, resulting shape = (batch_size,nSlices) --#
    projX = torch.mm(X, thetaT)  # (batch_size,ldim) dot (ldim,nSlices)

    # -- Project Y onto theta --#
    projY = torch.mm(Y, thetaT)

    # -- Calculate Sliced Wasserstein Distance --#
    # Sort projX and projY in ascending order
    # Meaning sort each column such that, for a given column,
    # the values in each row get larger or stay the same as
    # the row number increases
    projX_sort, _ = torch.sort(projX, dim=0)
    projY_sort, _ = torch.sort(projY, dim=0)

    # Compute || ||_p^p distance between projX_sort and projY_sort
    assert p in (1, 2), 'Not supported!'
    diff = projY_sort - projX_sort
    if p == 1:
        W_p = diff.abs()
    else:
        W_p = diff ** 2  # shape (batch_size, nSlices)

    # Average over nSlices and batch_size which reduces to a single number
    return torch.mean(W_p)


def latent_loss(Z: torch.Tensor, Ztilde: torch.Tensor, nSlices: int, p=2):
    # --------------------------------------------------------------------------------------
    # Calculates the latent loss (SWD) for training an SWAE model
    #
    # Inputs:  Z       = Sample from the true prior distribution, torch Tensor of size 
    #                    batch_size x ldim
    #          Ztilde  = Sample from the learned distribution, torch Tensor of size 
    #                    batch_size x ldim
    #          nSlices = Number of projections (slices) for calculating the Sliced
    #                    Wasserstein distance (SWD)
    #          p       = L_p norm used in defining Sliced-p-Wasserstein distance; default p=2
    #
    # Outputs: Average Sliced-p-Wasserstein distance
    # --------------------------------------------------------------------------------------
    
    return sliced_wd(Z, Ztilde, nSlices, p)


def cosine_sim(Z, X):
    # --------------------------------------------------------------------------------------
    # Calculates the key part of the regularization loss to encourage physicality of the
    # learned mappings. Given two batches of vectors, Z, X, of shape (batch_size x D), compute
    # the cosine similarity between each corresponding pairs of vectors in the batch.
    #
    # Inputs:  Z = Sample that is fed into the network
    #          X = Sample that is subsequently produced by the network
    #
    # Outputs: Loss vector of length batch_size
    # --------------------------------------------------------------------------------------

    # batch_size = len(X)
    # assert batch_size == len(Z)
    if isinstance(Z, np.ndarray) and isinstance(X, np.ndarray):
        Zhat = Z / np.linalg.norm(Z, axis=-1, keepdims=True)
        Xhat = X / np.linalg.norm(X, axis=-1, keepdims=True)
        # dot product of Zhat and Xhat
        return (Xhat * Zhat).sum(axis=-1)

    # Turn these into unit vectors
    Znorm = Z.norm(p=2, dim=1, keepdim=True)
    Zhat = Z.div(Znorm.expand_as(Z))

    Xnorm = X.norm(p=2, dim=1, keepdim=True)
    Xhat = X.div(Xnorm.expand_as(X))

    # Dot product of Zhat and Xhat
    return (Xhat * Zhat).sum(dim=-1)


def cosine_dist(Z, X):
    # --------------------------------------------------------------------------------------
    # Calculates full regularization loss to encourage physicality of the learned mappings.
    #
    # Inputs:  Z = Sample that is fed into the network
    #          X = Sample that is subsequently produced by the network
    #
    # Outputs: Loss vector of length batch_size
    # -------------------------------------------------------------------------------------- 
    return 1 - cosine_sim(Z, X)


def anchor_loss(Z: torch.Tensor, X: torch.Tensor, anchor_coords=tuple(range(3))):
    # --------------------------------------------------------------------------------------
    # Computes average of full regularization loss to encourage physicality of the learned
    # mappings.
    #
    # Inputs:  Z             = Sample that is fed into the network
    #          X             = Sample that is subsequently produced by the network
    #          anchor_coords = A tuple/list of integers corresponding to coordinates of X and Z
    #                          that should be "anchored";  default is (0, 1, 2), i.e. the first
    #                          3 coordinates, corresponding to momentum elements of the first
    #                          particle (electron).
    #
    # Outputs: Average loss
    # --------------------------------------------------------------------------------------
    
    batch_size = Z.size(0)

    # Make sure Z and X have same batchsize
    assert (batch_size == X.size(0))

    # Select first 3 elements of both Z and X
    Zsel = Z[:, anchor_coords]
    Xsel = X[:, anchor_coords]

    return cosine_dist(Zsel, Xsel).mean()  # avg across batch


def data_loss(X: torch.Tensor, Xtilde: torch.Tensor, p=2):
    # --------------------------------------------------------------------------------------
    # Calculates the data loss (MSE) for training an SWAE model 
    # 
    # Inputs:  X       = Sample from the true data distribution, torch Tensor of size 
    #                    batch_size x ldim
    #          Xtilde  = Sample from the learned distribution, torch Tensor of size 
    #                    batch_size x ldim
    #          p       = L_p norm used in defining Sliced-p-Wasserstein distance; default p=2
    #
    # Outputs: Average loss
    # 
    # Original from: https://github.com/skolouri/swae
    # --------------------------------------------------------------------------------------
    if p == 1:
        dist = (X - Xtilde).abs()
    elif p == 2:
        dist = (X - Xtilde) ** 2
    else:
        raise NotImplementedError
    return torch.mean(dist)


from configs import data_dir, float_type

def get_data_loaders(dataset_name, data_dir=data_dir, batch_size=100, train_data_ratio=0.6, dtype=float_type):
    # --------------------------------------------------------------------------------------
    # Sets up DataLoaders for both training and evaluation
    #
    # Inputs:  dataset_name     = Name of .hdf5 data
    #          data_dir         = Path to .hdf5 dataset, default is data_dir set in configs.py
    #          batch_size       = Number of samples in each batch, default is 100
    #          train_data_ratio = Faction of data to use for training, default is 0.6
    #          dtype            = Data type, default is float_type set in configs.py
    #
    # Outputs: Training and Evaluation loaders
    # --------------------------------------------------------------------------------------
    
    dataset = get_dataset(dataset_name, data_dir)
    z_data, x_data = dataset['z_data'], dataset['x_data']

    x_train_size = int(len(x_data) * train_data_ratio)
    x_train = x_data[0:x_train_size, :]
    x_eval = x_data[x_train_size:, :]

    z_train_size = int(len(z_data) * train_data_ratio)
    z_train = z_data[0:z_train_size, :]
    z_eval = z_data[z_train_size:, :]

    x_train, x_eval, z_train, z_eval = list(map(lambda x: x.astype(dtype), [x_train, x_eval, z_train, z_eval]))
    x_train, x_eval, z_train, z_eval = list(map(standardize, [x_train, x_eval, z_train, z_eval]))
    
    from torch.utils.data import DataLoader
    train_loaders = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True), \
                    DataLoader(dataset=z_train, batch_size=batch_size, shuffle=True)

    eval_loaders = DataLoader(dataset=x_eval, batch_size=batch_size, shuffle=True), \
                   DataLoader(dataset=z_eval, batch_size=batch_size, shuffle=True)

    return train_loaders, eval_loaders




#-------------------------------#
#-- General Utility Functions --#
#-------------------------------#

def get_dataset(dataset_name, data_dir=data_dir):
    # --------------------------------------------------------------------------------------
    # Loads data from .hdf5 file on disk.
    # 
    # Inputs:  dataset_name = Name of .hdf5 data
    #          data_dir     = Path to .hdf5 dataset, default is data_dir set in configs.py
    #
    # Outputs: Returns dictionary of zData and xData arrays
    # --------------------------------------------------------------------------------------
    import os
    data_path = os.path.join(data_dir, dataset_name + '.hdf5')
    import h5py
    fdata = h5py.File(data_path, 'r')

    # -- Get the keys --#
    # zkey = list(fdata.keys())[0]
    # xkey = list(fdata.keys())[1]
    zkey = 'FDL' # "Feynman Diagram Level" historical name
    xkey = 'ROL' # "Reconstructed Object Level" historical name

    # -- Get the data --#
    zgp = fdata.get(zkey)
    if isinstance(zgp, h5py.Dataset):  # For hdf5 files created by me
        zData = zgp
    else:
        zData = zgp.get('zData')

    xgp = fdata.get(xkey)
    if isinstance(xgp, h5py.Dataset):  # For hdf5 files created by me
        xData = xgp
    else:
        xData = xgp.get('xData')

    return dict(z_data=zData, x_data=xData)


def standardize(data_arr):
    # --------------------------------------------------------------------------------------
    # Standardize data by subtracting the mean and dividing by the standard-deviation
    #
    # Inputs:  data_arr = Array of data to be standardized
    #
    # Outputs: Standardized data_arr
    # --------------------------------------------------------------------------------------
    mu = np.mean(data_arr, axis=0)
    sig = np.std(data_arr, axis=0)
    data_arr = (data_arr - mu) / sig
    return data_arr





#--------------------------------------------------------#
#-- General and Statistical Analysis Utility Functions --#
#--------------------------------------------------------#

def calcInvM(v):
    # --------------------------------------------------------------------------------------
    # Calculate invariant mass according to the Minkowski metric.
    #
    # Inputs:  v = Array of 4-vectors of particles, shape N x 4
    # 
    # Outputs: Invariant mass of each particle
    # --------------------------------------------------------------------------------------
    M2 = v[:,3]**2 - v[:,2]**2 - v[:,1]**2 - v[:,0]**2
    
    # Replace all negative entries with zero
    M2_fixed = np.where(M2<0., 0., M2)

    return np.sqrt(M2_fixed)

def Zboson_mass(Y):
    # --------------------------------------------------------------------------------------
    # Calculate Z boson invariant mass according to the Minkowski metric and 4-momenta conservation
    #
    # Inputs:  Y = Array of pairs of 4-vectors of particles which should reconstruct to the Z bosons 
    #              mass (i.e. electron, positron); shape is N, 8 such that the first 4 columns 
    #              correspond to the electron and the last 4 to the positron
    #
    # Outputs: Invariant mass of each particle
    # --------------------------------------------------------------------------------------
    return calcInvM(Y[:, 0:4] + Y[:, 4:8])


def wd_1d(x, y, p=2):
    # --------------------------------------------------------------------------------------
    # Compute 1D Wasserstein distance from two 1D empirical distributions, given two vectors  
    # of samples from each. Torch Version.
    #
    # Inputs:  x = Sample from distribution 1
    #          y = Sample from distribution 2
    #          p = L_p norm used in calculating the 1D p-Wasserstein distance; default p=2
    #
    # Outputs: 1D Wasserstein distance between distribution 1 and distribution 2
    # --------------------------------------------------------------------------------------

    # Check if inputs are numpy arrays, if so convert to torch tensors
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
        
    common_len = min(len(x), len(y))
    x = x[:common_len]
    y = y[:common_len]
    x, _ = torch.sort(x, dim=0)
    y, _ = torch.sort(y, dim=0)
    if p == 1:
        cost = (x - y).abs()
    elif p == 2:
        cost = (x - y) ** 2
    else:
        raise NotImplementedError
    return torch.mean(cost)

def wd_1d_numpy(X, Y, p=2):
    # --------------------------------------------------------------------------------------
    # Compute 1D Wasserstein distance from two 1D empirical distributions, given two vectors  
    # of samples from each. Numpy Version.
    #
    # Inputs:  x = Sample from distribution 1
    #          y = Sample from distribution 2
    #          p = L_p norm used in calculating the 1D p-Wasserstein distance; default p=2
    #
    # Outputs: 1D Wasserstein distance between distribution 1 and distribution 2
    # --------------------------------------------------------------------------------------

    # Make sure Z and Ztilde have same batchsize                                                              
    assert(X.shape[0]==Y.shape[0])
    batch_size = X.shape[0]

    #-- Calculate 1D Wasserstein Distance --#                                                                 
    # Sort Z and Ztilde in ascending order                                                                    
    # Meaning sort each column such that, for a given column,                                                 
    # the values in each row get larger or stay the same as                                                   
    # the row number increases                                                                                
    X_sort = np.sort(X, axis=0)
    Y_sort = np.sort(Y, axis=0)

    # Compute L_p cost between projZ_sort and projZtilde_sort 
    if p == 1:
        WP = np.abs(Y_sort - X_sort)
    elif p == 2:
        WP = (Y_sort - X_sort)**2 # Shape (batch_size, nSlices)
    else:
        raise NotImplementedError

    # Average over nSlices and batch_size which reduces to a single number                                    
    return np.mean(WP)

def ReducedChiSq(X, Y):
    # --------------------------------------------------------------------------------------
    # Compute Chi2 test statistic and pvalue 
    #
    # Inputs:  X = 1D numpy array of length N where the values are all > 5
    #          Y = 1D numpy array of length N where the values are all > 5
    #
    # Outputs: Reduced Chi2, degrees of freedom, pvalue
    #
    # Note that Chi2 is calculated assuming X has errors \sqrt(X) and Y has errors \sqrt(Y).
    # We also assume that X and Y are completely independently drawn.
    # --------------------------------------------------------------------------------------
 
    #-- Make sure X, Y are same shape and all have values greater than 5 --#                                  
    assert(X.shape == Y.shape)
    
    if(np.any(X<5)):
        print(X)
    if(np.any(Y<5)):
        print(Y)
    assert np.all(X>=5), 'X has at least one value which is less than 5'
    assert np.all(Y>=5), 'Y has at least one value which is less than 5'

    #-- Get DOF --#                                                                                           
    n = X.shape[0]
    dof = n - 1

    #-- Compute Chi2 --#                                                                                      
    C2 = np.sum((X-Y)**2/(X+Y))

    #-- Compute Reduced Chi2 --#                                                                              
    RC2 = C2/dof

    #-- Compute pvalue --#                                                                                    
    p = 1. - stats.chi2.cdf(C2, dof)

    return RC2, dof, p

def KSTest(X, Y):
    # --------------------------------------------------------------------------------------
    # Compute KS test statistic and pvalue
    # 
    # Inputs:  X = 1D numpy array of length N
    #          Y = 1D numpy array of length N
    #
    # Outputs: KS test statistic, pvalue
    #
    # Note we are doing a two-sample KS test and our test is two-sided since we are looking for 
    # surplus or deficit deviations: 
    # https://www.statisticssolutions.com/should-you-use-a-one-tailed-test-or-a-two-tailed-test-for-your-data-analysis/
    # --------------------------------------------------------------------------------------

    stat, p = stats.ks_2samp(X, Y) # Default is alternative='two-sided'                                                 

    return stat, p

def runStatAnalysis(X, Y, bins, p=2):
    # --------------------------------------------------------------------------------------
    # Run full statistical analysis comparing X and Y via 1D Wasserstein distance, KS test, Chi2 test
    #
    # Inputs:  X    = 1D numpy array of length N
    #          Y    = 1D numpy array of length N
    #          bins = Array of bins to use when making histograms of X and Y data; must be chosen 
    #                 to ensure each bin has >5 entries
    #          p       = L_p norm used in defining Sliced-p-Wasserstein distance; default p=2
    #
    # Outputs: 1D Wasserstein distance, KS test statistic, KS test pvalue, Reduced Chi2, degrees of freedom, pvalue
    # --------------------------------------------------------------------------------------
    
    #-- Calculate W1 distance --#                                                                             
    W1D = wd_1d_numpy(X, Y, p=p)

    #-- Calculate KS statistic --#                                                                            
    KSstat, KSpval = KSTest(X, Y)

    #-- Calculate chi2 statistic --#                                                                          
    n1, _, _ = plt.hist(X, bins, histtype='step', density=False)
    n2, _, _ = plt.hist(Y, bins, histtype='step', density=False)
    plt.close()

    RC2, dof, chi2pval = ReducedChiSq(n1, n2)

    return(W1D, KSstat, KSpval, RC2, dof, chi2pval)

def sliced_wd_numpy(Y, Ytilde, L, p=2):
    # --------------------------------------------------------------------------------------
    # Compute (approximately) the sliced Wasserstein distance between two distributions p_x
    # and p_y, using samples X and Y from the two distributions.
    #
    # Inputs:  X       = Sample from distribution 1, numpy array of size batch_size x ldim
    #          Y       = Sample from distribution 2, numpy array of size batch_size x ldim
    #          nSlices = Number of projections (slices) for calculating the Sliced
    #                    Wasserstein distance (SWD)
    #          p       = L_p norm used in defining Sliced-p-Wasserstein distance; default p=2
    #
    # Outputs: Average Sliced-p-Wasserstein distance 
    #
    # Original from: https://github.com/skolouri/swae
    # --------------------------------------------------------------------------------------

    batch_size = Y.shape[0]

    # Make sure Y and Ytilde have same batch_size                                                             
    assert(Y.shape[0] == Ytilde.shape[0])

    # Get other dimensions of Y data                                                                          
    ldim = Y.shape[1]

    #-- Generate random slices --#                                                                            
    # shape is (L, ldim)                                                                                      
    theta  = generateTheta(L,ldim)

    #-- Project Y onto theta, Y dot thetaT, resulting shape = (batch_size, L) --#                             
    projY = np.matmul(Y, theta.T) #(batch_size,ldim) dot (ldim,L)                                             

    #-- Project Ytilde onto theta --#                                                                         
    projYtilde = np.matmul(Ytilde, theta.T)

    #-- Calculate Sliced Wasserstein Distance --#                                                             
    return wd_1d_numpy(projY, projYtilde, p=p)

def sciNotationStringLaTeX(x):
    # --------------------------------------------------------------------------------------
    # Takes a number x and returns this number expressed in scientific notation, LaTeX form
    #
    # Inputs:  x = number
    #
    # Outputs: x as a LaTeX formatted string in scientific notation; precision is 3
    # --------------------------------------------------------------------------------------

    s = '%.2E'% x
    snew = r'$'+s[0:4]+r' \times 10^{'+s[-3:]+r'}$'

    return snew

def statTableSingle(statList12, Wunit="r'[GeV$^2$]'", figLabel="", cline=True):
    # --------------------------------------------------------------------------------------
    # Takes stat information for z-space data and formats it into the core of a LaTeX table 
    #
    # Inputs:  statList12 = List of stats information
    #          Wunit      = Unit on the Wasserstein distance
    #          figLabel   = Which figure does this stats information refer to
    #          cline      = Whether to use cline or hline
    # --------------------------------------------------------------------------------------
    
    # Print line in table                                                                                     
    string = r'\footnotesize{\textbf{Fig. '
    string += figLabel
    string += r'}} & \footnotesize{'
    string += sciNotationStringLaTeX(statList12[0]) # W distance 12                                           
    string += r'} & \footnotesize{('
    string += r'$%.3f$, $%d$'% (statList12[3], statList12[4])
    string += r')} & \footnotesize{'
    string += sciNotationStringLaTeX(statList12[1])
    if(cline):
        string += r'}\\\cline{2-4}'
    else:
        string += r'}\\\hline'

    print(string)

def statTableDouble(statList12, statList13, Wunit="r'[GeV$^2$]'", figLabel="", cline=True):
    # --------------------------------------------------------------------------------------
    # Takes stat information for x-space data and formats it into the core of a LaTeX table 
    #
    # Inputs:  statList12 = List of stats information between x and \tilde{x}
    #          statList13 = List of stats information between x and \tilde{x}'
    #          Wunit      = Unit on the Wasserstein distance
    #          figLabel   = Which figure does this stats information refer to
    #          cline      = Whether to use cline or hline
    # --------------------------------------------------------------------------------------
    
    # Print line in table                                                                                     
    string = r'\footnotesize{\textbf{Fig. '
    string += figLabel
    string += r'}} & \footnotesize{'
    string += sciNotationStringLaTeX(statList12[0]) # W distance 12                                           
    string += r'} & \footnotesize{('
    string += r'$%.3f$, $%d$'% (statList12[3], statList12[4]) #chi2R, dof 12                                  
    string += r')} & \footnotesize{'
    string += sciNotationStringLaTeX(statList12[1]) # KS 12                                                   

    string += r'} & \footnotesize{'
    string += sciNotationStringLaTeX(statList13[0]) # W distance 13
    string += r'} & \footnotesize{('
    string += r'$%.3f$, $%d$'% (statList13[3], statList13[4]) #chi2R, dof 13                                  
    string += r')} & \footnotesize{'
    string += sciNotationStringLaTeX(statList13[1]) # KS 13

    if(cline):
        string += r'}\\\cline{2-7}'
    else:
        string += r'}\\\hline'

    print(string)
