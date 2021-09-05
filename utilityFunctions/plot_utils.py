import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from func_utils import wd_1d


def compare_hist2d(data_mats, coords, titles, bins=100, vmin=0, vmax=None, cmap='jet', xlim=None, ylim=None, **kwargs):
    # --------------------------------------------------------------------------------------
    # Compare 2D histograms of data from two distributions. Given two data
    # matrices, a 2D histogram is created from two columns (specified by
    # 'coords'), from each data matrix, and the two 2d histograms are plotted
    # side by side.
    # vmin and vmax are shared so that the coloring is consistent between the
    # two. Some trial and error is needed to set vmin, vmax, xlim, and ylim right.
    #
    # Inputs:  data_mats  = Data to use for plotting, list of length 2
    #          coords     = array of length 2 of column indices; two columns
    #          titles     = Titles for plots, list of length 2
    #          bins       = Number of bins, default 100
    #          vmin, vmax = Min and max of 2D histogram
    #          cmap       = Color map; default 'jet'
    #          xlim, ylim = x and y limits of the plot
    # Example call:
    # compare_hist2d([x_train, z_train], [8, 9], [r'$p(x)$', r'$p(z)$'], vmax=200)
    # --------------------------------------------------------------------------------------

    if coords is  None:
        coords = np.array([0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    
    # Ensure that data all has the same number of samples for plotting
    sizeList = [] 
    for i in range(len(data_mats)):
        sizeList.append(data_mats[i].shape[0])
    nmin = np.min(sizeList)
    for i in range(len(data_mats)):
        data_mats[i] = data_mats[i][0:nmin]
    
    for k, data in enumerate(data_mats):
        ax = axes[k]
        title = titles[k]
        var_letter = title.split('(')[1][0] # usually 'z' or 'x'
        hist2dres = ax.hist2d(data[:, coords[0]], data[:, coords[1]], bins=bins,
                       cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        print('x range:', data[:, coords[0]].min(), data[:, coords[0]].max())
        print('y range:', data[:, coords[1]].min(), data[:, coords[1]].max())
        ax.set_xlabel(r'$%s_{%d}$' %(var_letter, coords[0]))
        ax.set_ylabel(r'$%s_{%d}$' %(var_letter, coords[1]))
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
    plt.colorbar(hist2dres[-1], cax=cax)


def transportPlot(ZX, zlim=(-2.5,+2.5), xlim=(-2.5,+2.5), nz=5, nx=5, name='Transport.png', dec=True, title='', save_figs=False, img_dir='Images'):
    # -------------------------------------------------------------------------------------
    # Creates single transport plot of data
    #
    # Inputs:  ZX        = Z,X data pairs numpy array
    #          zlim      = Limits for z axis, note add +0.5 for binning reasons
    #          xlim      = Limits for x axis, note add +0.5 for binning reasons
    #          nz        = Number of bins along z axis
    #          nx        = Number of bins along x axis
    #          dec       = If dec=True then plot results for decoder, if dec=False then plot results for encoder
    #          title     = Title for plot
    #          save_figs = Whether to save image of plot
    #          img_dir   = Image directory to use
    #
    # Outputs: Creates transport plot; saves image if flag is set
    #
    # Code modified from https://github.com/dkirkby/MachineLearningStatistics/blob/master/notebooks/Probability.ipynb
    # -------------------------------------------------------------------------------------

    # Horizontal axis is z
    # Vertical axis is x

    #-- Choose larger lim range as lim --#
    if(zlim[1]-zlim[0] > xlim[1]-xlim[0]):
        lim = zlim
    else:
        lim = xlim

    #-- Calculate bin edges --#
    edgesz = np.linspace(*lim, nz + 1)
    edgesx = np.linspace(*lim, nx + 1)

    #-- Calculate bin centers --#
    centersz = 0.5 * (edgesz[1:] + edgesz[:-1])
    centersx = 0.5 * (edgesx[1:] + edgesx[:-1])

    #-- Import color scheme --#
    import matplotlib.colors
    cmap = matplotlib.colors.ListedColormap(
        sns.color_palette("husl", nz).as_hex())

    #-- Histogram data --#
    hist, _, _ = np.histogram2d(ZX[:, 1], ZX[:, 0], [edgesx,edgesz])

    #-- Create joint plot --#
    sns.set(style="ticks", font_scale=1.2)
    if(dec==True):
        data = pd.DataFrame(ZX, columns=['z', 'x'])
        g = sns.JointGrid(x="z", y="x", data=data, ratio=2, xlim=lim, ylim=lim, height=8,  space=0.05)
    else:
        data = pd.DataFrame(ZX, columns=['x', 'z'])
        g = sns.JointGrid(x="x", y="z", data=data, ratio=2, xlim=lim, ylim=lim, height=8,  space=0.05)

    g.ax_joint.imshow(hist, extent=lim+lim, origin='lower', interpolation='none', cmap='gray_r')

    # Remove grid lines
    g.ax_joint.grid(False)
    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)

    # Add meaningful labels
    if(dec==True):
        g.set_axis_labels(xlabel=r'$\mathcal{Z}$', ylabel=r'$\mathcal{X}$', fontsize=18)
    else:
        g.set_axis_labels(xlabel=r'$\mathcal{X}$', ylabel=r'$\mathcal{Z}$', fontsize=18)

    idx = np.arange(nz)

    hVertL=[]
    colL=[]
    cenL=[]

    for i in idx:
        g.ax_marg_x.hist(centersz, edgesz, weights=np.sum(hist,axis=0)*(idx == i), histtype='barstacked', color=cmap(i))
        hVertL.append(hist[:, i])
        colL.append(cmap(i))
        cenL.append(centersx)

    #print("l ",hVertL)
    #print("cenL ",cenL)
    g.ax_marg_y.hist(cenL, bins=edgesx, weights=hVertL, histtype='barstacked', color=colL, orientation='horizontal')
    g.ax_marg_x.set_title(title, fontsize=30)


    #-- Save dummy file for later plotting --#
    if save_figs:
        name = img_dir+name
        g.savefig(name, dpi=500)

    #-- Suppress Seaborn output --#
    if img_dir=='': # Implies this is being used in fullTransportPlot()
        plt.close()



def fullTransportPlot(Z, X, nzList, nxList, limzList, limxList, pltDim, titleList, dec=True):
    # -------------------------------------------------------------------------------------
    # Creates transport plots along all axes of data of data
    #
    # Inputs:  Z         = Z data
    #          X         = X data
    #          nzList    = A list of number of bins along the z axis to use in each subplot
    #          nxList    = A list of number of bins along the x axis to use in each subplot
    #          limzList  = A list of the tuple z limits to use in each subplot
    #          limxList  = A list of the tuple x limits to use in each subplot
    #          pltDim    = A tuple describing the subplots Ex. (1,2) means 1 row and 2 columns
    #          titleList = List of titles for the plots
    #          dec       = If dec=True then plot results for decoder, if dec=False then plot results for encoder
    #
    # Outputs: Creates transport plots along all dimensions
    # -------------------------------------------------------------------------------------

    # -- Check dimensions --#
    assert (Z.shape == X.shape)
    assert (len(nzList) == Z.shape[1])
    assert (len(nxList) == X.shape[1])
    assert (len(limzList) == len(nzList))
    assert (len(limxList) == len(nxList))
    assert (len(titleList) == len(nzList))

    # -- Create ZX data pairings for each dimension --#
    n = Z.shape[1]
    ZXList = [np.concatenate((Z[:, i].reshape(-1, 1), X[:, i].reshape(-1, 1)), axis=1)
              for i in range(n)]

    # -- Create full plot and and transport plots to subplots --#
    fig_size = (12 * pltDim[1], 12 * pltDim[0])
    fig = plt.figure(figsize=fig_size)

    for j in range(pltDim[1]):
        for i in range(pltDim[0]):
            k = j + pltDim[1] * i

            # Call transportplot which creates Transport.png
            transportPlot(ZXList[k], zlim=limzList[k], xlim=limxList[k], nz=nzList[k], nx=nxList[k], name='dummyTransport.png', dec=dec, save_figs=True, img_dir='')

            # Open image and add to subplot
            img = mpimg.imread('dummyTransport.png')
            ax = plt.subplot2grid(pltDim, (i, j))
            ax.set_title(titleList[k])
            ax.imshow(img)
            plt.axis('off')

    plt.show()

def diffPlot(z, diffxz, lim, dec=True):
    # -------------------------------------------------------------------------------------
    # Creates difference plots
    #
    # Inputs:  z        = z sorted numpy array Ex. z[isort,0:10] => shape = (100,10)
    #          diffxz   = diffxz sorted numpy array
    #          lim      = z axis limits (or x axis lims if dec=False)
    #          dec      = If dec=True then plot results for decoder, if dec=False then plot results for encoder
    #
    # Outputs: If dec=True then plots z vs x-z
    #          If dec=False then plots x vs z-x
    # -------------------------------------------------------------------------------------
    import matplotlib.cm as cm

    N = z.shape[0]

    # -- Make figure --#
    fig, ax1 = plt.subplots()

    # -- Colorize scatter plot points --#
    colors = cm.tab20b(np.linspace(0, 1, N))
    iarr = np.arange(N)

    for i, c in zip(iarr, colors):
        plt.scatter(z[i, :], diffxz[i, :], s=15, marker='s', color=c)

    ax1.set_xlim(lim)

    # -- Add labels --#
    if (dec == True):
        ax1.set_xlabel('z')
        ax1.set_ylabel('x - z')
    else:
        ax1.set_xlabel('x')
        ax1.set_ylabel('z - x')

    ax1.xaxis.label.set_size(16)
    ax1.yaxis.label.set_size(16)

    plt.savefig('dummyDiffPlot.png', dpi=500)
    plt.close()


def fullDiffPlot(Z, X, limList, pltDim, titleList, dec=True):
    # -------------------------------------------------------------------------------------
    # Creates difference plots along all axes of data
    #
    # Inputs:  Z         = List of fixed Z data torch tensor format
    #          X         = List of fixed X data torch tensor format
    #          limzList  = A list of the tuple z limits to use in each subplot
    #          pltDim    = A tuple describing the subplots Ex. (1,2) means 1 row and 2 columns
    #          titleList = List of titles
    #          dec       = Boolean object to determine whether we are plotting a decoder (True) or encoder (False)
    #
    # Outputs: Difference plots along all dimensions
    # -------------------------------------------------------------------------------------

    #-- Check dimensions --#
    assert (len(Z) == len(X))
    assert (len(titleList) == len(limList))

    num_data_points = len(Z)
    common_num_repeats = min([len(Z[t]) for t in range(num_data_points)])  # each data case may have a different number of samples (because the decoder can stochastically reject)
    print('common_num_repeats', common_num_repeats)
    for t in range(num_data_points):  # make sure each data case gets the same number of repeated samples
        assert len(X[t]) == len(Z[t])
        X[t] = X[t][:common_num_repeats]
        Z[t] = Z[t][:common_num_repeats]

    #-- Create full difference plots to subplots --#
    fig_size = (12 * pltDim[1], 12 * pltDim[0])
    fig = plt.figure(figsize=fig_size)

    for j in range(pltDim[1]):
        for i in range(pltDim[0]):
            k = j + pltDim[1] * i

            # Calculate difference arrays
            diff_xz = np.array([(X[t][:, k] - Z[t][:, k]) for t in range(len(Z))])
            z_arr = np.array([Z[t][:, k] for t in range(len(Z))])

            # Sort and make difference plot
            isort = np.argsort(z_arr[:, k])
            diffPlot(z_arr[isort, 0:10], diff_xz[isort, 0:10], lim=limList[k], dec=dec)

            # Open image and add to subplot
            img = mpimg.imread('dummyDiffPlot.png')
            ax = plt.subplot2grid(pltDim, (i, j))
            ax.set_title(titleList[k], fontsize=30)
            ax.imshow(img)
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def sciNotationString(x, p='%.2E'):
    # --------------------------------------------------------------------------------------
    # Takes a number x and returns this number expressed in scientific notation up to precision p
    #
    # Inputs:  x = Number
    #          p = Desired precision
    #
    # Outputs: x as a formatted string in scientific notation for plotting
    # --------------------------------------------------------------------------------------

    s = p% x
    snew = r'$'+s[0:2+int(p[2])]+'$ x $10^{'+s[-3:]+'}$'

    return snew

def ratioPlotSingle(Y1, Y2, varBins, xRange, yRange, ratioRange, axisName, statList12=[], legend=False, Wunit=r'[GeV$^2$]', save_figs=False, img_dir='Images', ablLambda='',  ablBeta=''):
    # --------------------------------------------------------------------------------------
    # Plots a histogram of two samples (i.e. z and \tilde{z}) of a single variable (e.g. E)
    # It also plots the ratio to truth (i.e. \tilde{z}/z) in a lower subplot.
    #
    # Inputs:  Y1             = True sample (i.e. z) of a single variable
    #          Y2             = Simulated sample (i.e. \tilde{z}) of a single variable
    #          varBins        = Bins to use when plotting
    #          xRange, yRange = Ranges of plot
    #          ratioRange     = y axis range of ratio plot
    #          axisName       = Name to put on x-axis label (e.g. E)
    #          statList12     = Statistics calculated between Y1 and Y2; note this makes the plot rather crowded
    #          legend         = Whether to include a plot legend
    #          Wunit          = Unit of the Wasserstein distance to use if plotting statistics
    #          save_figs      = Whether to save resulting plot as an image
    #          img_dir        = Directory to save image to
    #          ablLambda      = Default ''. If '', do not add lambda value to save file name. 
    #                           If not '', appends this to end of file name Ex. *_<ablLambda>.png.
    #                           Recommend to set ablLambda = e.g. 'lamb=1'
    #          ablBeta        = Default ''. If '', do not add beta value to save file name.
    #                           If not '', appends this to end of file name Ex. *_<ablBeta>.png.
    #                           Recommend to set ablBeta = e.g. 'beta=50'
    #
    # Outputs: Plot; saves image to file if indicated
    #
    # Inspired by figure 4 from here: https://arxiv.org/pdf/1907.03764.pdf
    # --------------------------------------------------------------------------------------

    #-- Create plot layout --#
    fontStr = 'Arial'
    pltDim=(4,3)
    fig_size = (2*pltDim[1], 2*pltDim[0])

    fig = plt.figure(constrained_layout=False, figsize=fig_size)
    gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)

    axVar   = fig.add_subplot(gs[0:-1, :])
    axRatio = fig.add_subplot(gs[-1, :])

    #-- Create Histograms --#
    n1, _, _ = axVar.hist(Y1, bins=varBins, histtype='step', label=r'Ground truth: $z$', density=False, color='black', linewidth=1.9) #color='darkred'
    n2, _, _ = axVar.hist(Y2, bins=varBins, histtype='step', label=r'OTUS encoder: $x \rightarrow \tilde{z}$', density=False, color='#00BFFF', linewidth=1.9, linestyle='dashed')
    #-- Add in statistics details --#
    if(len(statList12)!=0):
        stat = True
    else:
        stat = False

    if(stat):
        nmax  = np.max([np.max(n1),np.max(n2)])

        Delta = 1./0.55
        flist = np.array([0.75, 0.69, 0.64, 0.58])

        ymax = Delta*nmax
        yRange = (yRange[0], ymax)
        ylist = flist*ymax

        xcenter12 = xRange[0] + 0.5*(xRange[1] - xRange[0])

        axVar.text(xcenter12, ylist[0], r"$z$ vs $\tilde{z}$", horizontalalignment='center', fontsize=16)
        axVar.text(xcenter12, ylist[1], r'$W$: '+sciNotationString(statList12[0])+' '+Wunit, horizontalalignment='center', fontsize=14)
        axVar.text(xcenter12, ylist[2], r"$(\chi^2_R, dof)$: (%.3f, %d)"% (statList12[3], statList12[4]), horizontalalignment='center', fontsize=14)
        axVar.text(xcenter12, ylist[3], r'$KS$: '+sciNotationString(statList12[1]), horizontalalignment='center', fontsize=14)
    else:
        nmax  = np.max([np.max(n1),np.max(n2)])
        Delta = 1./0.75

        ymax = Delta*nmax
        yRange = (yRange[0], ymax)

    #-- Calculate Ratio --#
    assert(np.all(n1)>0.)
    ratio = n2/n1

    ratioErr = ratio * np.sqrt(1/n2 + 1/n1)

    binCenters = []
    for i in range(len(varBins)-1):
        binCenters.append(varBins[i] + 0.5*(varBins[i+1] - varBins[i]))

    axRatio.plot(xRange,[1,1], '--', color='gray', zorder=1)
    axRatio.scatter(binCenters, ratio, s = 15, marker='o', color='#00BFFF', zorder=2)
    axRatio.errorbar(binCenters, ratio, yerr=ratioErr, ls='none', color='#00BFFF', zorder=3)

    #-- Format plot --#

    # Labels
    axVar.set(ylabel='Counts')
    axVar.yaxis.label.set_size(16)

    axRatio.set(xlabel=axisName)
    #axRatio.set(ylabel=r'$\tilde{z}/z$')
    axRatio.set(ylabel='Ratio to truth')
    axRatio.yaxis.label.set_size(16)
    axRatio.xaxis.label.set_size(18)

    # Limits and tick marks
    axVar.ticklabel_format(axis='y', style='sci', scilimits=(0,3), useMathText=True)
    axVar.set_xlim(xRange[0], xRange[1])
    axVar.set_ylim(yRange[0], yRange[1])
    axVar.set_xticks([])

    axRatio.set_xlim(xRange[0], xRange[1])
    axRatio.set_ylim(ratioRange[0], ratioRange[1])

    # Make background white
    axVar.set_facecolor('white')
    axRatio.set_facecolor('white')

    # Remove grid lines
    axVar.grid(False)
    axRatio.grid(False)

    # Bring border back
    axVar.spines['bottom'].set_color('black')
    axVar.spines['top'].set_color('black')
    axVar.spines['right'].set_color('black')
    axVar.spines['left'].set_color('black')

    axRatio.spines['bottom'].set_color('black')
    axRatio.spines['top'].set_color('black')
    axRatio.spines['right'].set_color('black')
    axRatio.spines['left'].set_color('black')

    # Add in legend
    if(legend):
        #axVar.legend(fontsize='x-large', loc='center right', facecolor='white')
        axVar.legend(fontsize='x-large', loc='upper left', facecolor='white')

    # Save Image (optional)
    if save_figs:
        # Remove special characters for label
        axisName = axisName.replace("[GeV]", "")
        axisName = axisName.replace("[GeV/c]", "")
        axisName = axisName.replace("[GeV/$c^2$]", "")
        axisName = axisName.replace("{", "")
        axisName = axisName.replace("}", "")
        axisName = axisName.replace(" ", "")
        axisName = axisName.replace("$", "")

        # Save image
        if ablLambda == '' and ablBeta == '':
            plt.savefig(img_dir+'/ratio_zpred_z_'+axisName+'.png', dpi=500)
        elif ablLambda != '' and ablBeta == '':
            plt.savefig(img_dir+'/ratio_zpred_z_'+axisName+'_'+ablLambda+'.png', dpi=500)
        elif ablLambda == '' and ablBeta != '':
            plt.savefig(img_dir+'/ratio_zpred_z_'+axisName+'_'+ablBeta+'.png', dpi=500)
            
    plt.show()

def ratioPlotDouble(Y1, Y2, Y3, varBins, xRange, yRange, ratioRange, axisName, statList12=[], statList13=[], legend=False, Wunit=r'[GeV$^2$]', save_figs=False, img_dir='Images', ablLambda='',  ablBeta=''):
    # --------------------------------------------------------------------------------------
    # Plots a histogram of three samples (i.e. z, \tilde{z}, and \tilde{z}') of a single variable (e.g. E)
    # It also plots the ratio to truth (i.e. \tilde{z}/z and \tilde{z}'/z) in a lower subplot.
    #
    # Inputs:  Y1             = True sample (i.e. z) of a single variable
    #          Y2             = First simulated sample (i.e. \tilde{z}) of a single variable
    #          Y3             = Second simulated sample (i.e. \tilde{z}') of a single variable
    #          varBins        = Bins to use when plotting
    #          xRange, yRange = Ranges of plot
    #          ratioRange     = y axis range of ratio plot
    #          axisName       = Name to put on x-axis label (e.g. E)
    #          statList12     = Statistics calculated between Y1 and Y2; note this makes the plot rather crowded
    #          statList13     = Statistics calculated between Y1 and Y3; note this makes the plot rather crowded
    #          legend         = Whether to include a plot legend
    #          Wunit          = Unit of the Wasserstein distance to use if plotting statistics
    #          save_figs      = Whether to save resulting plot as an image
    #          img_dir        = Directory to save image to
    #          ablLambda      = Default ''. If '', do not add lambda value to save file name. 
    #                           If not '', appends this to end of file name Ex. *_<ablLambda>.png.
    #                           Recommend to set ablLambda = e.g. 'lamb=1'
    #          ablBeta        = Default ''. If '', do not add beta value to save file name.
    #                           If not '', appends this to end of file name Ex. *_<ablBeta>.png.
    #                           Recommend to set ablBeta = e.g. 'beta=50'
    #
    # Outputs: Plot; saves image to file if indicated
    #
    # Inspired by figure 4 from here: https://arxiv.org/pdf/1907.03764.pdf
    # --------------------------------------------------------------------------------------

    #-- Create plot layout --#
    fontStr = 'Arial'
    pltDim=(4,3)
    fig_size = (2*pltDim[1], 2*pltDim[0])

    fig = plt.figure(constrained_layout=False, figsize=fig_size)
    gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)

    axVar   = fig.add_subplot(gs[0:-1, :])
    axRatio = fig.add_subplot(gs[-1, :])

    #-- Create Histograms --#
    n1, _, _ = axVar.hist(Y1, bins=varBins, histtype='step', label=r'Ground truth: $x$', density=False, color='black', linewidth=1.9)
    n2, _, _ = axVar.hist(Y2, bins=varBins, histtype='step', label=r'OTUS encoder-decoder: $x \rightarrow \tilde{z} \rightarrow \tilde{x}$', density=False, color='green', linewidth=1.9, linestyle=(0, (1, 1))) #color='#ED7014'
    n3, _, _ = axVar.hist(Y3, bins=varBins, histtype='step', label=r'OTUS decoder: $z \rightarrow \tilde{x}^\prime$', density=False, color='#9104DC', linewidth=1.9, linestyle='dashed')

    #-- Add in statistics details --#
    if(len(statList12)!=0 and len(statList13)!=0):
        stat = True
    else:
        stat = False

    if(stat):
        nmax  = np.max([np.max(n1),np.max(n2),np.max(n3)])

        Delta = 1./0.50
        flist = np.array([0.69, 0.63, 0.58, 0.53])

        ymax = Delta*nmax
        yRange = (yRange[0], ymax)
        ylist = flist*ymax

        xcenter12 = xRange[0] + 0.25*(xRange[1] - xRange[0])
        xcenter13 = xRange[0] + 0.75*(xRange[1] - xRange[0])

        axVar.text(xcenter12, ylist[0], r"$x$ vs $\tilde{x}$", horizontalalignment='center', fontsize=16)
        axVar.text(xcenter12, ylist[1], r'$W$: '+sciNotationString(statList12[0])+' '+Wunit, horizontalalignment='center', fontsize=13)
        axVar.text(xcenter12, ylist[2], r"$(\chi^2_R, dof)$: (%.3f, %d)"% (statList12[3], statList12[4]), horizontalalignment='center', fontsize=13)
        axVar.text(xcenter12, ylist[3], r'$KS$: '+sciNotationString(statList12[1]), horizontalalignment='center', fontsize=13)

        axVar.text(xcenter13, ylist[0], r"$x$ vs $\tilde{x}^\prime$", horizontalalignment='center', fontsize=16)
        axVar.text(xcenter13, ylist[1], r'$W$: '+sciNotationString(statList13[0])+' '+Wunit, horizontalalignment='center', fontsize=13)
        axVar.text(xcenter13, ylist[2], r"$(\chi^2_R, dof)$: (%.3f, %d)"% (statList13[3], statList13[4]), horizontalalignment='center', fontsize=13)
        axVar.text(xcenter13, ylist[3], r'$KS$: '+sciNotationString(statList13[1]), horizontalalignment='center', fontsize=13)
    else:
        nmax  = np.max([np.max(n1),np.max(n2),np.max(n3)])
        Delta = 1./0.65

        ymax = Delta*nmax
        yRange = (yRange[0], ymax)

    #-- Calculate Ratios --#
    assert(np.all(n1)>0.)
    ratio12 = n2/n1
    ratioErr12 = ratio12 * np.sqrt(1/n2 + 1/n1)

    ratio13 = n3/n1
    ratioErr13 = ratio13 * np.sqrt(1/n3 + 1/n1)

    binCenters = []
    for i in range(len(varBins)-1):
        binCenters.append(varBins[i] + 0.5*(varBins[i+1] - varBins[i]))

    axRatio.plot(xRange,[1,1], '--', color='gray', zorder=1)

    axRatio.scatter(binCenters, ratio12, s = 15, marker='o', label=r'$\tilde{x}/x$', color='green', zorder=2)
    axRatio.errorbar(binCenters, ratio12, yerr=ratioErr12, ls='none', color='green', zorder=3)

    axRatio.scatter(binCenters, ratio13, s = 15, marker='s', label=r'$\tilde{x}^\prime/x$', color='#9104DC', zorder=4)
    axRatio.errorbar(binCenters, ratio13, yerr=ratioErr13, ls='none', color='#9104DC', zorder=5)

    #-- Format plot --#

    # Labels
    axVar.set(ylabel='Counts')
    axVar.yaxis.label.set_size(16)

    axRatio.set(ylabel='Ratio to truth')
    axRatio.yaxis.label.set_size(16)
    axRatio.set(xlabel=axisName)
    axRatio.xaxis.label.set_size(18)
    #axRatio.legend(fontsize='large', loc='upper left', ncol=2, facecolor='white')

    # Limits and tick marks
    axVar.ticklabel_format(axis='y', style='sci', scilimits=(0,3), useMathText=True)
    axVar.set_xlim(xRange[0], xRange[1])
    axVar.set_ylim(yRange[0], yRange[1])
    axVar.set_xticks([])

    axRatio.set_xlim(xRange[0], xRange[1])

    axRatio.set_ylim(ratioRange[0], ratioRange[1])

    # Make background white
    axVar.set_facecolor('white')
    axRatio.set_facecolor('white')

    # Remove grid lines
    axVar.grid(False)
    axRatio.grid(False)

    # Bring border back
    axVar.spines['bottom'].set_color('black')
    axVar.spines['top'].set_color('black')
    axVar.spines['right'].set_color('black')
    axVar.spines['left'].set_color('black')

    axRatio.spines['bottom'].set_color('black')
    axRatio.spines['top'].set_color('black')
    axRatio.spines['right'].set_color('black')
    axRatio.spines['left'].set_color('black')

    # Add in legend
    if(legend):
        #axVar.legend(fontsize='x-large', loc='center right', facecolor='white')
        axVar.legend(fontsize='x-large', loc='upper left', facecolor='white')

    # Save Image (optional)
    if save_figs:
        # Remove special characters for label
        axisName = axisName.replace("[GeV]", "")
        axisName = axisName.replace("[GeV/c]", "")
        axisName = axisName.replace("[GeV/$c^2$]", "")
        axisName = axisName.replace("{", "")
        axisName = axisName.replace("}", "")
        axisName = axisName.replace(" ", "")
        axisName = axisName.replace("$", "")

        # Save image
        if ablLambda == '' and ablBeta == '':
            plt.savefig(img_dir+'/ratio_xpred_xpredtruth_x_'+axisName+'.png', dpi=500)
        elif ablLambda != '' and ablBeta == '':
            plt.savefig(img_dir+'/ratio_xpred_xpredtruth_x_'+axisName+'_'+ablLambda+'.png', dpi=500)
        elif ablLambda == '' and ablBeta != '':
            plt.savefig(img_dir+'/ratio_xpred_xpredtruth_x_'+axisName+'_'+ablBeta+'.png', dpi=500)

    plt.show()


def plotFunction(dataList, pltDim, binsList, particleNameList=[], nameList=[]):
    # --------------------------------------------------------------------------------------
    # Generic plotting function to use in experiment notebooks
    #
    # Inputs:  dataList         = Number of functions to compare (either 2 or 3). Order is either
    #                             [z, \tilde{z}] or [x, \tilde{x}, \tilde{x}']
    #          pltDim           = Tuple of (nrows, ncols)
    #          binsList         = List of bins to use in each subplot as an array.
    #          particleNameList = If known, list of the particles for which the plot is being made.
    #                             These will be the y-axis labels on the axs[i,0] subplots.
    #          nameList         = If using this function to plot a single plot, then input the names
    #                             for the x and y axis. Ex. nameList = [r'x-axis', r'y-axis'].
    #                             If using this to plot a series of random slices, leave this blank as well.
    #                             If using this to plot a series of known quantities include their x-axis labels here.
    #                             Ex. nameList = [r'x-axis1', r'x-axis2', ... r'x-axisn']
    #
    # Outputs: Plot
    #-------------------------------------------------------------------------------------------------------------------

    # Ensure that data all has the same number of samples for plotting
    sizeList = [] 
    for i in range(len(dataList)):
        sizeList.append(dataList[i].shape[0])

    nmin = np.min(sizeList)

    for i in range(len(dataList)):
        dataList[i] = dataList[i][0:nmin]
    
    # Check whether we are plotting a single plot or multiple subplots
    singlePlot = False
    if (dataList[0].ndim == 1):
        singlePlot = True

    # Check whether we are plotting z-space or x-space data and assign histogram parameters accordingly
    assert (len(dataList)==2 or len(dataList)==3)
    if len(dataList)==3:
        labelList     = [r'Ground truth: $x$', r'OTUS encoder-decoder: $x \rightarrow \tilde{z} \rightarrow \tilde{x}$', r'OTUS decoder: $z \rightarrow \tilde{x}^\prime$']
        colorList     = ['black', 'green', '#9104DC']
        linestyleList = ['solid', 'dotted', 'dashed']
    else:
        labelList     = [r'Ground truth: $z$', r'OTUS encoder: $x \rightarrow \tilde{z}$']
        colorList     = ['black','#00BFFF']
        linestyleList = ['solid', 'dashed']

    # Set the dimensions of the figure
    nrows, ncols = pltDim[0], pltDim[1]
    #!assert nrows*ncols == 1 or nrows*ncols == dataList[0].shape[1]  # Check dimensions are as expected
    if singlePlot:
        fig_size = (8, 8)
    else:
        fig_size = (6*pltDim[1], 6*pltDim[0])

    fig, axs = plt.subplots(pltDim[0], pltDim[1], figsize=fig_size, squeeze=False)
    fontStr = 'Arial'

    # Plot histograms of the data
    for i in range(nrows):
        for j in range(ncols):
            k = j + ncols*i

            #!
            if k >= len(binsList):
                continue

            nMaxList = []

            # Loop over types of data
            for l in range(len(dataList)):

                if singlePlot:
                    n, _, _ = axs[i, j].hist(dataList[l], bins=binsList[k], histtype='step', label=labelList[l], density=False, color=colorList[l], linewidth=1.9, linestyle=linestyleList[l])
                else:
                    n, _, _ = axs[i, j].hist(dataList[l][:,k], bins=binsList[k], histtype='step', label=labelList[l], density=False, color=colorList[l], linewidth=1.9, linestyle=linestyleList[l])

                nMaxList.append(np.max(n))

            #-- Format plot --#

            # Labels
            if len(particleNameList) == 0 and not(singlePlot):
                axs[i, 0].set(ylabel='Counts')
                axs[i, 0].yaxis.label.set_size(16)

                if(len(nameList) == 0):
                    axs[i,j].set(xlabel='Random Slice %d'%k)
                    axs[i,j].xaxis.label.set_size(12)
                else:
                    axs[i,j].set(xlabel=nameList[k])
            elif len(particleNameList) != 0 and not(singlePlot):
                axs[i, 0].set(ylabel=r'%s Counts'%particleNameList[i])
                axs[i, 0].yaxis.label.set_size(16)

                columnNameList = [r'$p_x$', r'$p_y$', r'$p_z$', r'$E$']
                axs[i,j].set(xlabel=columnNameList[j])
                axs[i,j].xaxis.label.set_size(12)
            elif singlePlot and len(nameList) != 0:
                axs[i, 0].set(ylabel=r'%s Counts'%nameList[1])
                axs[i, 0].yaxis.label.set_size(16)

                axs[i,j].set(xlabel=nameList[0])
                axs[i,j].xaxis.label.set_size(16)
            else:
                axs[i, 0].set(ylabel=r'Counts')
                axs[i, 0].yaxis.label.set_size(16)

            # Tick marks and limits
            axs[i,j].ticklabel_format(axis='y', style='sci', scilimits=(0,3), useMathText=True)

            nmax  = np.max(nMaxList)
            Delta = 1./0.75
            ymax = Delta*nmax
            yRange = (0, ymax)
            axs[i,j].set_ylim(yRange[0], yRange[1])

            # Make background white
            axs[i,j].set_facecolor('white')

            # Remove grid lines
            axs[i,j].grid(False)

            # Bring border back
            axs[i,j].spines['bottom'].set_color('black')
            axs[i,j].spines['top'].set_color('black')
            axs[i,j].spines['right'].set_color('black')
            axs[i,j].spines['left'].set_color('black')

            #-- Add in 1D Wasserstein Distance Value --#

            # Calculate 1D Wasserstein Distance
            WDList = []
            for l in range(1, len(dataList)):
                if singlePlot:
                    WDList.append(wd_1d(dataList[0], dataList[l]))
                else:
                    WDList.append(wd_1d(dataList[0][:,k], dataList[l][:,k]))

            # Add Text to Plot
            Wunit = r'GeV$^2$'
            xRange = [binsList[k][0], binsList[k][-1]]
            xcenter = xRange[0] + 0.5*(xRange[1] - xRange[0])

            if len(dataList)==3:
                if singlePlot:
                    farr = np.array([.95, .90])
                else:
                    farr = np.array([.93, .87])

                ylist = farr*ymax
                axs[i,j].text(xRange[1], ylist[0], r'$W(x, \tilde{x})$: '+sciNotationString(WDList[0], p='%.5E')+' '+Wunit, horizontalalignment='right', color=colorList[1], fontsize=11)
                axs[i,j].text(xRange[1], ylist[1], r'$W(x, \tilde{x}^\prime)$: '+sciNotationString(WDList[1], p='%.5E')+' '+Wunit, horizontalalignment='right', color=colorList[2], fontsize=11)

                #-- Add in Legend --#
                axs[i,j].legend(fontsize='small', loc='upper left', facecolor='white')

            else:
                if singlePlot:
                    farr = np.array([.95])
                else:
                    farr = np.array([.93])

                ylist = farr*ymax
                axs[i,j].text(xRange[1], ylist[0], r'$W(z, \tilde{z})$: '+sciNotationString(WDList[0], p='%.5E')+' '+Wunit, horizontalalignment='right', color=colorList[1], fontsize=12)

                #-- Add in Legend --#
                axs[i,j].legend(fontsize='medium', loc='upper left', facecolor='white')

    plt.tight_layout()
    plt.show()
