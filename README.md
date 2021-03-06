![plot](OTUSLogo.png)

# Optimal Transport based Unfolding and Simulation (OTUS)
This repo contains the code and data used in the paper "Foundations of a Fast, Data-Driven, Machine-Learned Simulator": https://arxiv.org/abs/2101.08944


## Software Setup

### Software Used for Data Generation
* Madgraph5 v.2.6.3.2 [1]

* Pythia v.8.240 [2]

* Delphes v.3.4.1 [3]

* ROOT v.6.08/00 [4]

### Software Environment for the OTUS Experiments
To reproduce the Python environment for the OTUS experiments, do the following:

 1. Create a barebones Python 3.6.9 installation, including the (usually pre-installed) pip package manager which we'll use in step 2. There are many ways to do this (e.g., with [virtualenv](https://pypi.org/project/virtualenv/), or [venv](https://docs.python.org/3/library/venv.html)); we recommend the following setup with [conda](https://anaconda.org/anaconda/conda): `conda create --name py36-otus python=3.6.9; conda activate py36-otus`. conda will make sure to set up pip within this new environment.

2. Within the above Python environment, run `pip install -r requirements.txt` to install the required packages.

The main dependencies are Python 3.6.9, PyTorch 1.6, Numpy 1.17.4, and Jupyter 1.0 (see `requirements-core.txt`). You may use other versions of these packages, but may not be able to exactly reproduce the reported results.

Additionally, the CUDA 10.0 and CuDNN 7.6 libraries were used for experiments run on the GPU.

## Computing
The following computing devices were used for the OTUS experiments:

* Intel® Xeon® Gold 5218 CPU (the results should be reproducible across most Intel CPUs)

* NVIDIA® TITAN RTX™ GPU

## Additional Helpful Links
* SWAE paper: https://arxiv.org/abs/1804.01947, https://openreview.net/forum?id=H1xaJn05FQ

* SWAE code:  https://github.com/skolouri/swae  (PyTorch version: https://github.com/eifuentes/swae-pytorch)

## References
[1] Johan Alwall et al. MadGraph 5 : Going Beyond. arxiv:1106.0522. 2011. URL: http://arxiv.org/abs/1106.0522.

[2] Torbjorn Sjostrand, Stephen Mrenna, and Peter Z. Skands. “PYTHIA 6.4 Physics and Manual”. In: JHEP 0605
(2006), p. 026. DOI: 10.1088/1126-6708/2006/05/026. arXiv: hep-ph/0603175 [hep-ph].

[3] J. de Favereau et al. “DELPHES 3, A modular framework for fast simulation of a generic collider experiment”.
In: JHEP 02 (2014), p. 057. DOI: 10.1007/JHEP02(2014)057. arXiv: 1307.6346 [hep-ex].

[4] R. Brun and F. Rademakers. “ROOT: An object oriented data analysis framework”. In: Nucl. Instrum. Meth. A
389 (1997). Ed. by M. Werlen and D. Perret-Gallix, pp. 81–86. DOI: 10.1016/S0168-9002(97)00048-X.
