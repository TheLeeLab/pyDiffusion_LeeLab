.. image:: https://zenodo.org/badge/791338990.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.11066161

Introduction
============

``pyDiffusion_LeeLab`` is a Python package written in support of SPT work from the lab of Steven F. Lee. The code is a set of python classes that can be run from scripts, interactive notebooks and so on to analyse tracking data. Example notebooks are provided, showing user analyses. The code has been tested in Python 3.10.12.

A list of package requirements are noted in the "requirements.txt" file. These can be installed with the command:

``pip install -r requirements.txt``

The current implementation has been developed in Python 3 and tested on an Ubuntu 20.04 machine, as well as several Windows 10 machines.

Literature
**********
The codes are based on optimal analyses for diffusion coefficients set out in four papers that are well worth a read.

1. [Michalet and Berglund, Optimal diffusion coefficient estimation in single-particle tracking](https://link.aps.org/doi/10.1103/PhysRevE.85.061916)
2. [Vestergaard, Blainey and Flyvbjerg, Optimal estimation of diffusion coefficients from single-particle trajectories](https://link.aps.org/doi/10.1103/PhysRevE.89.022726)

the Michalet and Berglund is dependent on the authors' (separate) previous two papers:

3. [Michalet, Mean square displacement analysis of single-particle trajectories with localization error: Brownian motion in an isotropic medium](https://link.aps.org/doi/10.1103/PhysRevE.82.041914)
4. [Berglund, Statistics of camera-based single-particle tracking](https://link.aps.org/doi/10.1103/PhysRevE.82.011917)


Getting Started with pyDiffusion_LeeLab
***************************************

To get started, one can try the example notebooks, which are individual notebooks that highlight the code's capabilities.
