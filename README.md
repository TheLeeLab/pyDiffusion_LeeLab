
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10932778.svg)](https://doi.org/10.5281/zenodo.10932778)



## Installation Instructions

The package "pyDiffusion_LeeLab" (written in support of various experiments going on in-lab) is a set of python classes that can be run from scripts, interactive notebooks and so on to analyse single particle tracking data. Example notebooks are provided, showing user analyses. The code has been tested in Python 3.10.12.

A list of package requirements are noted in the "requirements.txt" file. These can be installed with the command:

`pip install -r requirements.txt`

Documentation can be found at [read the docs](https://pydiffusion_leelab.readthedocs.io/).

The codes are based on optimal analyses for diffusion coefficients set out in four papers that are well worth a read.

1. [Michalet and Berglund, Optimal diffusion coefficient estimation in single-particle tracking](https://link.aps.org/doi/10.1103/PhysRevE.85.061916)
2. [Vestergaard, Blainey and Flyvbjerg, Optimal estimation of diffusion coefficients from single-particle trajectories](https://link.aps.org/doi/10.1103/PhysRevE.89.022726)

the Michalet and Berglund is dependent on the authors' (separate) previous two papers:

3. [Michalet, Mean square displacement analysis of single-particle trajectories with localization error: Brownian motion in an isotropic medium](https://link.aps.org/doi/10.1103/PhysRevE.82.041914)
4. [Berglund, Statistics of camera-based single-particle tracking](https://link.aps.org/doi/10.1103/PhysRevE.82.011917)

## Contributing

Patches and contributions are very welcome! Please see CONTRIBUTING.md and the CODE_OF_CONDUCT.md for more details. Please report any bugs by opening an issue.