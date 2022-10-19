# Dual Communities

This code accompanies the publication that can be found at [ArXiv:&nbsp;2105.06687](https://arxiv.org/abs/2105.06687).
Community structures represent an important feature of networks. In the connected publication and also this code, we show introduce a different class of communities: dual communities. Dual communities arise naturally in optimized network structures.
 They shape, similarly to the conventional communities (i.e. primal communities), important network features such as robustness.

## Installation

This code was developed with `python 3.10` and by using conda as an environment management system.

The requirements can be found in `requirements.txt`. 

To setup the exact conda environment in the path=`<path>`that was used in developing this code, you can use the `dual_env.yml` file and activate it.

```shell
conda env create --prefix=<path> -f dual_env.yml
conda activate <path>
```

Furthermore, to run the examples in the Notebook `examples.ipynb`, you need to run 

```shell
conda install -c conda-forge nb_conda_kernels notebook
```
to be able to run Jupyter Notebooks.

## Usage

### Basic Usage
Since the code uses [sagemath](https://doc.sagemath.org/html/en/installation/conda.html) for the conversion of graphs to dual graphs, it needs to be run in the sage console instead of the more common `python` or `ipython`-consoles.

The basic usage for a few examples can be found in the jupyter-notebook 
[examples.ipynb](./examples.ipynb).

### Results Presented in Publication

The scripts found in   [dual_communities/results](./dual_communities/results) can be used to reproduce results presented in the publication.

The scripts `paper_data` and `paper_plots` are meta scripts that collect data generation functions and plot functions, respectively.


## Data 
The publication uses, in addition to the elementary example, graphs generated by two data sources.

1. Leaf venation networks which are available upon request from the authors of [DOI:&nbsp;10.1371/journal.pcbi.1004680](https://doi.org/10.1371/journal.pcbi.1004680).

2. The topology of the Central European grid was generated from PyPSA-Euro and is available at  [DOI:&nbsp;10.5281/zenodo.3886532](https://doi.org/10.5281/zenodo.3886532).






