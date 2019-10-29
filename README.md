# RL-Based Composite MRAC

## Installation

You need to install a python package, the FDCL NRF Fym.
There are two ways to install this package.

1. Using an enclosed submodule

First, you can clone this repository with a fym submodule.
```bash
$ git clone --recurse-submodules https://github.com/seong-hun/rlcmrac.git
```
Then, create a virtual environment.
We recommend to use `conda` to install the virtual environment with a provided `environment.yml` file.
```bash
$ conda env create -n rlcmrac -f environment.yml
$ conda activate rlcmrac
```
Now change to the submodule directory and install the `fym` package.
```bash
$ cd submodules/fym
$ pip install -e .
```

2. Directly clone the Fym repository

*TBD*

## Run the Simulation

To run all the simulations,
```bash
$ python main.py --all
```

If you want to plot the result after the running, you can use `--plot` option.
```bash
$ python main.py --all --plot
```
