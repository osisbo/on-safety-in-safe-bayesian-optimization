# On Safety in Safe Bayesian Optimization
This repository contains python implementations of SafeOpt++ and the Lipschitz-only Safe Bayesian Optimization (LoSBO) algorithm based on a SafeOpt implementation [1] and code to reproduce the results from the paper "On Safety in Safe Bayesian Optimization".

## Installation
Clone the repository. Then, into an environment with e.g. Python 3.8.5, you can install all needed packages with:

```pip install -r requirements.txt```

## Reproducing the experiments with synthetic target functions
To reproduce the experiments with synthetic target functions of the paper, you can now run the python scripts corresponding to the different experiments from the command line.
The scripts take three different arguments:
1. argument: float corresponding to the kernel length scale of the function class (e.g. set to 0.2)
2. argument: int corresponding to the prescribed RKHS-norm of the function class (e.g. set to 2)
3. argument: flag for deciding whether to use the original random seeds of the paper or not (for using the random seeds of the paper set to 1, otherwise set to 0).

For example, for running the experiment comparing SafeOpt++ and SafeOpt on the function class with kernel length scale of 0.2 and prescribed RKHS-norm 2
using the original random seeds of the paper, you can run the script *experiments_safeoptpp.py* with:

```python experiments_safeoptpp.py 0.2 2 1```

Running the scripts may take a few hours or days. Also, the scripts *experiments_safeoptpp.py* and *experiments_losbo.py* use multiple processes
and therefore also multiple cores on your machine (*experiments_losbo.py* uses 20 cores and *experiments_losbo.py* uses 10).
As results, each process produces a folder with the result data stored in a pickle file (as a dictionary) and a text file summarizing the most important metrics.
In order to get the overall results of the scripts *experiments_safeoptpp.py* and *experiments_losbo.py*,
you can use the merging scripts which can be found in the *utils* folder. These scripts should be started in a folder only containing the 20 or 10 folders
produced by *experiments_safeoptpp.py* or *experiments_losbo.py* and will produce a new folder containing the summarized results.

The data is also directly available at [Google Drive](https://drive.google.com/drive/folders/1saMie4KguzUCTjptYWpt_bV-hhHcqvew?usp=sharing). In the dictionaries containing the results, the string "fiedler" corresponds to SafeOpt++, the string "chowdhury" corresponds to SafeOpt and the string "fiedler-lipschitz" corresponds to LoSBO.

## Reproducing the gym experiment
To reproduce the gym experiment, you can just run the python script *experiment_controller_optimization* with:

```python experiment_controller_optimization.py```

## Reproducing the figures
For reproducing the figures of the paper, you can use *results_plotter_safeoptpp.py* and *results_plotter_losbo.py*
which can be found in the *utils* folder as well. Both scripts take again three command line arguments:
1. argument: path to the folder containing the overall results of either *experiments_safeoptpp.py* or *experiments_losbo.py*
2. argument: float corresponding to the kernel length scale of the function class (e.g. set to 0.2)
3. argument: int corresponding to the prescribed RKHS-norm of the function class (e.g. set to 2).

For example, for reproducing the figures comparing SafeOpt++ and SafeOpt on the function class with kernel length scale of 0.2
and prescribed RKHS-norm 2, you can run the script *results_plotter_safeoptpp.py* with:

```python results_plotter_safeoptpp.py [PATH TO FOLDER] 0.2 2```

Note that on Linux, for example, the plotting scripts required the cm-super and dvipng packages to be installed which could be done with

```sudo apt install cm-super``` and ```sudo apt-get install dvipng```.

## References
[1] F. Berkenkamp, A. P. Schoellig, A. Krause, Safe Controller Optimization for Quadrotors with Gaussian Processes in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2016, pp. 491-496.
