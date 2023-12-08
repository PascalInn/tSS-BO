# tSS-BO
An opensource prototype code for synthetic benchmark experiments in coming conference D.A.T.E. 2024 'tSS-BO: Scalable Bayesian Optimization for Analog Circuit Sizing via Truncated Subspace Sampling'

Author: Tianchen Gu tcgu18@fudan.edu.cn

# Requirement
PyTorch

GPyTorch

NumPy

SciPy

*BoTorch (synthetic functions from BoTorch package, optional)

# Introduction
To solve the high-dimensional problems, we combine the CMA-ES with BO method. We enhance the CMA method for high-dimensional occasions using normalized approximate gradients and last center update steps. The gradients are approximated through two different approaches, i.e., gradient sketching and local GP models. BO methods are performed to select the random samples of the truncated Gaussian distribution, parameterized by the mean (center), covariance (adaptive covariance and the deviation sigma), truncated by the search space. 

# Usage
You can define the problem with normalized input range of [-0.5, 0.5] ^ d. The problem function should be able to be fed with n X d tensor, and to return a n X m tensor, where the first column of the result matrix is the objective and other m - 1 columns give the constraints.

This method is designed for MINIMIZATION problems with <=0 constraints.

The optimization process is in the './src/main.py' method, 'main_solver' for non-constrained problems, e.g. in the test files of the main directory, and 'main_solver_constrained' for constrained problems.

The input parameters are same for 'main_solver' and 'main_solver_constrained'. The difference is the definition of the problem function.

Following is the explaination of useful input parameters:

funct: problem function, callable, input n X d tensor with n samples of d variables in [-0.5, 0.5] and output n X m tensor for 1 objective and m - 1 constraints. 

dim: problem dimensionality, int

nMax: maximum simulation budget, default 3000

dataset_file: result pkl filename. We save the history x and y in the pkl file. default: './dataset_tSS_BO.pkl'

use_BO: bool, default True. If false, BO method is not used, only truncated subspace sampling is used.

k: int, default 100. The significance of the preceding k-iteration features will decay to 1% of the origin. 

init_x: optional, n_init X d tensor. Feed some given initial points. Default: None, initialization with random

init_y: optional, n_init X m tensor. Feed some given initial fitness. Default: None, initialization using funct.

sigma: initial deviation of adaptive matrix, default 0.2.

mu: the ratio of points for center update, default 0.5.

batch_size: batch size for each optimization iteration, default 20. A comment is to set the batch size proportional to log(dim)

n_training: local GP model training number, the more training sample used, the more time GP model used for training and prediction. Default min(dim * 2, 500)

n_candidates: number of random samples of truncated Gaussian distribution, default 200

Some other input parameters might not be helpful or should be deleted.

# Example
A simple script to run the tSS-BO method can be written as:

 from src.main import *
 
 import pickle
 
 def f(x): # define the objective function
 
 ...
 
 return y # numpy array

 dim = 100 # dimensionality for x
 
 k = 3 * dim # decaying factor for previous features
 
 filename = 'result.pkl' # result file

 main_solver(f, dim, bounds, sigma = 0.2, mu = 0.5, batch_size = 20, n_candidates = 200, n_resample = 10, nMax = 3000, k = k, dataset_file = filename)

# Experiments
The synthetic experiments in paper 'tSS-BO: Scalable Bayesian Optimization for Analog Circuit Sizing via Truncated Subspace Sampling' can be conducted through the scripts in the main directory named as 'test_XXX_my_all_DDD.py', where XXX gives the test function name and DDD gives the dimensionality.

The compared methods are from:
TuRBO

