import os
cpu_lim = 2
os.environ['OMP_NUM_THREADS']=str(cpu_lim)
os.environ['OPENBLAS_NUM_THREADS']=str(cpu_lim)
os.environ['MKL_NUM_THREADS']=str(cpu_lim)
os.environ['VECLIB_MAXIMUM_THREADS']=str(cpu_lim)
os.environ['NUMEXPR_NUM_THREADS']=str(cpu_lim)
import numpy as np
import torch
import gpytorch
import botorch
from src.main import *

dim = 100
funct = botorch.test_functions.synthetic.Rosenbrock(dim = dim)
num = 100

k = 300

fa = lambda x: funct.evaluate_true(x * 5).detach().numpy()

bounds = torch.tensor([[-0.5, 0.5]] * dim)
lb = bounds[:, 0]
ub = bounds[:, 1]

repeat_num = 10


for i in range(repeat_num):
    print('start repeat, ', i)
    dataset_file = './result/dataset_%d_tSS_BO_rosenbrock_%d.pkl'%(i, dim)
    main_solver(fa, dim, bounds, init_x = None, init_y = None,
                    sigma = 0.2, mu = 0.5, c1 = None, c2 = None, allround_flag = False, greedy_flag = False,
                    n_training = None, batch_size = 20, n_candidates = 200, n_resample = 10, nMax = 3000 , k = k, dataset_file = dataset_file, use_TS = False
                     )

#run with no prior
for i in range(repeat_num):
    print('start repeat, ', i)
    dataset_file = './result/dataset_%d_tSS_random_rosenbrock0816_%d.pkl'%(i, dim)
    main_solver(fa, dim, bounds, init_x = None, init_y = None,
                    sigma = 0.2, mu = 0.5, c1 = None, c2 = None, allround_flag = False, greedy_flag = False,
                    n_training = None, batch_size = 20, n_candidates = 200, n_resample = 10, nMax = 3000 , k = k, dataset_file = dataset_file, use_BO = False
                     )

