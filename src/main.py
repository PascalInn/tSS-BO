from .tSS_BO import tSubspace
from .util import *

import numpy as np
import torch
import torch.multiprocessing as mp
import math
import time
import pickle
import os

mp.set_sharing_strategy('file_system')
strat_time = time.time()
def print_err(err):
    print(err)
    
def parallel_simulate(i, f, x, fX_pool, pid_set, output = 1):
    pid = os.getpid()
    if pid in pid_set:
        index = pid_set[pid]
    else:
        pid_set[pid] = None
        index = list(pid_set.keys()).index(pid)
        pid_set[pid] = index
    fX_pool[i] = [f(x, index, output)]

def main_solver(funct, dim, bounds, init_x = None, init_y = None,
                sigma = 0.2, mu = 0.5, c1 = None, c2 = None, allround_flag = False, greedy_flag = False,
                n_training = None, batch_size = 20, n_candidates = 200, n_resample = 10, nMax = 3000 , k = 100, dataset_file = './dataset_tTs_bo.pkl', use_BO = True, use_TS = False,
                calculate_model_gradient = True
                 ):
    t_funct = lambda x: torch.tensor(funct(x))
    t00 = time.time()
    if n_training is None:
        n_training = min(dim * 2, 500)

    subspace = tSubspace(dim, bounds, sigma = sigma, mu = mu, c1 = c1, c2 = c2, allround_flag = allround_flag, greedy_flag = greedy_flag, k = k)
    model_list = None
    m_n_s = None
    evaluation_hist = [torch.empty(size = torch.Size([0, dim])), torch.empty(size = torch.Size([0]))]
    t0 = time.time()
    if init_x is not None:
        x = torch.tensor(init_x)
        if init_y is None:
            init_y = t_funct(x)
            
        new_x = x[torch.argsort(init_y), :]
        evaluation_hist[0] = torch.cat([evaluation_hist[0], x], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], init_y.ravel()])
        weights_prime = torch.tensor(
            [
                math.log((x.size(0) + 1) * mu) - math.log(i + 1)
                for i in range(x.size(0))
            ]
        )


        mu_num = math.floor(x.size(0) * mu)
        mu_eff = (torch.sum(weights_prime[:mu_num]) ** 2) / torch.sum(weights_prime[:mu_num] ** 2)

        positive_sum = torch.sum(weights_prime[weights_prime > 0])
        negative_sum = torch.sum(torch.abs(weights_prime[weights_prime < 0]))
        weights = torch.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            0.9 / negative_sum * weights_prime,
        )

        X_mu = new_x[:mu_num,:]
        #X_mu_delta = (X_mu - self.mean.view([self.dim , -1])) / self.sigma
        mean_prior = torch.sum(X_mu * weights[:mu_num].view([-1, 1]), 0)

        f_mean_prior = t_funct(mean_prior)

        subspace.set_new_mean(mean_prior, f_mean_prior)

        J0 = subspace._get_prior_gradient(x.t(), init_y)
        subspace.prior = J0
        evaluation_hist[0] = torch.cat([evaluation_hist[0], mean_prior.view([1, -1])], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], f_mean_prior.ravel()])
    t1 = time.time()
    print('initialization time, ', (t1 - t0))
    g = 0
    while evaluation_hist[1].size(0) <= nMax:
        g = g + 1
        print('iteration ,', g)
        
        t0 = time.time()
        print('elasped time,', t0 - t00)
        X_candidates = subspace.sample_candidates(n_candidates, n_resample).t()

        t1 = time.time()
        print('candidate generation time, ', (t1 - t0))

        if evaluation_hist[0].size(0) and use_BO:
            X = evaluation_hist[0]
            Y = evaluation_hist[1]
            X_center = subspace.mean.ravel()

            if X.size(0) >= n_training:
                D, B = subspace._eigen_decomposition()
                X, Y = select_training_set(
                        X_center,
                        X,
                        Y,
                        B,
                        D,
                        n_training = n_training
                    )

            if use_TS:
                X_cand, model_list, m_n_s = select_candidate_TS_unconstrained(
                        X,
                        Y,
                        X_candidates,
                        batch_size = batch_size
                    )
            else:
                X_cand, model_list, m_n_s = select_candidate_EI_unconstrained(
                        X,
                        Y,
                        X_candidates,
                        batch_size = batch_size
                    )
        else:
            X_cand = X_candidates[:batch_size, :]
        t2 = time.time()
        print('candidate selection time, ', (t2 - t1))
        
        if not calculate_model_gradient:
            model_list = None
            m_n_s = None

        if subspace.mean_f is None:
            X_cand = torch.cat([subspace.mean.view([1, -1]), X_cand], 0)
            Y_cand = t_funct(X_cand)
            t3 = time.time()
            subspace.set_mean_f(Y_cand[0])
            subspace.update_subspace(X_cand[1:, :].t(), Y_cand[1:], GP_model_list = model_list, mean_and_std = m_n_s)
            t4 = time.time()

        else:
            Y_cand = t_funct(X_cand)
            t3 = time.time()
            subspace.update_subspace(X_cand.t(), Y_cand, GP_model_list = model_list, mean_and_std = m_n_s)
            t4 = time.time()
        print('simulation time, ', (t3 -t2))
        print('update subspace time, ', (t4 -t3))

        evaluation_hist[0] = torch.cat([evaluation_hist[0], X_cand], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], Y_cand.ravel()])
        print('best y', evaluation_hist[1].min())

        with open(dataset_file, 'wb') as f:
            pickle.dump(evaluation_hist[1], f)
            
def main_solver_constrained(funct, dim, bounds, init_x = None, init_y = None,
                sigma = 0.2, mu = 0.5, c1 = None, c2 = None, allround_flag = False, greedy_flag = False,
                n_training = None, batch_size = 20, n_candidates = 200, n_resample = 10, nMax = 3000 , k = 100, dataset_file = './dataset_tTs_bo.pkl', use_BO = True, use_TS = True, outdim = 4, calculate_model_gradient = False
                 ):
    t_funct = lambda x: torch.tensor(funct(x)).float()
    #funct returns [obj, cons1, cons2, ... ,] for each sample in one line

    if n_training is None:
        n_training = min(dim * 2, 500)

    subspace = tSubspace(dim, bounds, sigma = sigma, mu = mu, c1 = c1, c2 = c2, allround_flag = allround_flag, greedy_flag = greedy_flag, k = k)

    evaluation_hist = [torch.empty(size = torch.Size([0, dim])), torch.empty(size = torch.Size([0, outdim]))]
    t0 = time.time()
    if init_x is not None:
        x = torch.tensor(init_x)
        if init_y is None:
            init_y = t_funct(x)
        init_cv = init_y[:, 1:].clip(min = 0).sum(1)   
        new_x = x[torch.argsort(init_y[:, 0] + init_cv), :]
        evaluation_hist[0] = torch.cat([evaluation_hist[0], x], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], init_y], 0)
        weights_prime = torch.tensor(
            [
                math.log((x.size(0) + 1) * mu) - math.log(i + 1)
                for i in range(x.size(0))
            ]
        )


        mu_num = math.floor(x.size(0) * mu)
        mu_eff = (torch.sum(weights_prime[:mu_num]) ** 2) / torch.sum(weights_prime[:mu_num] ** 2)

        positive_sum = torch.sum(weights_prime[weights_prime > 0])
        negative_sum = torch.sum(torch.abs(weights_prime[weights_prime < 0]))

        weights = torch.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            0.9 / negative_sum * weights_prime,
        )

        X_mu = new_x[:mu_num,:]
        #X_mu_delta = (X_mu - self.mean.view([self.dim , -1])) / self.sigma
        
        mean_prior = torch.sum(X_mu * weights[:mu_num].view([-1, 1]), 0)

        f_mean_prior = torch.tensor(funct(mean_prior)).float().ravel()
        cv_prior = f_mean_prior[1:].clip(min = 0).sum()


        subspace.set_new_mean(mean_prior, f_mean_prior[0] + cv_prior)

        J0 = subspace._get_prior_gradient(x.t(), init_y[:, 0] + init_cv)

        #J0 = subspace._get_prior_gradient(x.t(), init_y.sum(1))
        subspace.prior = J0
        evaluation_hist[0] = torch.cat([evaluation_hist[0], mean_prior.view([1, -1])], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], f_mean_prior.view([1, -1])], 0)
    t1 = time.time()
    print('initialization time, ', (t1 - t0))
    g = 0
    while evaluation_hist[0].size(0) <= nMax:
        g = g + 1
        print('iteration ,', g)
        t0 = time.time()
        X_candidates = subspace.sample_candidates(n_candidates, n_resample).t()

        t1 = time.time()
        print('candidate generation time, ', (t1 - t0))

        if evaluation_hist[0].size(0) and use_BO:
            X = evaluation_hist[0]
            Y = evaluation_hist[1]
            X_center = subspace.mean.ravel()

            if X.size(0) >= n_training:
                D, B = subspace._eigen_decomposition()
                X, Y = select_training_set(
                        X_center,
                        X,
                        Y,
                        B,
                        D,
                        n_training = n_training
                    )

            if use_TS:
                X_cand, model_list, m_n_s = select_candidate_TS_constrained(
                        X,
                        Y,
                        X_candidates,
                        batch_size = batch_size
                    )
            else:
                X_cand, model_list, m_n_s = select_candidate_EI_constrained(
                        X,
                        Y,
                        X_candidates,
                        batch_size = batch_size
                    )
        else:
            X_cand = X_candidates[:batch_size, :]
        t2 = time.time()
        print('candidate selection time, ', (t2 - t1))
        if not calculate_model_gradient:
            model_list, m_n_s = None, None

        if subspace.mean_f is None:
            Y_cand = t_funct(X_cand)
            t3 = time.time()
            CV = Y_cand[:,1:].clip(min = 0).sum(1)
            subspace.set_mean_f(Y_cand[0, 0] + CV[0])
            subspace.update_subspace(X_cand[1:, :].t(), Y_cand[1:, 0] + CV[1:], GP_model_list = model_list, mean_and_std = m_n_s)
            t4 = time.time()


        else:
            Y_cand = t_funct(X_cand)
            CV = Y_cand[:,1:].clip(min = 0).sum(1)
            t3 = time.time()
            subspace.update_subspace(X_cand.t(), Y_cand[:, 0] + CV, GP_model_list = model_list, mean_and_std = m_n_s)
            t4 = time.time()
        print('simulation time, ', (t3 -t2))
        print('update subspace time, ', (t4 -t3))

        evaluation_hist[0] = torch.cat([evaluation_hist[0], X_cand], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], Y_cand], 0)
        CV_all = evaluation_hist[1][:, 1:].clip(min = 0).sum(1)
        print('best y', torch.min(CV_all+evaluation_hist[1][:, 0]))

        with open(dataset_file, 'wb') as f:
            pickle.dump(evaluation_hist, f)





