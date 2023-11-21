import torch
import numpy as np
import gpytorch
import botorch

from .GP_torch import *
import math

import gpytorch.settings as gpts

from contextlib import ExitStack

from scipy.stats import norm

from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.analytic import LogExpectedImprovement
#from botorch.acquisition.monte_carlo import qLogExpectedImprovement

from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling

def select_training_set(
    X_center,
    X,
    Y,
    B=None,
    D=None,
    n_training = 100,
    use_C = True,
    ):
    X_center = X_center.view([1, -1])
    X_diff = X - X_center

    if use_C:
        #print(B)
        #print(D)
        C_2 = B.mm(torch.diag(1 / D)).mm(B.t())
        X_diff = X_diff.mm(C_2.t())

    X_diff_norm = torch.linalg.norm(X_diff, axis = 1)
    sortargs = torch.argsort(X_diff_norm)

    return X[sortargs[:n_training], ...], Y[sortargs[:n_training], ...]
    
def sample_model_gradient(
    model_list,
    y_mean,
    y_std,
    X,
    delta = 0.01,     
    ):
    size_list = len(model_list)
    n_dim = X.size(-1)
    
    y_mean = y_mean.view(-1)
    y_std = y_std.view(-1)
    
    X_list = torch.tile(X.view(1,-1), torch.Size([n_dim, 1]))
    X_list = torch.cat([X_list + delta * torch.eye(n_dim), X_list - delta * torch.eye(n_dim)], 0)
    
    Y_sample = torch.empty(torch.Size([0, 2 * n_dim]))
    with torch.no_grad():
        for i in range(size_list):
            mvn = model_list[i](X_list)
            Y_sample = torch.cat([Y_sample, model_list[i].likelihood(mvn).sample(torch.Size([1])) * (y_std[i] + 1e-6) + y_mean[i]], 0)
    if size_list > 1:
        Y_sample[1:, :] = Y_sample[1:, :].clip(min = 0.0)
    
    Y_sample_plus = Y_sample[:, :n_dim].sum(0)
    Y_sample_minus = Y_sample[:, n_dim:].sum(0)
    g_sample = (Y_sample_plus - Y_sample_minus) / delta / 2
    
    return g_sample

def select_candidate_EI_unconstrained(
    X,
    Y,
    X_candidate,
    batch_size,
    #n_candidates,
    sampler = 'cholesky',
    use_keops=False,
    device = 'cpu',
    noise_flag = True,  
    ):
    X_torch = torch.tensor(X).to(device = device)
    X_candidate_torch = torch.tensor(X_candidate).to(device = device)
    Y_torch = torch.tensor(Y).to(device = device)

    assert torch.all(torch.isfinite(Y_torch))
    y_mean, y_std = Y_torch.mean(), Y_torch.std()

    train_y = (Y_torch - Y_torch.mean()) / ( 1e-6 + Y_torch.std())
    #print(train_y)
    #print(X_torch.shape)
    #print(train_y.shape)
    #print(train_y.ravel().shape)
    

    #model = GP({'train_x': X_torch, 'train_y': train_y, 'train_sigma':None}, kernel = 'MAT52')
    #model.fit()
    #base_kernel = gpytorch.kernels.MaternKernel(lengthscale_constraint = Interval(0.005, math.sqrt(X_torch.size(-1))),
    #                                                    nu = 2.5, ard_num_dims = X_torch.size(-1))
    #covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_constraint = Interval(0.05, 20))

    #model = SingleTaskGP(X_torch, train_y.view([-1,1]), 
    #                     likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = Interval(5e-4, 0.2)),
    #                     covar_module=covar_module)
    
    #mll = ExactMarginalLogLikelihood(model.likelihood, model)

    #with gpts.cholesky_max_tries(29):
    #    fit_gpytorch_mll(mll)

    model = GP({'train_x': X_torch, 'train_y': train_y, 'train_sigma':None}, kernel = 'MAT52')
    model.fit()

    mvn = model(X_candidate_torch)

    mean = mvn.mean.detach().numpy()
    var = mvn.variance.detach().numpy()
    best_f = Y_torch.min().detach().numpy()

    stdd = np.sqrt(var + 1e-12)
    normed = - (mean - best_f) / stdd
    EI = stdd * (normed * norm.cdf(normed) + norm.pdf(normed))
    log_EI = np.log(np.maximum(0.000001, EI))
    tmp = np.minimum(-40, normed) ** 2
    log_EI_approx = np.log(stdd) - tmp/2 - np.log(tmp-1)
    log_EI_approx = log_EI * (normed > -40) + log_EI_approx * (normed <= -40)

    #ei_acq = LogExpectedImprovement(
    #    model = model,
    #    best_f = Y_torch.min(),
    #    maximize = False,
    #)

    #ei_value = ei_acq(X_candidate_torch.view([X_candidate_torch.size(0), 1, X_candidate_torch.size(1)])).ravel()
    ei_argsort = np.argsort(-1 * log_EI_approx)
    return X_candidate_torch[ei_argsort[:batch_size], :], [model], [y_mean, y_std]

def select_candidate_TS_unconstrained(
    X,
    Y,
    X_candidate,
    batch_size,
    #n_candidates,
    sampler = 'lanczos',
    use_keops=False,
    device = 'cpu',
    noise_flag = True,
    ):
    
    X_torch = torch.tensor(X).to(device = device)
    X_candidate_torch = torch.tensor(X_candidate).to(device = device)
    Y_torch = torch.tensor(Y).to(device = device)

    assert torch.all(torch.isfinite(Y_torch))
    y_mean, y_std = Y_torch.mean(), Y_torch.std()

    train_y = (Y_torch - Y_torch.mean()) / ( 1e-5 + Y_torch.std())
    #print(train_y)
    #print(X_torch.shape)
    #print(train_y.shape)
    #print(train_y.ravel().shape)
    

    model = GP({'train_x': X_torch, 'train_y': train_y, 'train_sigma':None}, kernel = 'MAT52')
    model.fit()

    with ExitStack() as es:
        if sampler == "cholesky":
            es.enter_context(gpts.max_cholesky_size(float("inf")))
        elif sampler == "ciq":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(
                gpts.minres_tolerance(2e-3)
            )  # Controls accuracy and runtime
            es.enter_context(gpts.num_contour_quadrature(15))
        elif sampler == "lanczos":
            es.enter_context(
                gpts.fast_computations(
                    covar_root_decomposition=True, log_prob=True, solves=True
                )
            )
            es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(False))
        elif sampler == "rff":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

    with torch.no_grad():
        #thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        #X_next = thompson_sampling(X_candidate_torch, num_samples=batch_size, observation_noise = noise_flag)
        y_cand_dist = model(X_candidate_torch)
        y_cand = model.likelihood(y_cand_dist).sample(torch.Size([batch_size])).t().cpu().detach()

    X_cand_select = torch.ones((batch_size, X_candidate_torch.size(-1)))
    for i in range(batch_size):
        # Pick the best point and make sure we never pick it again
        indbest = torch.argmin(y_cand[:, i])
        X_cand_select[i, :] = X_candidate_torch[indbest, :]
        y_cand[indbest, :] = torch.inf

    return X_cand_select, [model], [y_mean, y_std]
    
def select_candidate_EI_constrained(
    X,
    Y,
    X_candidate,
    batch_size,
    #n_candidates,
    sampler = 'cholesky',
    use_keops=False,
    device = 'cpu',
    noise_flag = True,  
    ):
    X_torch = torch.tensor(X).to(device = device)
    X_candidate_torch = torch.tensor(X_candidate).to(device = device)
    Y_torch = torch.tensor(Y).to(device = device)
    #print(Y_torch)
    assert torch.all(torch.isfinite(Y_torch))
    #print(Y_torch)
    if torch.any(torch.all(Y_torch[:, 1:] <= 0, 1)):
        y_feasi = Y_torch[torch.all(Y_torch[:, 1:] <= 0, 1), :]
        y_best = y_feasi[torch.argmin(y_feasi[:, 0]), :]
    else:
        y_best = Y_torch[np.argmin(Y_torch[:, 1:].clip(min = 0.0).sum(1)), :]
    
    y_mean, y_std = Y_torch.mean(0), Y_torch.std(0)    
    
    Y_torch = (Y_torch - y_mean ) / ( 1e-6 + y_std)
    #train_y = (Y_torch[:, 0] - Y_torch[:, 0].mean()) / ( 1e-6 + Y_torch[:, 0].std())
    #cons_y = (Y_torch[:, 1:] - y_mean[1:] ) / (1e-6 + y_std[1:])
    #print(train_y)
    #print(X_torch.shape)
    #print(train_y.shape)
    #print(train_y.ravel().shape)
    

    #model = GP({'train_x': X_torch, 'train_y': train_y, 'train_sigma':None}, kernel = 'MAT52')
    #model.fit()
    #base_kernel = gpytorch.kernels.MaternKernel(lengthscale_constraint = Interval(0.005, math.sqrt(X_torch.size(-1))),
    #                                                    nu = 2.5, ard_num_dims = X_torch.size(-1))
    #covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_constraint = Interval(0.05, 20))

    #model = SingleTaskGP(X_torch, train_y.view([-1,1]), 
    #                     likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = Interval(5e-4, 0.2)),
    #                     covar_module=covar_module)
    
    #mll = ExactMarginalLogLikelihood(model.likelihood, model)

    #with gpts.cholesky_max_tries(29):
    #    fit_gpytorch_mll(mll)
    models = []
    for i in range(Y_torch.shape[1]):
        model = GP({'train_x': X_torch, 'train_y': Y_torch[:, i:i+1], 'train_sigma':None}, kernel = 'MAT52')
        model.fit()
        models.append(model)
    
    mvns = []
    with torch.no_grad():
        for i in range(Y_torch.shape[1]):
            
            mvns.append(models[i](X_candidate_torch))
    
    if torch.all(y_best[1:] <= 0):
        #EI approx
        mean = mvns[0].mean.detach().numpy()
        var = mvns[0].variance.detach().numpy()
        best_f = (y_best[0] - y_mean[0] )/(y_std[0] + 1e-6)

        stdd = np.sqrt(var + 1e-12)
        normed = - (mean - best_f.item()) / stdd
        EI = stdd * (normed * norm.cdf(normed) + norm.pdf(normed))
        log_EI = np.log(np.maximum(0.000001, EI))
        tmp = np.minimum(-40, normed) ** 2
        log_EI_approx = np.log(stdd) - tmp/2 - np.log(tmp-1)
        log_EI_approx = log_EI * (normed > -40) + log_EI_approx * (normed <= -40)
    else:
        log_EI_approx = np.zeros(X_candidate_torch.size(0))
    
    #calc log PI
    log_PI = np.zeros(X_candidate_torch.size(0))
    for ii in range(1, Y_torch.shape[1]):
        mean = mvns[ii].mean.detach().numpy() * (y_std[ii].detach().numpy() + 1e-6)+ y_mean[ii].detach().numpy()
        var = mvns[ii].variance.detach().numpy() * (y_std[ii].detach().numpy() + 1e-6)**2
        ps = np.sqrt(var + 1e-12)
        log_PI = log_PI + np.log(norm.cdf(-mean / ps))
    
    log_EI_approx = log_PI + log_EI_approx
        
    

    #ei_acq = LogExpectedImprovement(
    #    model = model,
    #    best_f = Y_torch.min(),
    #    maximize = False,
    #)

    #ei_value = ei_acq(X_candidate_torch.view([X_candidate_torch.size(0), 1, X_candidate_torch.size(1)])).ravel()
    ei_argsort = np.argsort(-1 * log_EI_approx)
    return X_candidate_torch[ei_argsort[:batch_size], :], models, [y_mean, y_std]