import numpy as np
import torch
import math
from .util import *


class tSubspace:
    def __init__(self, 
                dim,
                bounds,
                W_prior = None,
                mean = None,
                start_f = None,
                gradient_prior = None,
                sigma = 0.2,
                mu = 0.5,
                c1 = None,
                c2 = None,
                allround_flag = False,
                greedy_flag = False,
                k = 100
                ):
        self.dim = dim
        if bounds is None:
            self.bounds = bounds
        else:
            self.bounds = torch.tensor(bounds)
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]
        if mean is None:
            self.mean = torch.rand(self.dim) * (self.ub - self.lb) + self.lb
        else:
            self.mean = torch.tensor(mean)

        self.mean = self.mean.view([self.dim, 1])

        if sigma is None:
            self.sigma = (torch.mean(self.ub) - torch.mean(self.lb)) / 5
        else:
            self.sigma = sigma

        self.mean_f = start_f

        self.mu = mu
        if gradient_prior is None:
            self.prior = torch.zeros([dim, 1])
        else:
            self.prior = torch.tensor(gradient_prior).view([dim, 1])

        if c1 is None:
            c1 = (1 - math.exp(math.log(0.01)/k)) / 2

        if c2 is None:
            c2 = (1 - math.exp(math.log(0.01)/k)) / 2

        self._c_j = c1
        self._c_p = c2
        self._c_W = 1 - self._c_j - self._c_p

        if W_prior is None:
            self._W = torch.eye(self.dim)
        else:
            self._W = torch.tensor(W_prior)

        self._chi_n = math.sqrt(self.dim) * (
            1.0 - (1.0 / (4.0 * self.dim)) + 1.0 / (21.0 * (self.dim**2))
        )

        self.allround_flag = allround_flag
        self.greedy_flag = greedy_flag
        self.value_sqrt = None
        self.Q = None

    def set_mean_f(self, mean_f):
        self.mean_f = torch.tensor(mean_f)

    def set_new_mean(self, mean, mean_f = None):
        self.mean = torch.tensor(mean)
        self.mean_f = mean_f

    def _get_prior_gradient(self, prior_x, prior_y, alpha = 0.01):
        assert self.mean_f is not None

        X_torch = torch.tensor(prior_x).view([self.dim, -1])
        Y_torch = torch.tensor(prior_y).view([-1, 1])

        Sk = X_torch - self.mean.view([self.dim, -1])

        Yk = Y_torch - self.mean_f

        J = torch.pinverse(Sk.mm(Sk.t()) + alpha * torch.eye(self.dim)).mm(Sk)
        J = J.mm(Yk)

        return J

    def desketch_gradient(self, X_torch, Y_torch):
        assert self.mean_f is not None

        Sk = X_torch - self.mean.view([self.dim, -1])

        Yk = Y_torch - self.mean_f

        Ak = Sk.mm(torch.pinverse(Sk.t().mm(Sk)))

        Jk = self.prior + Ak.mm(Yk - Sk.t().mm(self.prior))

        return Jk.view([self.dim, -1])

    def compute_pk(self, X_torch, Y_torch):
        Y_arg = torch.argsort(Y_torch.ravel())
        X_torch = X_torch[:, Y_arg]

        if self.greedy_flag:
            #greedy weights
            weights = torch.zeros_like(Y_arg)
            weights[0] = 1
            self.mu_num = 1
            self.mu_eff = 1
        else:

            #cmaes weights
            weights_prime = torch.tensor(
                [
                    math.log((X_torch.size(1) + 1) * self.mu) - math.log(i + 1)
                    for i in range(X_torch.size(1))
                ]
            )


            self.mu_num = math.floor(X_torch.size(1) * self.mu)
            self.mu_eff = (torch.sum(weights_prime[:self.mu_num]) ** 2) / torch.sum(weights_prime[:self.mu_num] ** 2)

            positive_sum = torch.sum(weights_prime[weights_prime > 0])
            negative_sum = torch.sum(torch.abs(weights_prime[weights_prime < 0]))
            weights = torch.where(
                weights_prime >= 0,
                1 / positive_sum * weights_prime,
                0.9 / negative_sum * weights_prime,
            )

        if self.allround_flag:
            #allround update
            X_delta = (X - self.mean.view([self.dim , -1])) / self.sigma
            p_mu = torch.sum(X_delta * weights.view([1, -1]), 1)
        else:
            #cmaes update
            X_mu = X_torch[:, :self.mu_num]
            X_mu_delta = (X_mu - self.mean.view([self.dim , -1])) / self.sigma
            p_mu = torch.sum(X_mu_delta * weights[:self.mu_num].view([1, -1]), 1)

        

        return p_mu.view([self.dim, -1])
            

    def update_subspace(self, new_x, new_y, new_mean_f = None, GP_model_list = None, mean_and_std = None):
        X_torch = torch.tensor(new_x).view([self.dim, -1])
        Y_torch = torch.tensor(new_y).view([-1, 1])

        if new_mean_f is not None:
            self.mean_f = new_mean_f
            
        Gk = None
            
        if GP_model_list is not None and mean_and_std is not None:
            Gk = sample_model_gradient(GP_model_list, mean_and_std[0], mean_and_std[1], self.mean, delta = 0.01).view([-1, 1])
            Gk = Gk / Gk.norm() * self._chi_n 


        Jk = self.desketch_gradient(X_torch, Y_torch)

        Pk = self.compute_pk(X_torch, Y_torch)

        self.prior = Jk

        Jk = Jk / Jk.norm() * self._chi_n

        self.mean = self.mean.ravel() + Pk.ravel() * self.sigma
        self.mean_f = None

        D, Q = self._eigen_decomposition()

        W_2 = Q.mm(torch.diag(1/D).mm(Q.t()))

        Pk_normalized_norm = W_2.mm(Pk).norm()
        #print('previous W eigenvalues, ')
        #print(D)
        print('pk norm, ', Pk.norm())
        print('normalized pk norm, ', Pk_normalized_norm / self._chi_n - 1)
        print('another normalized pk norm, ', Pk_normalized_norm * math.sqrt(self.mu_eff) / self._chi_n)

        #c = (self.mu_eff + 2) / (self.mu_eff + X_torch.size(1) + 5)
        c = (self.mu_eff + 2) / (self.mu_eff + self.dim + 5)
        #self.sigma = self.sigma * 0.96
        #self.sigma = self.sigma * math.exp( c / (1 + c) * (Pk_normalized_norm * math.sqrt(self.mu_eff)/ self._chi_n - 1))
        self.sigma = self.sigma * math.exp( c / (1 + c) * (Pk_normalized_norm / self._chi_n - 1))
        print('sigma, ', self.sigma)


        #Pk = Pk * self.mu_eff
        if Gk is None:
            self._W = self._c_W * self._W + self._c_j * Jk.mm(Jk.t()) + self._c_p * Pk.mm(Pk.t())
        else:
            self._W = self._c_W * self._W + self._c_j * Jk.mm(Jk.t()) * 3 / 5 + self._c_p * Pk.mm(Pk.t()) * 4 / 5 + self._c_j * Gk.mm(Gk.t()) * 3 / 5
        self.value_sqrt = None
        self.Q = None

    def _eigen_decomposition(self):
        if self.value_sqrt is not None and self.Q is not None:
            return self.value_sqrt, self.Q

        W = self._W/2 + self._W.t() / 2
        value, Q = torch.linalg.eigh(W)
        value_sqrt = torch.sqrt(torch.where(value > 1e-12, value, 1e-12))

        self._W = Q.mm(torch.diag(value_sqrt ** 2)).mm(Q.t())
        self.value_sqrt = value_sqrt
        self.Q = Q

        return value_sqrt, Q

    def sample_candidates(self, n_candidate = 100, n_resample=10):
        D, B = self._eigen_decomposition()
        x = torch.empty(size = torch.Size([self.dim, 0]))
        for i in range(n_resample):
            if x.size(1) >= n_candidate:
                break
            z = torch.randn(self.dim, n_candidate)
            y = B.mm(torch.diag(D)).mm(z)
            x_candidate = self.mean.view([self.dim, 1]) + y * self.sigma
            if self.bounds is None:
                x = x_candidate
            else:
                inbox = torch.all(x > self.lb.view([-1,1]), 0).multiply(
                        torch.all(x < self.ub.view([-1,1]), 0)
                    )
                if inbox.size(0):
                    x = torch.cat([x, x_candidate[:, inbox]], 1)

        if x.size(1) < n_candidate:
            n_sample = n_candidate - x.size(1)
            z = torch.randn(self.dim, n_sample)
            y = B.mm(torch.diag(D)).mm(z)
            x_candidate = self.mean.view([self.dim, 1]) + y * self.sigma
            x = torch.cat([x, x_candidate], 1)

        x = x.clip(min = self.lb.view([-1, 1]), max = self.ub.view([-1, 1]))
        x = x[:, :n_candidate]
        return x









