import gpytorch
import torch
import torch.optim as optim
import math
from gpytorch.constraints.constraints import Interval
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class NARGPKernel(gpytorch.kernels.Kernel):
    def __init__(self, n, kernel = 'RBF'):
        super(NARGPKernel, self).__init__()
        self.n = n
        if kernel == 'RBF':
            self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.n-1, active_dims = torch.arange(self.n- 1))
            self.rbf2 = gpytorch.kernels.RBFKernel(ard_num_dims=self.n-1, active_dims = torch.arange(self.n - 1))
            self.rbf3 = gpytorch.kernels.RBFKernel(active_dims = torch.tensor([self.n - 1]))
            self.mykernel = gpytorch.kernels.ScaleKernel(self.rbf2 * self.rbf3) + gpytorch.kernels.ScaleKernel(self.base_kernel)
        elif kernel == 'MAT52':
            self.base_kernel = gpytorch.kernels.MaternKernel(nu = 2.5, ard_num_dims=self.n-1, active_dims = torch.arange(self.n- 1))
            self.rbf2 = gpytorch.kernels.MaternKernel(nu = 2.5, ard_num_dims=self.n-1, active_dims = torch.arange(self.n - 1))
            self.rbf3 = gpytorch.kernels.MaternKernel(nu = 2.5, active_dims = torch.tensor([self.n - 1]))
            self.mykernel = gpytorch.kernels.ScaleKernel(self.rbf2 * self.rbf3) + gpytorch.kernels.ScaleKernel(self.base_kernel)
                    

    def forward(self, x1, x2, **params):
        return self.mykernel(x1, x2)


class GP(ExactGP, GPyTorchModel):
    def __init__(self, dataset, kernel = 'RBF', inner_kernel = 'RBF', k = 1):
        self.x_train = dataset['train_x']
        self.y_train = dataset['train_y'].view([-1])
        self.n_train = self.x_train.size(0)
        self.n_dim = self.x_train.size(1)
        #self.num_outputs = 1
        self.sigma_train = dataset['train_sigma']
        if self.sigma_train is not None:
            super(GP, self).__init__(self.x_train, self.y_train, gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self.sigma_train))
        else:
            super(GP, self).__init__(self.x_train, self.y_train, gpytorch.likelihoods.GaussianLikelihood(noise_constraint = Interval(5e-4, 0.2)))
        #self.num_outputs = 1
        if kernel == 'RBF':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims = self.n_dim)
        elif kernel == 'MAT52':
            base_kernel = gpytorch.kernels.MaternKernel(lengthscale_constraint = Interval(0.005, math.sqrt(self.n_dim)),
                                                        nu = 2.5, ard_num_dims = self.n_dim)

        self.mean_module = gpytorch.means.ConstantMean()
        if k == 1:
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_constraint = Interval(0.05, 20))
        else:
            self.covar_module = NARGPKernel(self.n_dim, kernel = inner_kernel)

    def fit(self):
        self.train()
        self.likelihood.train()
        
        optimizer = optim.Adam(self.parameters(), lr = 0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        num_epochs = 100
        for _ in range(num_epochs):
            optimizer.zero_grad()
            output = self(self.x_train)
            loss = -mll(output, self.y_train)

            loss.backward()

            optimizer.step()

        self.eval()
        self.likelihood.eval()

    def forward(self, x):

        mean_x = self.mean_module(x)
        #print(mean_x.detach())
        
        covar_x = self.covar_module(x)
        #print(covar_x.evaluate())
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist
        
    def predict(self, x, full_cov = 0):
        #with torch.no_grad():
        pred = self(x)
        if full_cov:
            return pred.mean.detach(), pred.covariance_matrix.detach()
        else:
            return pred.mean.detach(), pred.variance.detach()
    
if __name__=='__main__':
    x = torch.rand(100, 10)
    y = torch.sum(x ** 2, 1) + x[:, 0]
    sigma = x[:, 0] ** 2 / 100

    dataset = {'train_x': x, 'train_y': y, 'train_sigma':sigma}

    gpmodel = GP(dataset)
    gpmodel.fit()

    gpmodel2 = GP(dataset, k = 0)
    gpmodel2.fit()
    
    for i in range(9, 10):
        test_x = torch.rand(i, 10)

        #print('testing point,', test_x)
        with torch.no_grad():
            pred_y, cov_y = gpmodel.predict(test_x, 0)
            #pred_y2, cov_y2 = gpmodel.predict(test_x, 1)

            print('predict y,', pred_y)
            print('predict sigma,', cov_y)

            pred = gpmodel(test_x)
            print('forward mean,', pred.mean.detach())
            print('forward variance,', pred.variance.detach())
            print('forward covariance,', pred.covariance_matrix.detach())
