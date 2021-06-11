import torch
from collections import OrderedDict

class VariationalApprox(object):
    def __init__(self, model, num_mc_sample, init_covar_value):
        self.model = model
        self.num_mc_sample = num_mc_sample
        self.init_covar_value = init_covar_value

        self.mean = self.init_mean()
        self.covar = self.init_covariance()

    def init_mean(self):
        return OrderedDict([
            (name, param.clone().detach().requires_grad_(True)) for (name, param) in self.model.meta_named_parameters()
        ])

    def init_covariance(self):
        return OrderedDict([
            (name, param.new_full(param.size(), fill_value=self.init_covar_value).requires_grad_(True))
            for (name, param) in self.model.meta_named_parameters()
        ])

    def exp_covar(self, covar):
        return OrderedDict([(name, torch.exp(cov)) for name, cov in covar.items()])

    def detach_model_params(self):
        for param in self.model.meta_parameters():
            param.requires_grad = False

    def sample_params(self, n_sample=None, detach_mean_cov=False):
        params = OrderedDict()
        for (name, mean), cov in zip(self.mean.items(), self.exp_covar(self.covar).values()):
            if n_sample == 1:
                params_sample_size = [*mean.size()]
            elif n_sample is None:
                params_sample_size = [self.num_mc_sample, *mean.size()]
            else:
                params_sample_size = [n_sample, *mean.size()]

            params[name] = \
                (mean.detach() + cov.detach().sqrt()
                 * torch.randn(*params_sample_size, dtype=mean.dtype, device=mean.device)).requires_grad_(True) \
                    if detach_mean_cov \
                else mean + cov.sqrt() * torch.randn(*params_sample_size, dtype=mean.dtype, device=mean.device)

        return params

    def update_mean_cov(self):
        self.mean_old = OrderedDict([(name, mu.clone().detach()) for (name, mu) in self.mean.items()])
        self.covar_old = OrderedDict([(name, cov.clone().detach()) for (name, cov) in self.covar.items()])