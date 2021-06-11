import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import \
    MetaLinear, MetaConv1d, MetaConv2d, MetaConv3d, MetaBatchNorm1d, MetaBatchNorm2d, MetaBatchNorm3d

from tqdm import tqdm
from collections import OrderedDict

from train.inner import inner_maml
from train.util import concat_param
from curvtorch.util_curv import iter_named_modules
from curvtorch.backward_context import KroneckerContext


class LaplaceApprox(object):
    def __init__(self, model, is_lapl_list, max_lapl_list_len, nll_supp_wrt_metaparam, hessian_xterm, kfac_init_mult,
                 upd_scale, device):
        self.model = model
        self.is_lapl_list = is_lapl_list
        self.max_lapl_list_len = max_lapl_list_len
        self.nll_supp_wrt_metaparam = nll_supp_wrt_metaparam
        self.hessian_xterm = hessian_xterm
        self.kfac_init_mult = [kfac_init_mult] * len(OrderedDict(iter_named_modules(self.model))) \
            if not isinstance(kfac_init_mult, list) \
            else kfac_init_mult
        self.upd_scale = upd_scale
        self.device = device

        self.mean = self.init_mean()
        # this is the nll_query hessian wrt to task-adapted parameters
        self.tpar_act_cov, self.tpar_grad_cov, self.tpar_bn_fisher \
            = self.init_kfac_blockdiag_fisher(kfac_init_mult=self.kfac_init_mult)
        # this is the nll_supp hessian wrt to meta-params
        if nll_supp_wrt_metaparam is True:
            self.mpar_act_cov, self.mpar_grad_cov, self.mpar_bn_fisher \
                = self.init_kfac_blockdiag_fisher(kfac_init_mult=self.kfac_init_mult)

    def init_mean(self):
        mean = [OrderedDict([
            (name, torch.zeros_like(param, requires_grad=False)) for name, param in self.model.meta_named_parameters()
        ])]
        return mean

    def init_kfac_blockdiag_fisher(self, kfac_init_mult):
        act_cov = OrderedDict()
        grad_cov = OrderedDict()
        bn_fisher = OrderedDict()

        for (name, module), mult in zip(iter_named_modules(self.model), kfac_init_mult):
            if isinstance(module, (MetaConv1d, MetaConv2d, MetaConv3d, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                act_cov[name] = mult * torch.ones(
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1] + int(module.bias is not None),
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1] + int(module.bias is not None),
                    dtype=module.weight.dtype,
                    device=self.device,
                    requires_grad=False)
                grad_cov[name] = torch.ones(
                    module.out_channels, module.out_channels,
                    dtype=module.weight.dtype,
                    device=self.device,
                    requires_grad=False
                )
            elif isinstance(module, (MetaLinear, nn.Linear)):
                act_cov[name] = mult * torch.ones(
                    module.in_features + int(module.bias is not None),
                    module.in_features + int(module.bias is not None),
                    dtype=module.weight.dtype,
                    device=self.device,
                    requires_grad=False)
                grad_cov[name] = torch.ones(
                    module.out_features, module.out_features,
                    dtype=module.weight.dtype,
                    device=self.device,
                    requires_grad=False
                )
            elif isinstance(module, (MetaBatchNorm1d, MetaBatchNorm2d, nn.BatchNorm1d, nn.BatchNorm2d)):
                n_ch = module.num_features
                bn_fisher[name] = mult * torch.ones(
                    n_ch, 2, 2, dtype=module.weight.dtype, device=self.device, requires_grad=False
                )
            else:
                raise NotImplementedError
        return [act_cov], [grad_cov], [bn_fisher]

    def init_kfac_blockdiag_fisher_denom(self):
        act_op_denom = OrderedDict([
            (name, torch.tensor(0., device=self.device)) for name, module in iter_named_modules(self.model)
            if isinstance(module, (MetaLinear, MetaConv1d, MetaConv2d, MetaConv3d,
                                   nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        ])
        grad_op_denom = OrderedDict([
            (name, torch.tensor(0., device=self.device)) for name, module in iter_named_modules(self.model)
            if isinstance(module, (MetaLinear, MetaConv1d, MetaConv2d, MetaConv3d,
                                   nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        ])
        bn_op_denom = OrderedDict([
            (name, torch.tensor(0., device=self.device)) for name, module in iter_named_modules(self.model)
            if isinstance(module, (MetaBatchNorm1d, MetaBatchNorm2d, MetaBatchNorm3d,
                                   nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        ])
        return act_op_denom, grad_op_denom, bn_op_denom

    @staticmethod
    def gaussian_logprob_conv(param, name, module, mean, act_cov, grad_cov):
        # for single task, so mean etc is an element in OrderedDict(), return (x - mean)^T cov (x - mean) without * -1/2
        mean_permuted = mean[name + '.weight'].permute(0, 2, 3, 1)
        mean_flat = mean_permuted.reshape(mean_permuted.size(0), -1)
        if param is None:
            weight_flat = module.weight.permute(0, 2, 3, 1).reshape(module.out_channels, -1)
            param_mean_diff = concat_param(weight_flat, module.bias) - concat_param(mean_flat, mean[name + '.bias']) \
                if module.bias is not None else weight_flat - mean_flat
        else:
            weight_flat = param[name + '.weight'].permute(0, 2, 3, 1).reshape(module.out_channels, -1)
            param_mean_diff \
                = concat_param(weight_flat, param[name + '.bias']) - concat_param(mean_flat, mean[name + '.bias']) \
                if module.bias is not None else weight_flat - mean_flat
        cov_diff_prod = torch.matmul(torch.matmul(grad_cov[name], param_mean_diff), act_cov[name].t())
        diff_cov_diff_prod = torch.dot(param_mean_diff.t().reshape(-1), cov_diff_prod.t().reshape(-1))
        return diff_cov_diff_prod

    @staticmethod
    def gaussian_logprob_linear(param, name, module, mean, act_cov, grad_cov):
        if param is None:
            param_mean_diff \
                = concat_param(module.weight, module.bias) \
                  - concat_param(mean[name + '.weight'], mean[name + '.bias']) \
                if module.bias is not None \
                else module.weight - mean[name + '.weight']
        else:
            param_mean_diff \
                = concat_param(param[name + '.weight'], param[name + '.bias']) \
                  - concat_param(mean[name + '.weight'], mean[name + '.bias']) \
                if module.bias is not None \
                else param[name + '.weight'] - mean[name + '.weight']
        cov_diff_prod = torch.matmul(torch.matmul(grad_cov[name], param_mean_diff), act_cov[name].t())
        diff_cov_diff_prod = torch.dot(param_mean_diff.t().reshape(-1), cov_diff_prod.t().reshape(-1))
        return diff_cov_diff_prod

    @staticmethod
    def gaussian_logprob_unitbn(param, name, module, mean, fisher):
        param_mean_diff = torch.stack([module.weight, module.bias], dim=-1) \
                          - torch.stack([mean[name + '.weight'], mean[name + '.bias']], dim=-1) if param is None \
            else torch.stack([param[name + '.weight'], mean[name + '.bias']], dim=-1) \
                 - torch.stack([mean[name + '.weight'], mean[name + '.bias']], dim=-1)
        diff_cov_diff_prod = torch.matmul(
            torch.matmul(param_mean_diff.unsqueeze(-2), fisher[name]), param_mean_diff.unsqueeze(-1)
        )
        return diff_cov_diff_prod.sum()

    def laplace_approx(self, param=None):
        logprob_task = []
        upd_scale_ls = [1.] + self.upd_scale[:(len(self.mean) - 1)] if isinstance(self.upd_scale, list) \
            else [1.] + [self.upd_scale] * (len(self.mean) - 1)

        upd_scale_list = [] + upd_scale_ls
        means = [] + self.mean
        act_covs = [] + self.tpar_act_cov
        grad_covs = [] + self.tpar_grad_cov
        bn_fishers = [] + self.tpar_bn_fisher

        if self.nll_supp_wrt_metaparam:
            upd_scale_list += upd_scale_ls
            means += self.mean
            act_covs += self.mpar_act_cov
            grad_covs += self.mpar_grad_cov
            bn_fishers += self.mpar_bn_fisher

        # consider hessian cross terms (first multiply the kfac-s, then append them to the list)
        if self.hessian_xterm is not None:
            # check if hessian_xterm only contains ("tpar_mpar", "mpar_tpar", "mpar_tpar_mpar")
            if sum(xterm not in ("tpar_mpar", "mpar_tpar", "mpar_tpar_mpar") for xterm in self.hessian_xterm) != 0:
                raise ValueError(
                    '''Each element of "hessian_xterm" must be one of ("tpar_mpar", "mpar_tpar", "mpar_tpar_mpar").''')

            if 'tpar_mpar' in self.hessian_xterm:
                tpar_mpar_act_cov, tpar_mpar_grad_cov, tpar_mpar_bn_fisher = [], [], []

                for tpar_act_cov, tpar_grad_cov, tpar_bn_fisher, mpar_act_cov, mpar_grad_cov, mpar_bn_fisher in \
                        zip(self.tpar_act_cov, self.tpar_grad_cov, self.tpar_bn_fisher,
                            self.mpar_act_cov, self.mpar_grad_cov, self.mpar_bn_fisher):
                    tpar_mpar_act_cov.append(OrderedDict([
                        (name, - torch.matmul(tpar_a_cov, mpar_a_cov.t()))
                        for (name, tpar_a_cov), mpar_a_cov in zip(tpar_act_cov.items(), mpar_act_cov.values())
                    ]))
                    tpar_mpar_grad_cov.append(OrderedDict([
                        (name, torch.matmul(tpar_g_cov, mpar_g_cov.t()))
                        for (name, tpar_g_cov), mpar_g_cov in zip(tpar_grad_cov.items(), mpar_grad_cov.values())
                    ]))
                    tpar_mpar_bn_fisher.append(OrderedDict([
                        (name, - torch.matmul(tpar_bn_fsh, mpar_bn_fsh.transpose(1, 2)))
                        for (name, tpar_bn_fsh), mpar_bn_fsh in zip(tpar_bn_fisher.items(), mpar_bn_fisher.values())
                    ]))

                upd_scale_list += upd_scale_ls
                means += self.mean
                act_covs += tpar_mpar_act_cov
                grad_covs += tpar_mpar_grad_cov
                bn_fishers += tpar_mpar_bn_fisher

            if 'mpar_tpar' in self.hessian_xterm:
                mpar_tpar_act_cov, mpar_tpar_grad_cov, mpar_tpar_bn_fisher = [], [], []

                for tpar_act_cov, tpar_grad_cov, tpar_bn_fisher, mpar_act_cov, mpar_grad_cov, mpar_bn_fisher in \
                        zip(self.tpar_act_cov, self.tpar_grad_cov, self.tpar_bn_fisher,
                            self.mpar_act_cov, self.mpar_grad_cov, self.mpar_bn_fisher):
                    mpar_tpar_act_cov.append(OrderedDict([
                        (name, - torch.matmul(mpar_a_cov, tpar_a_cov))
                        for (name, tpar_a_cov), mpar_a_cov in zip(tpar_act_cov.items(), mpar_act_cov.values())
                    ]))
                    mpar_tpar_grad_cov.append(OrderedDict([
                        (name, torch.matmul(mpar_g_cov, tpar_g_cov))
                        for (name, tpar_g_cov), mpar_g_cov in zip(tpar_grad_cov.items(), mpar_grad_cov.values())
                    ]))
                    mpar_tpar_bn_fisher.append(OrderedDict([
                        (name, - torch.matmul(mpar_bn_fsh, tpar_bn_fsh))
                        for (name, tpar_bn_fsh), mpar_bn_fsh in zip(tpar_bn_fisher.items(), mpar_bn_fisher.values())
                    ]))

                upd_scale_list += upd_scale_ls
                means += self.mean
                act_covs += mpar_tpar_act_cov
                grad_covs += mpar_tpar_grad_cov
                bn_fishers += mpar_tpar_bn_fisher

            if 'mpar_tpar_mpar' in self.hessian_xterm:

                mpar_tpar_mpar_act_cov, mpar_tpar_mpar_grad_cov, mpar_tpar_mpar_bn_fisher = [], [], []

                for tpar_act_cov, tpar_grad_cov, tpar_bn_fisher, mpar_act_cov, mpar_grad_cov, mpar_bn_fisher in \
                        zip(self.tpar_act_cov, self.tpar_grad_cov, self.tpar_bn_fisher,
                            self.mpar_act_cov, self.mpar_grad_cov, self.mpar_bn_fisher):

                    mpar_tpar_mpar_act_cov.append(OrderedDict([
                        (name, torch.matmul(torch.matmul(mpar_a_cov, tpar_a_cov), mpar_a_cov.t()))
                        for (name, tpar_a_cov), mpar_a_cov in zip(tpar_act_cov.items(), mpar_act_cov.values())
                    ]))
                    mpar_tpar_mpar_grad_cov.append(OrderedDict([
                        (name, torch.matmul(torch.matmul(mpar_g_cov, tpar_g_cov), mpar_g_cov.t()))
                        for (name, tpar_g_cov), mpar_g_cov in zip(tpar_grad_cov.items(), mpar_grad_cov.values())
                    ]))
                    mpar_tpar_mpar_bn_fisher.append(OrderedDict([
                        (name, torch.matmul(torch.matmul(mpar_bn_fs, tpar_bn_fs), mpar_bn_fs.transpose(1, 2)))
                        for (name, tpar_bn_fs), mpar_bn_fs in zip(tpar_bn_fisher.items(), mpar_bn_fisher.values())
                    ]))

                upd_scale_list += upd_scale_ls
                means += self.mean
                act_covs += mpar_tpar_mpar_act_cov
                grad_covs += mpar_tpar_mpar_grad_cov
                bn_fishers += mpar_tpar_mpar_bn_fisher

        # to make sure all lists have same length
        iter_ls = iter([upd_scale_list, means, act_covs, grad_covs, bn_fishers])
        length = len(next(iter_ls))
        if not all(len(ls) == length for ls in iter_ls):
            raise RuntimeError(
                '''Lengths of upd_scale_list, mean, act_cov, grad_cov or bn_fisher not the same. 
                Please check''')

        for upd_scale, mu, act_op, grad_op, fisher in zip(upd_scale_list, means, act_covs, grad_covs, bn_fishers):
            reg_allmodules = []
            for name, module in iter_named_modules(self.model):
                diff_cov_diff_prod = \
                    self.gaussian_logprob_conv(param, name, module, mu, act_cov=act_op, grad_cov=grad_op) \
                        if isinstance(module, (MetaConv1d, MetaConv2d, MetaConv3d, nn.Conv1d, nn.Conv2d, nn.Conv3d)) \
                    else self.gaussian_logprob_linear(param, name, module, mu, act_cov=act_op, grad_cov=grad_op) \
                        if isinstance(module, (MetaLinear, nn.Linear)) \
                    else self.gaussian_logprob_unitbn(param, name, module, mu, fisher) \
                        if isinstance(module, (MetaBatchNorm1d, MetaBatchNorm2d, nn.BatchNorm1d, nn.BatchNorm2d)) \
                    else NotImplementedError

                reg_allmodules.append(diff_cov_diff_prod)
            logprob_task.append(upd_scale * sum(reg_allmodules))

        return -0.5 * sum(logprob_task)

    def update_mean(self):
        mean = OrderedDict([
            (name, param.clone().detach()) for name, param in self.model.meta_named_parameters()
        ])
        if self.is_lapl_list:
            self.mean.append(mean)

            if self.max_lapl_list_len is not None and len(self.mean) > self.max_lapl_list_len:
                del self.mean[0]
        else:
            self.mean = [mean]

    def update_hessian(self, term, new_act_cov, new_grad_cov, new_bn_fisher, norm='fro'):
        # define which hessian term to update: tpar or mpar
        # 'tpar' is for hessian of nll_query wrt task-adapted parameters
        # 'mpar' is for hessian of nll_supp wrt meta-params
        if term == 'tpar':
            act_cov = self.tpar_act_cov
            grad_cov = self.tpar_grad_cov
            bn_fisher = self.tpar_bn_fisher
        elif term == 'mpar':
            act_cov = self.mpar_act_cov
            grad_cov = self.mpar_grad_cov
            bn_fisher = self.mpar_bn_fisher
        else:
            raise ValueError('term must be one of ("tpar" or "mpar").')

        if self.is_lapl_list:
            act_cov.append(new_act_cov)
            grad_cov.append(new_grad_cov)
            bn_fisher.append(new_bn_fisher)

            if self.max_lapl_list_len is not None and len(act_cov) > self.max_lapl_list_len:
                del act_cov[0]
                del grad_cov[0]
                del bn_fisher[0]
        else:
            # approx sum of two kronecker products as one k-prod
            # norm order 'fro' for frobenius norm in scaling for approximation
            for name in act_cov[0].keys():
                # compute multiplier that minimise norm of approx residual
                multiplier = torch.sqrt(
                    (self.upd_scale * torch.norm(act_cov[0][name], norm) * torch.norm(new_grad_cov[name], norm))
                    / (torch.norm(new_act_cov[name], norm) * torch.norm(grad_cov[0][name], norm))
                )
                act_cov[0][name] = new_act_cov[name] + act_cov[0][name] / multiplier
                grad_cov[0][name] = self.upd_scale * new_grad_cov[name] + multiplier * grad_cov[0][name]

    def get_fisher_bd_kfac_covs(self, dataloader, param=None, run_inner=True, nstep_inner=None, lr_inner=None,
                                exclude_module_name=None, verbose='KFAC calc'):
        act_op, grad_op, bn_op \
            = self.init_kfac_blockdiag_fisher(kfac_init_mult=[0. for _ in range(len(self.kfac_init_mult))])
        act_op, grad_op, bn_op = act_op[0], grad_op[0], bn_op[0]

        act_op_denom, grad_op_denom, bn_op_denom = self.init_kfac_blockdiag_fisher_denom()

        tqdm_total = dataloader.batch_sampler.__len__()
        supp_idx = dataloader.batch_sampler.num_way * dataloader.batch_sampler.num_shot

        with KroneckerContext(self.model) as ctxt:
            for images, labels in (
                    tqdm(dataloader, desc=verbose, total=tqdm_total) if verbose is not None else dataloader):
                support_img, query_img = images[:supp_idx, :], images[supp_idx:, :]
                support_lbl, query_lbl = labels[:supp_idx], labels[supp_idx:]

                # run inner loop
                if run_inner:
                    param_inner = inner_maml(
                        model=self.model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                        lr_inner=lr_inner, first_order=True, param_init=None
                    )
                    self.model.zero_grad()
                    out = self.model(x=query_img, param=param_inner)
                else:
                    self.model.zero_grad()
                    out = self.model(x=support_img, param=param)
                # currently only support 1 y-sample
                y_sample = torch.distributions.Categorical(F.softmax(out, dim=1)).sample((1,)).t()
                loss = F.cross_entropy(out, y_sample.squeeze(), reduction='sum')
                loss.backward()

                for name, mod in iter_named_modules(self.model):
                    if name != exclude_module_name:
                        if isinstance(mod, (MetaLinear, MetaConv1d, MetaConv2d, MetaConv3d,
                                            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                            act_op[name] += ctxt.activation_outprod[mod]
                            grad_op[name] += ctxt.preact_grad_outprod[mod]
                            act_op_denom[name] += ctxt.activation_outprod_denom[mod]
                            grad_op_denom[name] += ctxt.preact_grad_outprod_denom[mod]
                        elif isinstance(mod, (MetaBatchNorm1d, MetaBatchNorm2d, MetaBatchNorm3d,
                                              nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                            bn_op[name] += ctxt.batchnorm_grad_outprod[mod]
                            bn_op_denom[name] += ctxt.batchnorm_grad_outprod_denom[mod]
                    else:
                        pass

        # divide by denom
        act_cov = OrderedDict([
            (name, a_op / a_op_denom) for (name, a_op), a_op_denom in zip(act_op.items(), act_op_denom.values())
        ])
        grad_cov = OrderedDict([
            (name, g_op / g_op_denom) for (name, g_op), g_op_denom in zip(grad_op.items(), grad_op_denom.values())
        ])
        bn_fisher = OrderedDict([
            (name, b_op / b_op_denom) for (name, b_op), b_op_denom in zip(bn_op.items(), bn_op_denom.values())
        ])
        return act_cov, grad_cov, bn_fisher
