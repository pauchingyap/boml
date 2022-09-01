import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from boml.modules.module import MetaModuleMonteCarlo


class MetaConv2dMonteCarlo(nn.Conv2d, MetaModuleMonteCarlo):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None, mean=None, cov=None):
        if params is not None:
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                params['weight'], params.get('bias', None), self.stride,
                                _pair(0), self.dilation, self.groups)

            return F.conv2d(input, params['weight'], params.get('bias', None), self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            # input shape (n_sample, batch, in_ch, height, width)
            inp_sample = input.reshape(input.size(0) * input.size(1), *input.size()[2:])
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                preact_mean = F.conv2d(F.pad(inp_sample, expanded_padding, mode='circular'),
                                mean['weight'], mean.get('bias', None), self.stride,
                                _pair(0), self.dilation, self.groups)
                preact_cov = F.conv2d(F.pad(inp_sample, expanded_padding, mode='circular') ** 2,
                                       cov['weight'], cov.get('bias', None), self.stride,
                                       _pair(0), self.dilation, self.groups)
            else:
                preact_mean = F.conv2d(inp_sample ** 2, mean['weight'], mean.get('bias', None), self.stride,
                                       self.padding, self.dilation, self.groups)
                preact_cov = F.conv2d(inp_sample ** 2, cov['weight'], cov.get('bias', None), self.stride,
                                      self.padding, self.dilation, self.groups)

            preact = preact_mean + torch.sqrt(preact_cov) * torch.randn_like(preact_mean)
            return preact.reshape(input.size(0), input.size(1), *preact_mean.size()[1:])
