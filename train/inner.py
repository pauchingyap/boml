import torch
import torch.nn.functional as F
from collections import OrderedDict


def inner_maml(model, inputs, labels, nstep_inner, lr_inner, first_order, param_init=None):
    # zero gradients
    if param_init is None:
        model.zero_grad()
    else:
        for param in param_init.values():
            if param.grad is not None:
                param.grad.zero_()

    # taking first step
    outputs = model(inputs, param=param_init)
    loss = F.cross_entropy(input=outputs, target=labels, reduction='mean')
    if param_init is None:
        gradients = torch.autograd.grad(loss, model.meta_parameters(), create_graph=not first_order)
        param_inner = OrderedDict([
            (name, param - lr_inner * grad)
            for (name, param), grad in zip(model.meta_named_parameters(), gradients)
        ])
    else:
        gradients = torch.autograd.grad(loss, param_init.values(), create_graph=not first_order)
        param_inner = OrderedDict([
            (name, param - lr_inner * grad)
            for (name, param), grad in zip(param_init.items(), gradients)
        ])

    # taking remaining steps
    for _ in range(nstep_inner - 1):
        outputs = model(inputs, param=param_inner)
        loss = F.cross_entropy(input=outputs, target=labels, reduction='mean')
        gradients = torch.autograd.grad(loss, param_inner.values(), create_graph=not first_order)
        param_inner = OrderedDict([
            (name, param - lr_inner * grad) for (name, param), grad in zip(param_inner.items(), gradients)
        ])

    return param_inner
