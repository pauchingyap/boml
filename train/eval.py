import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict

from data_generate.sampler import SuppQueryBatchSampler
from train.inner import inner_maml
from train.util import get_accuracy


def meta_evaluation(evalset, num_task, num_shot, num_query_per_cls, model, nstep_inner, lr_inner):
    loss = []
    accuracy = []

    num_way = model.num_way

    evalsampler = SuppQueryBatchSampler(dataset=evalset, seqtask=False, num_task=num_task, num_way=num_way,
                                        num_shot=num_shot, num_query_per_cls=num_query_per_cls)
    evalloader = DataLoader(evalset, batch_sampler=evalsampler)

    for data in evalloader:
        images, labels = data

        supp_idx = num_way * num_shot
        support_img, query_img = images[:supp_idx, :], images[supp_idx:, :]
        support_lbl, query_lbl = labels[:supp_idx], labels[supp_idx:]

        param_init = OrderedDict([
            (name, param.clone().detach().requires_grad_(True)) for name, param in model.meta_named_parameters()
        ])
        param_inner = inner_maml(model=model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                                 lr_inner=lr_inner, first_order=True, param_init=param_init)
        # return loss, accuracy
        output = model(x=query_img, param=param_inner)
        with torch.no_grad():
            loss.append(F.cross_entropy(input=output, target=query_lbl, reduction='mean'))
            accuracy.append(get_accuracy(labels=query_lbl, outputs=output))

    loss_tensor = torch.stack(loss)
    acc_tensor = torch.stack(accuracy)

    loss_mean = loss_tensor.mean()
    acc_mean = acc_tensor.mean()

    return loss_mean.item(), acc_mean.item()


def meta_evaluation_vi(evalset, num_task, num_shot, num_query_per_cls, model, variational_obj, inner_on_mean,
                       n_sample=1, nstep_inner=None, lr_inner=None, device=None):
    loss = []
    accuracy = []

    evalsampler = SuppQueryBatchSampler(dataset=evalset, seqtask=False, num_task=num_task, num_way=model.num_way,
                                        num_shot=num_shot, num_query_per_cls=num_query_per_cls)
    evalloader = DataLoader(evalset, batch_sampler=evalsampler)

    for images, labels in evalloader:

        supp_idx = model.num_way * num_shot
        support_img, query_img = images[:supp_idx, :], images[supp_idx:, :]
        support_lbl, query_lbl = labels[:supp_idx], labels[supp_idx:]

        if inner_on_mean:
            param_init = OrderedDict([
                (name, param.clone().detach().requires_grad_(True)) for name, param in variational_obj.mean.items()
            ])
            mean_inner = inner_maml(
                model=model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                lr_inner=lr_inner, first_order=True, param_init=param_init
            )
            output = model(x=query_img, param=mean_inner)
            with torch.no_grad():
                nll = F.cross_entropy(input=output, target=query_lbl, reduction='mean')
                acc = get_accuracy(labels=query_lbl, outputs=output)
        else:
            nll = torch.tensor(0., device=device)
            acc = torch.tensor(0., device=device)

            for _ in range(n_sample):
                param_init = variational_obj.sample_params(n_sample=1, detach_mean_cov=True)
                param_inner = inner_maml(model=model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                                         lr_inner=lr_inner, first_order=True, param_init=param_init)
                output = model(x=query_img, param=param_inner)
                with torch.no_grad():
                    nll += F.cross_entropy(input=output, target=query_lbl, reduction='mean')
                    acc += get_accuracy(labels=query_lbl, outputs=output)

            nll.div_(n_sample)
            acc.div_(n_sample)

        loss.append(nll)
        accuracy.append(acc)

    loss_tensor = torch.stack(loss)
    acc_tensor = torch.stack(accuracy)

    loss_mean = loss_tensor.mean()
    acc_mean = acc_tensor.mean()

    return loss_mean.item(), acc_mean.item()
