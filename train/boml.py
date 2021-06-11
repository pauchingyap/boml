import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from functional.cross_entropy import cross_entropy
from data_generate.sampler import SuppQueryBatchSampler
from train.inner import inner_maml
from train.util import get_accuracy, kldiv_mvn_diagcov
from train.eval import meta_evaluation, meta_evaluation_vi


class BayesianOnlineMetaLearn(object, metaclass=ABCMeta):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def metatrain(self, *args):
        pass

    @abstractmethod
    def metatrain_seqtask(self, *args):
        pass


class BayesianOnlineMetaLearnLaplaceApprox(BayesianOnlineMetaLearn):
    def __init__(self, model, laplace_obj, device):
        super().__init__(model, device)
        self.laplace_obj = laplace_obj

    def metatrain(
            self, trainset, evalset, optimiser_outer, lr_scheduler_outer, lapl_approx_reg, nstep_outer, nstep_inner,
            lr_inner, first_order, seqtask, num_shot, num_query_per_cls, num_task_per_itr, task_by_supercls,
            eval_prev_task, eval_per_num_iter, num_eval_task, eval_task_by_supercls, nstep_inner_eval, writer, task_idx,
            logfile_path, verbose
    ):

        with trange(nstep_outer, desc='Meta-training {}'.format(verbose if verbose is not None else task_idx)) as pbar:
            for itr in range(nstep_outer):
                nll_query = torch.tensor(0., device=self.device)
                nll_supp = torch.tensor(0., device=self.device)
                accuracy_query = torch.tensor(0., device=self.device)

                trainsampler = SuppQueryBatchSampler(
                    dataset=trainset, seqtask=seqtask, num_task=num_task_per_itr, task_by_supercls=task_by_supercls,
                    num_way=self.model.num_way, num_shot=num_shot, num_query_per_cls=num_query_per_cls
                )
                trainloader = DataLoader(trainset, batch_sampler=trainsampler)

                for batch_idx, (images, labels) in enumerate(trainloader, 0):
                    supp_idx = self.model.num_way * num_shot
                    support_img, query_img = images[:supp_idx, :], images[supp_idx:, :]
                    support_lbl, query_lbl = labels[:supp_idx], labels[supp_idx:]

                    # accumulate log-likelihood of supp set wrt meta params
                    nll_supp += F.cross_entropy(self.model(support_img), support_lbl, reduction='mean')

                    # run k-step inner loop
                    param_inner = inner_maml(
                        model=self.model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                        lr_inner=lr_inner, first_order=first_order, param_init=None
                    )
                    # accumulate loss of query set wrt param_inner
                    out_query = self.model(x=query_img, param=param_inner)
                    nll_query += F.cross_entropy(input=out_query, target=query_lbl, reduction='mean')
                    with torch.no_grad():
                        accuracy_query += get_accuracy(labels=query_lbl, outputs=out_query)

                if lapl_approx_reg:
                    loss_outer = \
                        (nll_query + nll_supp) / num_task_per_itr - self.laplace_obj.laplace_approx(param=None) \
                            if self.laplace_obj.nll_supp_wrt_metaparam \
                        else nll_query / num_task_per_itr - self.laplace_obj.laplace_approx(param=None)
                else:
                    loss_outer = nll_query / num_task_per_itr

                accuracy_query.div_(num_task_per_itr)

                optimiser_outer.zero_grad()
                loss_outer.backward()
                optimiser_outer.step()

                # meta-evaluation
                if (itr + 1) % eval_per_num_iter == 0:
                    if not eval_prev_task:
                        evalset = [evalset]

                    for ldr_idx, evset in enumerate(evalset):
                        loss_eval, accuracy_eval = meta_evaluation(
                            evalset=evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls,
                            num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=self.model,
                            nstep_inner=nstep_inner_eval, lr_inner=lr_inner,
                        )
                        writer.add_scalar(
                            tag='loss_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss_meta_eval',
                            scalar_value=loss_eval, global_step= task_idx * nstep_outer + itr
                        )
                        writer.add_scalar(
                            tag='accuracy_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'accuracy_meta_eval',
                            scalar_value=accuracy_eval, global_step=task_idx * nstep_outer + itr
                        )

                if lr_scheduler_outer is not None:
                    if 'ReduceLROnPlateau' in lr_scheduler_outer.__class__.__name__:
                        lr_scheduler_outer.step(loss_outer)
                    else:
                        lr_scheduler_outer.step()

                pbar.update(n=1)
                if logfile_path is not None:
                    with open(logfile_path, 'a') as logfile:
                        logfile.write(str(pbar) + ('\n' if itr + 1 == nstep_outer else ' '))

    def metatrain_seqtask(
            self, trainloader, evalset, optimiser_outer, lr_scheduler_outer, lapl_approx_reg, lr_inner, nstep_outer,
            nstep_inner, first_order, num_query_per_cls, eval_per_num_epoch, num_eval_task, eval_task_by_supercls,
            nstep_inner_eval, writer, task_idx, verbose
    ):
        num_way = trainloader.batch_sampler.num_way
        num_shot = trainloader.batch_sampler.num_shot
        supp_idx = num_way * num_shot
        num_batch = trainloader.__len__()

        for epoch in (
                trange(nstep_outer, desc='metatrain {} {}'.format(verbose, task_idx)) if verbose is not None
                else range(nstep_outer)):

            nll_query = torch.tensor(0., device=self.device)
            nll_supp = torch.tensor(0., device=self.device)
            acc_query = torch.tensor(0., device=self.device)

            for batch_idx, (images, labels) in enumerate(trainloader, 0):
                support_img, query_img = images[:supp_idx, :], images[supp_idx:, :]
                support_lbl, query_lbl = labels[:supp_idx], labels[supp_idx:]

                # negloglik of support set
                nll_supp += F.cross_entropy(self.model(support_img), support_lbl, reduction='mean')

                # run k-step inner loop
                param_inner = inner_maml(
                    model=self.model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                    lr_inner=lr_inner, first_order=first_order, param_init=None
                )
                # negloglik of query set wrt param_inner
                out_query = self.model(x=query_img, param=param_inner)
                nll_query += F.cross_entropy(input=out_query, target=query_lbl, reduction='mean')
                with torch.no_grad():
                    acc_query += get_accuracy(labels=query_lbl, outputs=out_query)

            if lapl_approx_reg:
                loss_outer = (nll_query + nll_supp) / num_batch - self.laplace_obj.laplace_approx(param=None) \
                    if self.laplace_obj.nll_supp_wrt_metaparam \
                    else nll_query / num_batch - self.laplace_obj.laplace_approx(param=None)
            else:
                loss_outer = nll_query

            acc_query.div_(num_batch)

            optimiser_outer.zero_grad()
            loss_outer.backward()
            optimiser_outer.step()

            if lr_scheduler_outer is not None:
                lr_scheduler_outer.step()

            # meta-evaluation
            if (epoch + 1) % eval_per_num_epoch == 0:
                loss_eval, acc_eval = meta_evaluation(
                    evalset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_shot=num_shot,
                    num_query_per_cls=num_query_per_cls, model=self.model, nstep_inner=nstep_inner_eval,
                    lr_inner=lr_inner
                )
                if writer is not None:
                    ev_glob_step = task_idx * nstep_outer + epoch + 1
                    glb_step = ev_glob_step / eval_per_num_epoch if eval_per_num_epoch == nstep_outer else ev_glob_step
                    writer.add_scalar(tag='loss_meta_eval', scalar_value=loss_eval, global_step=glb_step)
                    writer.add_scalar(tag='accuracy_meta_eval', scalar_value=acc_eval, global_step=glb_step)


class BayesianOnlineMetaLearnVariationalInference(BayesianOnlineMetaLearn):
    def __init__(self, model, variational_obj, device):
        super().__init__(model, device)
        self.var_obj = variational_obj

    def metatrain(
            self, trainset, evalset, optimiser_outer, lr_scheduler_outer, nstep_outer, nstep_inner, lr_inner,
            first_order, seqtask, num_task_per_itr, task_by_supercls, num_shot, num_query_per_cls, eval_prev_task,
            eval_per_num_iter, num_eval_task, eval_task_by_supercls, nstep_inner_eval, writer, task_idx, logfile_path,
            verbose
    ):
        with trange(nstep_outer, desc='meta-train {}'.format(verbose if verbose is not None else task_idx)) as pbar:
            for itr in range(nstep_outer):
                nll_querysupp_gradient_wrt_mean = OrderedDict([
                    (name, torch.zeros_like(mu)) for name, mu in self.var_obj.mean.items()
                ])
                nll_querysupp_gradient_wrt_covar = OrderedDict([
                    (name, torch.zeros_like(cov)) for name, cov in self.var_obj.covar.items()
                ])

                negloglik_query = torch.tensor(0., device=self.device)
                negloglik_supp = torch.tensor(0., device=self.device)
                loss_outer = torch.tensor(0., device=self.device)
                accuracy_query = torch.tensor(0., device=self.device)

                trainsampler = SuppQueryBatchSampler(
                    dataset=trainset, seqtask=seqtask, num_task=num_task_per_itr, task_by_supercls=task_by_supercls,
                    num_way=self.model.num_way, num_shot=num_shot, num_query_per_cls=num_query_per_cls
                )
                trainloader = DataLoader(trainset, batch_sampler=trainsampler)

                nll_querysupp_one_task_divisor = (num_shot + num_query_per_cls) * self.model.num_way \
                                                 * self.var_obj.num_mc_sample

                for batch_idx, (images, labels) in enumerate(trainloader, 0):
                    supp_idx = self.model.num_way * num_shot
                    support_img, query_img = images[:supp_idx, :], images[supp_idx:, :]
                    support_lbl, query_lbl = labels[:supp_idx], labels[supp_idx:]

                    query_img = query_img.expand(self.var_obj.num_mc_sample, *query_img.size())
                    query_lbl = query_lbl.expand(self.var_obj.num_mc_sample, *query_lbl.size())

                    # query set adapted neg-log-likelihood mc estimate -- inner on mean instead of sampled theta
                    mean_inner = inner_maml(
                        model=self.model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                        lr_inner=lr_inner, first_order=first_order, param_init=self.var_obj.mean
                    )
                    out_query = self.model(x=query_img, mean=mean_inner, cov=self.var_obj.exp_covar(self.var_obj.covar))
                    nll_query = cross_entropy(input=out_query, target=query_lbl, reduction='sum')

                    support_img = support_img.expand(self.var_obj.num_mc_sample, *support_img.size())
                    support_lbl = support_lbl.expand(self.var_obj.num_mc_sample, *support_lbl.size())

                    # support set log-likelihood monte carlo estimate
                    # shape (n_sample, batch, num_way)
                    out_supp = self.model(
                        x=support_img, mean=self.var_obj.mean, cov=self.var_obj.exp_covar(self.var_obj.covar)
                    )
                    # reduction on n_sample dim only
                    nll_supp = cross_entropy(input=out_supp, target=support_lbl, reduction='sum')

                    # loss for the supp and query negloglik terms for one task
                    nll_querysupp_one_task = (nll_query + nll_supp) / nll_querysupp_one_task_divisor

                    # accumulate gradient for query and supp loss per batch
                    optimiser_outer.zero_grad()
                    nll_querysupp_one_task.backward()

                    with torch.no_grad():
                        for g_mu, g_cov, mu, cov in \
                            zip(nll_querysupp_gradient_wrt_mean.values(), nll_querysupp_gradient_wrt_covar.values(),
                                self.var_obj.mean.values(), self.var_obj.covar.values()):

                            g_mu += mu.grad
                            g_cov += cov.grad

                            # zero gradients after accumulating mean and cov gradients
                            mu.grad.zero_()
                            cov.grad.zero_()

                        # accumulate nll no_grad for graph
                        loss_outer += nll_querysupp_one_task
                        accuracy_query += get_accuracy(labels=query_lbl, outputs=out_query)
                        negloglik_query += nll_query
                        negloglik_supp += nll_supp

                loss_outer.div_(num_task_per_itr)
                accuracy_query.div_(num_task_per_itr)
                negloglik_query.div_(num_task_per_itr * nll_querysupp_one_task_divisor)
                negloglik_supp.div_(num_task_per_itr * nll_querysupp_one_task_divisor)

                # kl-div term
                kldiv = kldiv_mvn_diagcov(
                    mean_p=self.var_obj.mean, cov_p=self.var_obj.exp_covar(self.var_obj.covar),
                    mean_q=self.var_obj.mean_old, cov_q=self.var_obj.exp_covar(self.var_obj.covar_old)
                ) / (nll_querysupp_one_task_divisor * num_task_per_itr)

                with torch.no_grad():
                    loss_outer += kldiv

                # accumulate gradient for kldiv term
                optimiser_outer.zero_grad()
                kldiv.backward()

                kldiv_gradient_wrt_mean = OrderedDict([(name, mu.grad) for name, mu in self.var_obj.mean.items()])
                kldiv_gradient_wrt_covar = OrderedDict([(name, cov.grad) for name, cov in self.var_obj.covar.items()])

                # add nll and kldiv gradients together
                total_gradient_wrt_mean = OrderedDict([
                    (name, nll_grad / num_task_per_itr + kldiv_grad) for (name, nll_grad), kldiv_grad
                    in zip(nll_querysupp_gradient_wrt_mean.items(), kldiv_gradient_wrt_mean.values())
                ])
                total_gradient_wrt_covar = OrderedDict([
                    (name, nll_grad / num_task_per_itr + kldiv_grad) for (name, nll_grad), kldiv_grad
                    in zip(nll_querysupp_gradient_wrt_covar.items(), kldiv_gradient_wrt_covar.values())
                ])

                optimiser_outer.step_grad(gradient=list(total_gradient_wrt_mean.values())
                                                   + list(total_gradient_wrt_covar.values()))

                # meta-evaluation
                if (itr + 1) % eval_per_num_iter == 0:
                    if not eval_prev_task:
                        evalset = [evalset]

                    for ldr_idx, evset in enumerate(evalset):
                        loss_eval, accuracy_eval = meta_evaluation_vi(
                            evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls,
                            num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=self.model,
                            variational_obj=self.var_obj, inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval,
                            lr_inner=lr_inner, device=self.device
                        )
                        writer.add_scalar(
                            tag='loss_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss_meta_eval',
                            scalar_value=loss_eval, global_step=task_idx * nstep_outer + itr
                        )
                        writer.add_scalar(
                            tag='accuracy_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'accuracy_meta_eval',
                            scalar_value=accuracy_eval, global_step=task_idx * nstep_outer + itr
                        )

                if lr_scheduler_outer is not None:
                    if 'ReduceLROnPlateau' in lr_scheduler_outer.__class__.__name__:
                        lr_scheduler_outer.step(loss_outer)
                    else:
                        lr_scheduler_outer.step()

                pbar.update(n=1)
                if logfile_path is not None:
                    with open(logfile_path, 'a') as logfile:
                        logfile.write(str(pbar) + ('\n' if itr + 1 == nstep_outer else ' '))

    def metatrain_seqtask(
            self, trainloader, evalset, optimiser_outer, lr_scheduler_outer, lr_inner, nstep_outer, nstep_inner,
            first_order, eval_per_num_epoch, num_eval_task, eval_task_by_supercls, num_query_per_cls, nstep_inner_eval,
            writer, task_idx, verbose
    ):
        num_way = trainloader.batch_sampler.num_way
        num_shot = trainloader.batch_sampler.num_shot
        supp_idx = num_way * num_shot
        num_batch = trainloader.__len__()

        for epoch in (
                trange(nstep_outer, desc='metatrain {} {}'.format(verbose, task_idx)) if verbose is not None
                else range(nstep_outer)):

            nll_query = torch.tensor(0., device=self.device)
            nll_supp = torch.tensor(0., device=self.device)
            acc_query = torch.tensor(0., device=self.device)
            num_sample_per_epoch = torch.tensor(0., device=self.device)

            for batch_idx, (images, labels) in enumerate(trainloader, 0):

                num_sample_per_epoch += labels.size(0)

                support_img, query_img = images[:supp_idx, :], images[supp_idx:, :]
                support_lbl, query_lbl = labels[:supp_idx], labels[supp_idx:]

                query_img = query_img.expand(self.var_obj.num_mc_sample, *query_img.size())
                query_lbl = query_lbl.expand(self.var_obj.num_mc_sample, *query_lbl.size())

                # query set adapted neg-log-likelihood mc estimate -- inner on mean instead of sampled theta
                mean_inner = inner_maml(
                    model=self.model, inputs=support_img, labels=support_lbl, nstep_inner=nstep_inner,
                    lr_inner=lr_inner, first_order=first_order, param_init=self.var_obj.mean
                )
                out_query = self.model(x=query_img, mean=mean_inner, cov=self.var_obj.exp_covar(self.var_obj.covar))
                nll_query += cross_entropy(input=out_query, target=query_lbl, reduction='mean')
                with torch.no_grad():
                    acc_query += get_accuracy(labels=query_lbl, outputs=out_query)

                support_img = support_img.expand(self.var_obj.num_mc_sample, *support_img.size())
                support_lbl = support_lbl.expand(self.var_obj.num_mc_sample, *support_lbl.size())

                # support set log-likelihood monte carlo estimate
                # shape (n_sample, batch, num_way)
                out_supp \
                    = self.model(x=support_img, mean=self.var_obj.mean, cov=self.var_obj.exp_covar(self.var_obj.covar))
                # negloglik for support set
                nll_supp += cross_entropy(input=out_supp, target=support_lbl, reduction='mean')

            kldiv = kldiv_mvn_diagcov(
                mean_p=self.var_obj.mean, cov_p=self.var_obj.exp_covar(self.var_obj.covar),
                mean_q=self.var_obj.mean_old, cov_q=self.var_obj.exp_covar(self.var_obj.covar_old)
            ) / (num_sample_per_epoch * self.var_obj.num_mc_sample)

            loss_outer = (nll_query + nll_supp) / num_batch + kldiv

            acc_query.div_(num_batch)

            # update mean and covar
            optimiser_outer.zero_grad()

            loss_outer.backward()
            optimiser_outer.step()

            if lr_scheduler_outer is not None:
                lr_scheduler_outer.step()

            # meta-evaluation
            if (epoch + 1) % eval_per_num_epoch == 0:
                loss_eval, acc_eval = meta_evaluation_vi(
                    evalset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_shot=num_shot,
                    num_query_per_cls=num_query_per_cls, model=self.model, variational_obj=self.var_obj,
                    inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval, lr_inner=lr_inner, device=self.device
                )
                if writer is not None:
                    ev_glob_step = task_idx * nstep_outer + epoch + 1
                    glb_step = ev_glob_step / eval_per_num_epoch if eval_per_num_epoch == nstep_outer else ev_glob_step

                    writer.add_scalar(tag='loss_meta_eval', scalar_value=loss_eval, global_step=glb_step)
                    writer.add_scalar(tag='accuracy_meta_eval', scalar_value=acc_eval, global_step=glb_step)
