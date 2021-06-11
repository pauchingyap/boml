import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import datetime
import os
import json
import random
import logging

import optim
from config.configuration import get_run_name
from data_generate.dataset import FewShotImageDataset
from train import model as models
from train.variational import VariationalApprox
from train.boml import BayesianOnlineMetaLearnVariationalInference
from train.util import enlist_transformation

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(config, logger, run_spec, data_dir, seed=0):
    torch.manual_seed(seed)

    start_datetime = datetime.datetime.now()
    experiment_date = '{:%Y-%m-%d_%H:%M:%S}'.format(start_datetime)
    config['experiment_parent_dir'] = os.path.join(config['run_dir'], get_run_name(config['dataset_ls']))
    config['experiment_dir'] = os.path.join(
        config['experiment_parent_dir'], '{}_{}_{}'.format(run_spec, experiment_date, seed)
    )
    config['data_dir'] = data_dir

    # save config json file
    if not os.path.exists(config['experiment_dir']):
        os.makedirs(config['experiment_dir'])
    with open(os.path.join(config['experiment_dir'], 'config_{}_{}.json'.format(run_spec, seed)), 'w') as outfile:
        outfile.write(json.dumps(config, indent=4))

    # add file handler
    logfile_path = os.path.join(config['experiment_dir'], 'logfile_{}_{}.log'.format(run_spec, seed))
    filehandler = logging.FileHandler(filename=logfile_path, mode='a')
    logger.addHandler(filehandler)
    # to avoid excessive log prints when loading dataset
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    logger.info('running {}_{} seed {}'.format(run_spec, experiment_date, seed))

    # define tensorboard writer
    writer = SummaryWriter(os.path.join(config['experiment_dir'], 'logtb'))
    result_dir = os.path.join(config['experiment_dir'], 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # define model
    model = getattr(models, config['net'])(**config['net_kwargs']).to(device=config['device'])
    # define variational object
    var_approx = VariationalApprox(model=model, **config['variational_kwargs'])
    var_approx.detach_model_params()
    var_approx.update_mean_cov()
    # define meta-training object
    bomvi = BayesianOnlineMetaLearnVariationalInference(model, variational_obj=var_approx, device=config['device'])

    evalset = []
    for task_idx, task in enumerate(config['dataset_ls'], 0):
        # split directory for this dataset
        split_dir = os.path.join('./data_split', task)

        optim_outer = getattr(optim, config[task]['optim_outer_name']) \
            (list(var_approx.mean.values()) + list(var_approx.covar.values()), **config[task]['optim_outer_kwargs'])
        if config[task]['lr_sch_outer_name'] is None:
            scheduler_outer = None
        else:
            scheduler_outer = getattr(lr_scheduler, config[task]['lr_sch_outer_name']) \
                (optim_outer, **config[task]['lr_sch_outer_kwargs'])

        transformation = transforms.Compose(enlist_transformation(device=config['device'], **config['transfm_kwargs']))
        # load meta-train and meta-eval lists
        metatrain_ls = np.load(os.path.join(split_dir, 'metatrain.npy'), allow_pickle=True).tolist()
        metatest_ls = np.load(os.path.join(split_dir, 'metatest.npy'), allow_pickle=True).tolist()
        metatrain_dir_ls = [os.path.join(data_dir, metatrain) for metatrain in metatrain_ls]
        metatest_dir_ls = [os.path.join(data_dir, metatest) for metatest in metatest_ls]

        # define datasets
        trainset = FewShotImageDataset(
            task_list=metatrain_dir_ls, supercls=config[task]['supercls'], img_lvl=int(config[task]['supercls']) + 1,
            transform=transformation, device=config['device'], cuda_img=config['cuda_img'],
            verbose='{} trainset'.format(task)
        )
        evalset.append(FewShotImageDataset(
            task_list=metatest_dir_ls, supercls=config[task]['eval_supercls'],
            img_lvl=int(config[task]['eval_supercls']) + 1, transform=transformation, device=config['device'],
            cuda_img=config['cuda_img'], verbose='{} evalset'.format(task)
        ))

        # meta-training
        bomvi.metatrain(
            trainset=trainset, evalset=evalset, optimiser_outer=optim_outer, lr_scheduler_outer=scheduler_outer,
            eval_prev_task=True, writer=writer, task_idx=task_idx, logfile_path=logfile_path, verbose=task,
            **config['meta_train_eval_kwargs'], **config[task]['train_eval_kwargs']
        )
        # update mean and covariance
        var_approx.update_mean_cov()

        torch.save(var_approx.mean, f=os.path.join(result_dir, 'mean{}.pt'.format(task_idx)))
        torch.save(var_approx.covar, f=os.path.join(result_dir, 'covar{}.pt'.format(task_idx)))
        torch.save(model.state_dict(), f=os.path.join(result_dir, 'model{}.pt'.format(task_idx)))

        torch.cuda.empty_cache()

    logger.info('completed in {}'.format(datetime.datetime.now() - start_datetime))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('BOMVI Sequential Dataset')
    parser.add_argument('--config_path', type=str, help='Path of .json file to import config from')
    parser.add_argument('--data_dir', type=str, default='../data', help='Parental directory containing all datasets')
    args = parser.parse_args()

    # load config file
    jsonfile = open(str(args.config_path))
    config = json.loads(jsonfile.read())

    # define and configure log file
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    logging.captureWarnings(capture=True)
    logger = logging.getLogger(__name__)

    # train
    try:
        train(config=config, logger=logger, run_spec=os.path.splitext(os.path.split(args.config_path)[-1])[0],
              data_dir=args.data_dir, seed=random.getrandbits(24))
    except Exception as exc:
        logger.exception(exc)
    except KeyboardInterrupt as kbi:
        logger.exception(kbi)
