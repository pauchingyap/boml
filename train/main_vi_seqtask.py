import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import datetime
import os
import random
import json
import logging
from tqdm import tqdm

from boml.main import model as models
from boml.main.variational import VariationalApprox
from boml.main.boml import BayesianOnlineMetaLearnVariationalInference
from boml.main.util import enlist_transformation
from boml.data_generate.dataset import FewShotImageDataset
from boml.data_generate.sampler import SuppQueryBatchSampler

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(config, logger, run_spec, data_path, seed):
    torch.manual_seed(seed)
    start_datetime = datetime.datetime.now()

    experiment_date = '{:%Y-%m-%d_%H:%M:%S}'.format(start_datetime)
    config['experiment_parent_dir'] = os.path.join(config['run_dir'], config['dataset'])
    config['experiment_dir'] = os.path.join(config['experiment_parent_dir'],
                                            '{}_{}_{}'.format(run_spec, experiment_date, seed))
    config['data_path'] = data_path

    if not os.path.exists(config['experiment_dir']):
        os.makedirs(config['experiment_dir'])

    with open(os.path.join(config['experiment_dir'], 'config{}_{}.json'.format(0, run_spec)), 'w') as outfile:
        outfile.write(json.dumps(config, indent=4))

    logfile_path = os.path.join(config['experiment_dir'], 'logfile_{}_{}.log'.format(run_spec, seed))
    filehandler = logging.FileHandler(filename=logfile_path, mode='a')
    logger.addHandler(filehandler)
    # to avoid excessive log prints when loading dataset
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    logger.info('running {}_{} seed {}'.format(run_spec, experiment_date, seed))

    split_dir = os.path.join('./data_split', config['dataset'])

    # load task list
    taskls = np.load(config['tasklist_path'], allow_pickle=True).tolist()
    np.save(os.path.join(config['experiment_dir'], 'tasklist.npy'), taskls)
    tasklist = [[os.path.join(config['data_path'], path) for path in task] for task in taskls]

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

    transformation = transforms.Compose(enlist_transformation(device=config['device'], **config['transfm_kwargs']))
    metatest_ls = np.load(os.path.join(split_dir, 'metatest.npy'), allow_pickle=True).tolist()
    metatest_dir_ls = [os.path.join(data_path, metatest) for metatest in metatest_ls]
    evalset = FewShotImageDataset(task_list=metatest_dir_ls, supercls=True, img_lvl=2, transform=transformation,
                                  device=config['device'], cuda_img=config['cuda_img'], verbose=None)

    with tqdm(enumerate(tasklist), desc='bomvi seqtask', total=len(tasklist)) as pbar:
        for task_idx, task in enumerate(tasklist):
            optim_outer = getattr(optim, config['optim_outer_name'])\
                (list(var_approx.mean.values()) + list(var_approx.covar.values()), **config['optim_outer_kwargs'])
            if config['lr_sch_outer_name'] is None:
                scheduler_outer = None
            else:
                scheduler_outer = getattr(lr_scheduler, config['lr_sch_outer_name'])\
                    (optim_outer, **config['lr_sch_outer_kwargs'])

            trainset = FewShotImageDataset(task_list=task, supercls=True, img_lvl=1, transform=transformation,
                                           device=config['device'], cuda_img=config['cuda_img'], verbose=None)
            trainsampler = SuppQueryBatchSampler(dataset=trainset,  **config['trainsampler_kwargs'])
            trainloader = DataLoader(trainset, batch_sampler=trainsampler)

            # meta-training
            bomvi.metatrain_seqtask(
                trainloader=trainloader, evalset=evalset, optimiser_outer=optim_outer,
                lr_scheduler_outer=scheduler_outer, writer=writer, task_idx=task_idx, verbose=None,
                **config['meta_train_eval_kwargs']
            )
            # update mean and covariance
            var_approx.update_mean_cov()

            torch.save(var_approx.mean, f=os.path.join(result_dir, 'mean.pt'))
            torch.save(var_approx.covar, f=os.path.join(result_dir, 'covar.pt'))
            torch.save(model.state_dict(), f=os.path.join(result_dir, 'model.pt'))

            pbar.update(n=1)
            if logfile_path is not None:
                with open(logfile_path, 'a') as logfile:
                    logfile.write(str(pbar) + ('\n' if task_idx + 1 == len(tasklist) else ' '))

    logger.info('completed in {}'.format(datetime.datetime.now() - start_datetime))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('BOMVI Sequential Task')
    parser.add_argument('--config_path', type=str, help='Path of .json file to import config from')
    parser.add_argument('--data_path', type=str, help='Parental directory path containing all datasets')
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
              data_path=args.data_path, seed=random.getrandbits(24))
    except Exception as exc:
        logger.exception(exc)
    except KeyboardInterrupt as kbi:
        logger.exception(kbi)
