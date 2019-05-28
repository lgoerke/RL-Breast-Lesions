import os
import matplotlib
matplotlib.use('Agg')

import time
import torch
import torchvision
import shutil

from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler, SequentialSampler

from trainer_feat import FeatTrainer
import utils.datasets as u
from trainer_feat_pos import FeatTrainerPos
from trainer_feat_auto import FeatTrainerAuto
from trainer_feat_cat import FeatTrainerCat
from utils.torch_utils import *
import utils.transforms as t
from utils.utils import get_paths
from utils.config_utils import *
import torch.multiprocessing as mp
import pickle as pkl
from prepare_feat import *

def test(cfg_dict, model_string, rsyncing, learn_pos=False, toy=False, auto=False):
    """

    :param model_string:
    :param rsyncing:
    :param learn_pos:
    :param toy
    :return:
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Retrieve params
    opti, lr, mom, num_epochs = get_f_train_opti_lr_mom_epochs(cfg_dict)
    checkpoint_dir, tensorboard_dir, experiment_name = get_f_save_check_tensorB_expName(
        cfg_dict)
    selective_sampling, checkpoint_pretrained, cat = get_f_variants_selectiveS_checkPretrained_cat(
        cfg_dict)
    resume = get_seed_resume_lrSchedule_root(cfg_dict)[1]

    multiprocessing = False

    model = get_feature_model(model_string, experiment_name, selective_sampling, opti, lr, mom, checkpoint_pretrained,
                              learn_pos=learn_pos, cat=cat)

    if model_string == 'auto':
        criterion_lesion = nn.MSELoss()
    elif cat:
        criterion_lesion = nn.CrossEntropyLoss()
    else:
        criterion_lesion = nn.BCEWithLogitsLoss()
    if learn_pos:
        criterion_row = nn.MSELoss()
        criterion_col = nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), opti, lr, mom)

    print(model, flush=True)

    checkpoint_filename = os.path.join(
        checkpoint_dir, 'warmup_model_{}.pth.tar'.format(experiment_name))
    print('Write checkpoints to {}'.format(
        os.path.abspath(checkpoint_dir)), flush=True)
    print('Writing TensorBoard logs to {}'.format(
        os.path.abspath(tensorboard_dir)), flush=True)
    ######
    if os.path.exists(checkpoint_filename):
        # Don't load optimizer, otherwise LR might be too low already (yes?
        # TODO)
        model, _, initial_epoch = load_checkpoint(
            model, optimizer, filename=checkpoint_filename)

        # TODO better solution?
        optimizer = get_optimizer(model.parameters(), opti, lr, mom)
        model_path = 'checkpoint_{}.pth.tar'.format(experiment_name)
        print('Loading model checkpointed at epoch {}/{}'.format(initial_epoch,
                                                                 num_epochs), flush=True)
        

        # HERE multiprocessing
        if multiprocessing:
            if torch.cuda.is_available():
                model = torch.nn.DataParallel(model)



        if learn_pos:
            warmup_trainer = FeatTrainerPos(
                model, experiment_name=experiment_name,
                log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
                checkpoint_filename=model_path, for_testing=True)

            warmup_trainer.compile(loss_lesion=criterion_lesion,
                                   optimizer=optimizer)
        elif model_string == 'auto':
            warmup_trainer = FeatTrainerAuto(
                model, experiment_name=experiment_name,
                log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
                checkpoint_filename=model_path, for_testing=True)

            warmup_trainer.compile(loss=criterion_lesion,
                                   optimizer=optimizer)
        elif cat:
            warmup_trainer = FeatTrainerCat(
                model, experiment_name=experiment_name,
                log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
                checkpoint_filename=model_path, print_memory_usage=True, for_testing=True)
            warmup_trainer.compile(loss=criterion_lesion,
                                   optimizer=optimizer)
        else:
            warmup_trainer = FeatTrainer(
                model, experiment_name=experiment_name,
                log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
                checkpoint_filename=model_path, print_memory_usage=True, for_testing=True)

            warmup_trainer.compile(loss=criterion_lesion, optimizer=optimizer)

        batch_size = 32
        if cat:
            batch_size = 16
    #    val_loader = get_valloader_only(toy, rsyncing, batch_size=batch_size, num_workers=4, notebook=False, cat=cat)
            # Load datasets as usual #
        train_loader, val_loader = get_dataloaders(checkpoint_dir, rsyncing=rsyncing, selective_sampling=False,
                                                   warmup_trainer=None,
                                                   batch_size=batch_size, toy=toy,cat=cat)

        print('Start evaluation',flush=True)
        # Start Training #
        warmup_trainer.evaluate(val_loader, 0, for_testing=True)
        warmup_trainer.evaluate(train_loader, 0, for_testing=True)

    else:
        print('For testing, checkpoint {} has to exist'.format(os.path.abspath(checkpoint_filename)),flush=True)

if __name__ == '__main__':
    print('Do not run this file directly. Use run_script.py instead.')

