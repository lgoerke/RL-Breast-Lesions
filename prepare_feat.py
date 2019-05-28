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

def get_weights(checkpoint_dir, dataloader, training=True, size=1500):
    """

    :param size:
    :param checkpoint_dir:
    :param dataloader:
    :param training:
    :return:
    """
    mode_string = 'train' if training else 'val'
    if os.path.exists(os.path.join(checkpoint_dir, 'weights_balanced_{}.pkl'.format(mode_string))):
        print('Loading saved weights', flush=True)
        weights = pkl.load(open(os.path.join(
            checkpoint_dir, 'weights_balanced_{}.pkl'.format(mode_string)), 'rb'))
    elif mode_string == 'val':
        weights = u.get_balanced_weights_fixed_size(dataloader, size)
        pkl.dump(weights, open(os.path.join(checkpoint_dir,
                                            'weights_balanced_{}.pkl'.format(mode_string)), 'wb'))
    else:
        weights = u.get_balanced_weights(dataloader)
        pkl.dump(weights, open(os.path.join(checkpoint_dir,
                                            'weights_balanced_{}.pkl'.format(mode_string)), 'wb'))
    return weights


def get_sequential_trainloader(toy, rsyncing, batch_size=16, num_workers=os.cpu_count() - 1,
                               data_aug_vec=[0.5, 0.25, 0.5, 0.5], notebook=False):
    """

    :param toy:
    :param rsyncing:
    :param batch_size:
    :param num_workers:
    :param data_aug_vec:
    :param notebook:
    :return:
    """
    num_workers = 0
    if rsyncing:
        print('Rsynced data! (prepare feat)', flush=True)
    else:
        print('Using symbolic links! (prepare feat)', flush=True)
    print('Getting path ready..', flush=True)
    anno_path_train, _, png_path = get_paths(rsyncing, toy, notebook)

    # TODO
    # png_path = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset', 'png')
    # anno_path_train = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset',
    #                                'annotations/mscoco_train_full.json')

    trans_img = torchvision.transforms.Compose([t.Normalize(),t.BboxCrop(targetsize=224), t.RandomFlipImg(prob=data_aug_vec[0]),
                                                t.RandomGammaImg(prob=data_aug_vec[1], use_normal_distribution=True)])
    trans_bb = torchvision.transforms.Compose(
        [t.GetFiveBBs(), t.RandomTranslateBB(prob=data_aug_vec[2], pixel_range=10),
         t.RandomScaleBB(prob=data_aug_vec[3], max_percentage=0.1)])

    start_time = time.time()
    print('Creating Coco Dataset..', flush=True)

    trainset = u.dataset_coco(
        png_path, anno_path_train, transform=trans_img, bbox_transform=trans_bb,for_feature=True)
    print('Training set has', len(trainset), 'images', flush=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SequentialSampler(trainset),
                                              num_workers=num_workers, collate_fn=u.mammo_collate)
    print('Training loader has', len(trainloader), 'batches', flush=True)

    total_time = time.time() - start_time
    print('Creating Datasets took {:.0f} seconds.'.format(
        total_time), flush=True)

    return trainloader


def get_valloader_only(toy, rsyncing, batch_size=16, num_workers=os.cpu_count() - 1, notebook=False,cat=False):
    """

    :param toy:
    :param rsyncing:
    :param batch_size:
    :param num_workers:
    :param notebook:
    :return:
    """
    num_workers = 0
    if rsyncing:
        print('Rsynced data! (prepare feat)', flush=True)
    else:
        print('Using symbolic links! (prepare feat)', flush=True)
    print('Getting path ready..', flush=True)
    _, anno_path_val, png_path = get_paths(rsyncing, toy, notebook)
    
    start_time = time.time()
    print('Creating Coco Dataset..', flush=True)

    if not cat:
        valset = u.dataset_coco(png_path, anno_path_val,
                            transform=torchvision.transforms.Compose([t.Normalize(), t.BboxCrop(targetsize=224)]),bbox_transform=torchvision.transforms.Compose([t.GetFiveBBs()]),for_feature=True,cat=cat)
    else:
        valset = u.dataset_coco(png_path, anno_path_val,
                            transform=torchvision.transforms.Compose([t.Normalize(), t.BboxCropMult(targetsize=224)]),bbox_transform=torchvision.transforms.Compose([t.GetBBsMult()]),for_feature=True,cat=cat)
    print('Validation set has', len(valset), 'images', flush=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=SequentialSampler(valset),
                                            num_workers=num_workers, collate_fn=u.mammo_collate)
    print('Validation loader has', len(valloader), 'batches', flush=True)

    total_time = time.time() - start_time
    print('Creating Datasets took {:.0f} seconds.'.format(
        total_time), flush=True)
    return valloader


def get_dataloaders(checkpoint_dir, rsyncing, selective_sampling=False, warmup_trainer=None, batch_size=16,
                    num_workers=os.cpu_count() - 1, data_aug_vec=[0.5, 0.25, 0.5, 0.5], toy=False,
                    notebook=False,cat=False):
    """

    :param checkpoint_dir:
    :param rsyncing:
    :param selective_sampling:
    :param warmup_trainer:
    :param batch_size:
    :param num_workers:
    :param seed:
    :param data_aug_vec: probabilities for rnd flip, rnd gamma, rnd translation and rnd scale
    :param toy:
    :param notebook:
    :return:
    """
#     if torch.cuda.is_available():
#         mp.set_start_method('spawn')
    multiprocessing = False
    num_workers = 0
    sampler_size = 3000

    if rsyncing:
        print('Rsynced data! (prepare feat)', flush=True)
    else:
        print('Using symbolic links! (prepare feat)', flush=True)
    print('Getting path ready..', flush=True)
    anno_path_train, anno_path_val, png_path = get_paths(
        rsyncing, toy, notebook)

    # TODO
    # png_path = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset', 'png')
    # anno_path_train = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset',
    #                                'annotations/mscoco_train_full.json')
    # anno_path_val = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset',
    #                                'annotations/mscoco_train_full.json')

    print('Creating Coco Datasets..', flush=True)
    # t.ToTensor()
    if not cat:
        trans_img = torchvision.transforms.Compose([t.Normalize(),t.BboxCrop(targetsize=224), t.RandomFlipImg(prob=data_aug_vec[0]),
                                                    t.RandomGammaImg(prob=data_aug_vec[1], use_normal_distribution=True)])
        trans_bb = torchvision.transforms.Compose(
            [t.GetFiveBBs(), t.RandomTranslateBB(prob=data_aug_vec[2], pixel_range=10),
             t.RandomScaleBB(prob=data_aug_vec[3], max_percentage=0.1)])
    else:
        trans_img = torchvision.transforms.Compose([t.Normalize(),t.BboxCropMult(targetsize=224), t.RandomFlipImg(prob=data_aug_vec[0]),
                                                    t.RandomGammaImg(prob=data_aug_vec[1], use_normal_distribution=True)])
        trans_bb = torchvision.transforms.Compose(
            [t.GetBBsMult(), t.RandomTranslateBB(prob=data_aug_vec[2], pixel_range=10,cat=True),
             t.RandomScaleBB(prob=data_aug_vec[3], max_percentage=0.1,cat=True)])
    
    
    

    trainset = u.dataset_coco(
        png_path, anno_path_train, transform=trans_img, bbox_transform=trans_bb,for_feature=True,cat=cat)
    print('Training set has', len(trainset), 'images', flush=True)

    if not cat:
        valset = u.dataset_coco(png_path, anno_path_val,
                            transform=torchvision.transforms.Compose([t.Normalize(), t.BboxCrop(targetsize=224)]),bbox_transform=torchvision.transforms.Compose([t.GetFiveBBs()]),for_feature=True,cat=cat)
    else:
        valset = u.dataset_coco(png_path, anno_path_val,
                            transform=torchvision.transforms.Compose([t.Normalize(), t.BboxCropMult(targetsize=224)]),bbox_transform=torchvision.transforms.Compose([t.GetBBsMult()]),for_feature=True,cat=cat)
    print('Validation set has', len(valset), 'images', flush=True)

    
    
    if selective_sampling:
        if not warmup_trainer:
            print('Cannot calculate weights for selective sampling: no model given. Using normal sampling instead',
                  flush=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=RandomSampler(trainset),
                                              num_workers=num_workers, collate_fn=u.mammo_collate,
                                              pin_memory=multiprocessing)
        else:
            print('Getting weights for sampling..', flush=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SequentialSampler(trainset),
                                              num_workers=num_workers, collate_fn=u.mammo_collate,
                                              pin_memory=multiprocessing)
            weights = warmup_trainer.predict_dataset(trainloader)
            pkl.dump(weights, open(os.path.join(
                checkpoint_dir, 'weights_selective_train.pkl'), 'wb'))
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=WeightedRandomSampler(
                                                  weights.double(), sampler_size, replacement=False),
                                              num_workers=num_workers, collate_fn=u.mammo_collate,
                                              pin_memory=multiprocessing)
            
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=RandomSampler(trainset),
                                              num_workers=num_workers, collate_fn=u.mammo_collate,
                                              pin_memory=multiprocessing)
         
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            sampler=SequentialSampler(valset),
                                            num_workers=num_workers,
                                            collate_fn=u.mammo_collate, pin_memory=multiprocessing)
    
    print('Training loader has', len(trainloader), 'batches', flush=True)
    print('Validation loader has', len(valloader), 'batches', flush=True)
    return trainloader, valloader

def prepare_and_start_training(cfg_dict, model_string, rsyncing, learn_pos=False, toy=False, auto=False):
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
    if resume and os.path.exists(checkpoint_filename):
        # Don't load optimizer, otherwise LR might be too low already (yes?
        # TODO)
        model, _, initial_epoch = load_checkpoint(
            model, optimizer, filename=checkpoint_filename)

        # TODO better solution?
        optimizer = get_optimizer(model.parameters(), opti, lr, mom)
        model_path = 'checkpoint_{}.pth.tar'.format(experiment_name)
        print('Loading model checkpointed at epoch {}/{}'.format(initial_epoch,
                                                                 num_epochs), flush=True)
    else:
        initial_epoch = 0
        model_path = 'checkpoint_{}.pth.tar'.format(experiment_name)

    # HERE multiprocessing
    if multiprocessing:
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)


        
    if learn_pos:
        warmup_trainer = FeatTrainerPos(
            model, experiment_name=experiment_name,
            log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
            checkpoint_filename=model_path)

        warmup_trainer.compile(loss_lesion=criterion_lesion,
                               optimizer=optimizer)
    elif model_string == 'auto':
        warmup_trainer = FeatTrainerAuto(
            model, experiment_name=experiment_name,
            log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
            checkpoint_filename=model_path)

        warmup_trainer.compile(loss=criterion_lesion,
                               optimizer=optimizer)
    elif cat:
        warmup_trainer = FeatTrainerCat(
            model, experiment_name=experiment_name,
            log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
            checkpoint_filename=model_path, print_memory_usage=True)
        warmup_trainer.compile(loss=criterion_lesion,
                               optimizer=optimizer)
    else:
        warmup_trainer = FeatTrainer(
            model, experiment_name=experiment_name,
            log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
            checkpoint_filename=model_path, print_memory_usage=True)

        warmup_trainer.compile(loss=criterion_lesion, optimizer=optimizer)

    batch_size = 32
    if cat:
        batch_size = 16
    if selective_sampling:
        # Selective Sampling #
        train_loader, val_loader = get_dataloaders(checkpoint_dir, rsyncing=rsyncing, selective_sampling=True,
                                                   warmup_trainer=warmup_trainer, batch_size=batch_size, toy=toy, cat=cat)
    else:
        # Load datasets as usual #
        train_loader, val_loader = get_dataloaders(checkpoint_dir, rsyncing=rsyncing, selective_sampling=False,
                                                   warmup_trainer=None,
                                                   batch_size=batch_size, toy=toy,cat=cat)

    # Start Training #
    warmup_trainer.train(
        train_loader, val_loader=val_loader, initial_epoch=initial_epoch, num_epochs=num_epochs)

    best_model_path = os.path.join(
        checkpoint_dir, 'model_best_{}.pth.tar'.format(experiment_name))
    warmup_model_path = os.path.join(
        checkpoint_dir, 'warmup_model_{}.pth.tar'.format(experiment_name))
    shutil.move(best_model_path, warmup_model_path)
    # os.remove(best_model_path)


if __name__ == '__main__':
    print('Do not run this file directly. Use run_script.py instead.')
