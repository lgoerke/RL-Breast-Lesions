import os

import time
import torch
import shutil

from torch import nn
from trainer_qnet import QNetTrainer
from config import cfg
from utils.utils import get_paths
import utils.datasets as u
from utils.torch_utils import load_checkpoint, get_feature_model, get_optimizer, get_q_model

import models as m
import resnet as res
from utils.config_utils import *
import torch.multiprocessing as multiprocessing

def get_val_env_only(cfg_dict, model, rsyncing, toy=False, f_one=False):
    if cfg_dict.EXPERIMENTAL_ENV:
        from environment_exp import MammoEnv
    else:
        from environment import MammoEnv
    
    one_img, max_num_imgs_train, max_num_imgs_val, _, _, _, _, _ , dist_reward, dist_factor, with_hist = get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(
        cfg_dict)
    _, zeta, eta, iota, _ = get_q_rewards_gamma_zeta_eta_iota_clipVal(cfg_dict)
    tau = get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(cfg_dict)[3]

    print('Getting path ready..', flush=True)
    _, anno_path_val, png_path = get_paths(rsyncing, toy)
    print('Creating Coco Datasets..', flush=True)
    if one_img:
        # png_path = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset_small', 'png')
        # anno_path_train = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset_small',
        #                                'annotations/mscoco_train_full.json')

        # png_path = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset', 'png')
        # anno_path_train = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset',
        #                                'annotations/mscoco_train_full.json')

        png_path = os.path.join('../one_img_dataset', 'png')
        anno_path_train = os.path.join('../one_img_dataset',
                                       'annotations/mscoco_train_full.json')

        # TODO
        # png_path = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset', 'png')
        # anno_path_train = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset',
        #                                'annotations/mscoco_train_full.json')

        print(os.path.abspath(anno_path_train), flush=True)

        trainset = u.dataset_coco(png_path, anno_path_train, add_border=get_addBorder(cfg),f_one=f_one)
        print('Training set has', len(trainset), 'images', flush=True)
        val_env = MammoEnv(trainset, eta=eta, zeta=zeta, tau=tau, model=model, one_img=one_img,
                           max_no_imgs=max_num_imgs_val, iota=iota,dist_reward = dist_reward, dist_factor=dist_factor, with_hist=with_hist,f_one=f_one)
    else:
        valset = u.dataset_coco(png_path, anno_path_val, add_border=get_addBorder(cfg),f_one=f_one)
        print('Validation set has', len(valset), 'images')
        val_env = MammoEnv(valset, eta=eta, zeta=zeta, tau=tau, model=model, one_img=one_img,
                           max_no_imgs=max_num_imgs_val, iota=iota,dist_reward = dist_reward, dist_factor=dist_factor,with_hist=with_hist,f_one=f_one)
    return val_env


def get_envs(cfg_dict, model, rsyncing, toy=False,f_one=False):
    if cfg_dict.EXPERIMENTAL_ENV:
        from environment_exp import MammoEnv
    else:
        from environment import MammoEnv
    one_img, max_num_imgs_train, max_num_imgs_val, _, _, _, _, _, dist_reward, dist_factor, with_hist = get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(
        cfg_dict)
    _, zeta, eta, iota, _ = get_q_rewards_gamma_zeta_eta_iota_clipVal(cfg_dict)
    tau = get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(cfg_dict)[3]

    print('Getting path ready..', flush=True)
    anno_path_train, anno_path_val, png_path = get_paths(rsyncing, toy)
    print('Creating Coco Datasets..', flush=True)

    if one_img:
        # png_path = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset_small', 'png')
        # anno_path_train = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset_small',
        #                                'annotations/mscoco_train_full.json')

        # png_path = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset', 'png')
        # anno_path_train = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset',
        #                                'annotations/mscoco_train_full.json')

        png_path = os.path.join('../one_img_dataset', 'png')
        anno_path_train = os.path.join('../one_img_dataset',
                                       'annotations/mscoco_train_full.json')

        # TODO
        # png_path = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset', 'png')
        # anno_path_train = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset',
        #                                'annotations/mscoco_train_full.json')

        trainset = u.dataset_coco(png_path, anno_path_train, add_border=get_addBorder(cfg))
        train_env = MammoEnv(trainset, eta=eta, zeta=zeta, tau=tau, model=model, one_img=one_img,
                             max_no_imgs=max_num_imgs_train, iota=iota,dist_reward = dist_reward, dist_factor=dist_factor, with_hist=with_hist)
        val_env = MammoEnv(trainset, eta=eta, zeta=zeta, tau=tau, model=model, one_img=one_img,
                           max_no_imgs=max_num_imgs_val, iota=iota,dist_reward = dist_reward, dist_factor=dist_factor, with_hist=with_hist)
    else:
        # TODO transform for downsampling
        print('Building training set', flush=True)
        trainset = u.dataset_coco(png_path, anno_path_train, add_border=get_addBorder(cfg))
        print('Training set has', len(trainset), 'images', flush=True)

        print('Building validation set', flush=True)
        valset = u.dataset_coco(png_path, anno_path_val, add_border=get_addBorder(cfg),f_one=f_one)
        print('Validation set has', len(valset), 'images', flush=True)

        print('Make train env', flush=True)
        train_env = MammoEnv(trainset, eta=eta, zeta=zeta, tau=tau, model=model, one_img=one_img,
                             max_no_imgs=max_num_imgs_train, iota=iota,dist_reward = dist_reward, dist_factor=dist_factor,with_hist=with_hist)
        print('Make val env', flush=True)
        val_env = MammoEnv(valset, eta=eta, zeta=zeta, tau=tau, model=model, one_img=one_img,
                           max_no_imgs=max_num_imgs_val, iota=iota,dist_reward = dist_reward, dist_factor=dist_factor,with_hist=with_hist,f_one=f_one)
    return train_env, val_env


def prepare_and_start_training(cfg_dict, feat_model_string, rsyncing, toy=False):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Retrieve params Q
    opti, lr, mom, num_epochs, transitions_per_learning = get_q_train_opti_lr_mom_epochs_transPerUp(cfg_dict)
    checkpoint_dir, tensorboard_dir, experiment_name = get_q_save_check_tensorB_expName(cfg_dict)
    _, _, _, double, combi, param_noise, recurrent, hidden_rec, _, _, _ = get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(
        cfg_dict)
    _, resume, lr_schedule, _ = get_seed_resume_lrSchedule_root(cfg_dict)
    inputsize, hiddensize, outputsize = get_q_net_input_hidden_output(cfg_dict)
    kappa, decreasing_eps, target_eps, tau, test_tau, tau_schedule_epochs = get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(
        cfg_dict)
    clone_freq, max_steps, replaysize = get_q_hyper_cloneFreq_maxSteps_replaysize(cfg_dict)
    gamma, _, _, _ , clip_val = get_q_rewards_gamma_zeta_eta_iota_clipVal(cfg_dict)

    # Retrieve params Feat
    feat_checkpoint_dir, _, feat_experiment_name = get_f_save_check_tensorB_expName(cfg_dict)
    opti_feat = get_f_train_opti_lr_mom_epochs(cfg_dict)[0]

    # Load feature model and get environments #
    cat = get_f_variants_selectiveS_checkPretrained_cat(cfg_dict)[2]

    feature_model = get_feature_model(feat_model_string, feat_experiment_name, load_pretrained=True, opti='optim.Adam', lr=lr,
                                      mom=mom,
                                      checkpoint_pretrained=feat_checkpoint_dir,cat=cat)

    if feat_experiment_name == 'auto' or feat_experiment_name == 'resnet':
        feature_model = res.ResNetFeatures(feature_model)
    elif feat_experiment_name == 'resnet_pool':
        feature_model = res.ResNetFeaturesPool(feature_model)
    else:
        feature_model = m.NetNoDecisionLayer(feature_model)

    if torch.cuda.is_available():
        feature_model.cuda()

    is_simple =False
    with_pool = False
    if feat_experiment_name == 'simple':
        is_simple= True
    elif feat_experiment_name == 'resnet_pool':
        with_pool= True
   
    model = get_q_model(combi, recurrent, toy, inputsize, hiddensize, outputsize, feature_model=feature_model, hidden_rec=hidden_rec,cat=cat,simple=is_simple,with_pool=with_pool)

    # HERE
    if torch.cuda.is_available():
        model.cuda()

    
    if clip_val:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    if combi and recurrent <= 0:
        if feat_experiment_name == 'auto' or feat_experiment_name == 'resnet' or feat_experiment_name == 'resnet_pool':
            model_params = [
                {'params': model.conv1.parameters(), 'lr': lr / 10},
                {'params': model.bn1.parameters(), 'lr': lr / 10},
                {'params': model.relu.parameters(), 'lr': lr / 10},
                {'params': model.maxpool.parameters(), 'lr': lr / 10},
                {'params': model.layer1.parameters(), 'lr': lr / 10},
                {'params': model.layer2.parameters(), 'lr': lr / 10},
                {'params': model.layer3.parameters(), 'lr': lr / 10},
                {'params': model.layer4.parameters(), 'lr': lr / 10},
                {'params': model.qnet.parameters()}
            ]
        else:
            model_params = [
                {'params': model.features.parameters(), 'lr': lr / 10},
                {'params': model.qnet.parameters()}
            ]
    elif combi and recurrent > 0:
        if toy:
            model_params = [
                {'params': model.features.parameters(), 'lr': lr / 10},
                {'params': model.ll1.parameters()},
                {'params': model.ll2.parameters()},
                {'params': model.ll3.parameters()},
                {'params': model.relu2.parameters()},
                {'params': model.lstm.parameters()},
            ]
        else:
            model_params = [
                {'params': model.conv1.parameters(), 'lr': lr / 10},
                {'params': model.bn1.parameters(), 'lr': lr / 10},
                {'params': model.relu.parameters(), 'lr': lr / 10},
                {'params': model.maxpool.parameters(), 'lr': lr / 10},
                {'params': model.layer1.parameters(), 'lr': lr / 10},
                {'params': model.layer2.parameters(), 'lr': lr / 10},
                {'params': model.layer3.parameters(), 'lr': lr / 10},
                {'params': model.layer4.parameters(), 'lr': lr / 10},
                {'params': model.ll1.parameters()},
                {'params': model.ll2.parameters()},
                {'params': model.ll3.parameters()},
                {'params': model.relu2.parameters()},
                {'params': model.lstm.parameters()},
            ]
    else:
        model_params = model.parameters()
    optimizer = get_optimizer(model_params, opti, lr, mom)

    print(model, flush=True)

#     checkpoint_filename = os.path.join(checkpoint_dir, 'warmup_model_{}.pth.tar'.format(experiment_name))
    checkpoint_filename = os.path.join(checkpoint_dir, 'model_best_{}.pth.tar'.format(experiment_name))
    ######
    if resume and os.path.exists(checkpoint_filename):
        print('======',flush=True)
        # Don't load optimizer, otherwise LR might be too low already (yes? TODO)
        model, _, initial_epoch, replay_mem = load_checkpoint(model, optimizer, filename=checkpoint_filename, load_mem=True, checkpoint_dir=checkpoint_dir,experiment_name=experiment_name)

        # TODO better solution?
        if clip_val:
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        if combi and recurrent <= 0:
            if feat_experiment_name == 'auto' or feat_experiment_name == 'resnet' or feat_experiment_name == 'resnet_pool':
                model_params = [
                    {'params': model.conv1.parameters(), 'lr': lr / 10},
                    {'params': model.bn1.parameters(), 'lr': lr / 10},
                    {'params': model.relu.parameters(), 'lr': lr / 10},
                    {'params': model.maxpool.parameters(), 'lr': lr / 10},
                    {'params': model.layer1.parameters(), 'lr': lr / 10},
                    {'params': model.layer2.parameters(), 'lr': lr / 10},
                    {'params': model.layer3.parameters(), 'lr': lr / 10},
                    {'params': model.layer4.parameters(), 'lr': lr / 10},
                    {'params': model.qnet.parameters()}
                ]
            else:
                model_params = [
                    {'params': model.features.parameters(), 'lr': lr / 10},
                    {'params': model.qnet.parameters()}
                ]
        elif combi and recurrent > 0:
            if toy:
                model_params = [
                    {'params': model.features.parameters(), 'lr': lr / 10},
                    {'params': model.ll1.parameters()},
                    {'params': model.ll2.parameters()},
                    {'params': model.ll3.parameters()},
                    {'params': model.relu2.parameters()},
                    {'params': model.lstm.parameters()},
                ]
            else:
                model_params = [
                    {'params': model.conv1.parameters(), 'lr': lr / 10},
                    {'params': model.bn1.parameters(), 'lr': lr / 10},
                    {'params': model.relu.parameters(), 'lr': lr / 10},
                    {'params': model.maxpool.parameters(), 'lr': lr / 10},
                    {'params': model.layer1.parameters(), 'lr': lr / 10},
                    {'params': model.layer2.parameters(), 'lr': lr / 10},
                    {'params': model.layer3.parameters(), 'lr': lr / 10},
                    {'params': model.layer4.parameters(), 'lr': lr / 10},
                    {'params': model.ll1.parameters()},
                    {'params': model.ll2.parameters()},
                    {'params': model.ll3.parameters()},
                    {'params': model.relu2.parameters()},
                    {'params': model.lstm.parameters()},
                ]
        else:
            model_params = model.parameters()
        
        optimizer = get_optimizer(model_params, opti, lr, mom)
        
        model_path = 'checkpoint_{}.pth.tar'.format(experiment_name)
        print('Loading model checkpointed at epoch {}/{}'.format(initial_epoch, num_epochs))
        print('======',flush=True)
    else:
        print('======',flush=True)
        print('First things first',flush=True)
        model_path = 'checkpoint_{}.pth.tar'.format(experiment_name)
        replay_mem = None
        initial_epoch = 1
        print('======',flush=True)
    
    # Always start training from beginning
#     initial_epoch = 1
    

    if combi:
        train_env, val_env = get_envs(cfg_dict, model, rsyncing, toy, f_one=True)
    else:
        train_env, val_env = get_envs(cfg_dict, feature_model, rsyncing, toy, f_one=True)

    warmup_trainer = QNetTrainer(cfg_dict,
                                 model, train_env, experiment_name=experiment_name,
                                 log_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
                                 checkpoint_filename=model_path,
                                 tau_schedule=tau_schedule_epochs, recurrent=recurrent, replay_mem=replay_mem)

    warmup_trainer.compile(loss=criterion, optimizer=optimizer)

    # Start Training #
    warmup_trainer.train(
        val_env=val_env, initial_epoch=initial_epoch, num_epochs=num_epochs, decreasing_eps=decreasing_eps,
        target_eps=target_eps, transitions_per_learning=transitions_per_learning,
        lr_schedule=lr_schedule)

    best_model_path = os.path.join(warmup_trainer.checkpoint_dir, 'model_best_{}.pth.tar'.format(experiment_name))
    warmup_model_path = os.path.join(warmup_trainer.checkpoint_dir, 'warmup_model_{}.pth.tar'.format(experiment_name))
    shutil.move(best_model_path, warmup_model_path)
    # os.remove(best_model_path)


if __name__ == '__main__':
    print('Do not run this file directly. Use run_script.py instead.')
