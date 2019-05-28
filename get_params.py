import os

import time
import torch
import shutil

from torch import nn
from trainer_qnet import QNetTrainer
from config import cfg
from utils.utils import get_paths
import utils.datasets as u
from utils.torch_utils import *

import models as m
import resnet as res
from utils.config_utils import *
import torch.multiprocessing as multiprocessing
def get_params(cfg_dict, feat_model_string, rsyncing, toy=False):

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

    pytorch_total_params = sum(p.numel() for p in feature_model.parameters() if p.requires_grad)
    print('Featmodel (whole)',pytorch_total_params)
    
    if feat_experiment_name == 'auto' or feat_experiment_name == 'resnet':
        feature_model = res.ResNetFeatures(feature_model)
    else:
        feature_model = m.NetNoDecisionLayer(feature_model)

    if torch.cuda.is_available():
        feature_model.cuda()

    if feat_experiment_name == 'simple':
        is_simple= True
    else:
        is_simple = False
        
    pytorch_total_params = sum(p.numel() for p in feature_model.parameters() if p.requires_grad)
    print('Featmodel (not decision)',pytorch_total_params)
    
    model = get_q_model(combi, recurrent, toy, inputsize, hiddensize, outputsize, feature_model=feature_model, hidden_rec=hidden_rec,cat=cat,simple=is_simple)

    # HERE
    if torch.cuda.is_available():
        model.cuda()


    if clip_val:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    if combi and recurrent <= 0:
        if feat_experiment_name == 'auto' or feat_experiment_name == 'resnet':
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
    
#     print('That is the model:',model, flush=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Q-Model',pytorch_total_params)