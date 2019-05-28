import os
import time
import torch
import subprocess

from config import cfg_from_file

from torch import nn

from trainer_qnet import QNetTrainer
from config import cfg
from utils.torch_utils import *

import models as m
import pickle as pkl

from prepare_qnet import get_val_env_only

from utils.config_utils import *
from bootstrap import parse_args
import matplotlib

matplotlib.use('agg')


def plot_fct(cfg_dict, rsyncing, qstar=True ,toy=False):
    opti, lr, mom, num_epochs, transitions_per_learning = get_q_train_opti_lr_mom_epochs_transPerUp(cfg_dict)
    checkpoint_dir, log_dir, experiment_name = get_q_save_check_tensorB_expName(cfg_dict)
    opti_feat, lr, mom, _ = get_f_train_opti_lr_mom_epochs(cfg_dict)
    inputsize, hiddensize, outputsize = get_q_net_input_hidden_output(cfg_dict)
    _, _, _, double, combi, param_noise, recurrent, hidden_rec, _ , _, _ = get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(
        cfg_dict)
    _, max_steps, replaysize = get_q_hyper_cloneFreq_maxSteps_replaysize(cfg_dict)
    test_tau = get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(cfg_dict)[4]
    feat_experiment_name = get_f_save_check_tensorB_expName(cfg)[2]
    
    gamma, _, _, _ , clip_val = get_q_rewards_gamma_zeta_eta_iota_clipVal(cfg_dict)

    feat_checkpoint_dir = get_f_save_check_tensorB_expName(cfg_dict)[0]
    print('-------------')
    print('feat_experiment_name',feat_experiment_name)
    print('-------------')
    # Load feature model and get environments #
    cat = get_f_variants_selectiveS_checkPretrained_cat(cfg_dict)[2]

    feature_model = get_feature_model(feat_experiment_name, feat_experiment_name, load_pretrained=True, opti='optim.Adam', lr=lr,
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

    # checkpoint_filename = os.path.join(checkpoint_dir, 'checkpoint_{}.pth.tar'.format(experiment_name))
#     checkpoint_filename = os.path.join(checkpoint_dir, 'model_best_{}.pth.tar'.format(experiment_name))
    checkpoint_filename = os.path.join(checkpoint_dir, 'warmup_model_{}.pth.tar'.format(experiment_name))
    print('Load checkpoint from {}'.format(os.path.abspath(checkpoint_filename)))
    ######
    # TODO this if else should be before
    if os.path.exists(checkpoint_filename):
        if not os.path.isdir('{}/trajectories_test'.format(checkpoint_dir)):
            os.makedirs('{}/trajectories_test'.format(checkpoint_dir))

        # Don't load optimizer, otherwise LR might be too low already (yes? TODO)
        model, _, initial_epoch = load_checkpoint(model, optimizer, filename=checkpoint_filename)

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
            
        model_path = 'model_best_{}.pth.tar'.format(experiment_name)
        print('Loading model checkpointed at epoch {}'.format(initial_epoch))

        print('Get val env',flush=True)
        val_env = get_val_env_only(cfg_dict, feature_model, rsyncing=rsyncing, toy=toy)
        
        warmup_trainer = QNetTrainer(cfg_dict,
                                 model, val_env, experiment_name=experiment_name,
                                 log_dir='default', checkpoint_dir=checkpoint_dir,
                                 checkpoint_filename=model_path,
                                 tau_schedule=0, recurrent=recurrent)

        warmup_trainer.compile(loss=criterion, optimizer=optimizer)


        no_imgs = 10

        if qstar:
            print('Getting q fct', flush=True)
            start_time = time.time()
            return_dict = warmup_trainer.get_q_fct(val_env, no_imgs)
            total_time = time.time() - start_time
            print('Getting q fct took {:.0f} seconds.'.format(total_time), flush=True)
            print('Save at {}/qstar_map.pkl'.format(checkpoint_dir), flush=True)
            pkl.dump(return_dict, open('{}/qstar_map.pkl'.format(checkpoint_dir), 'wb'))
        else:
            print('Getting activation fct', flush=True)
            start_time = time.time()
            return_dict = warmup_trainer.get_activation_fcn(val_env, no_imgs)
            total_time = time.time() - start_time
            print('Getting activation fct took {:.0f} seconds.'.format(total_time), flush=True)
            print('Save at {}/activation_map.pkl'.format(checkpoint_dir), flush=True)
            pkl.dump(return_dict, open('{}/activation_map.pkl'.format(checkpoint_dir), 'wb'))

        # for idx in range(no_imgs):  
        #     figures_all_tile_sizes,rows_all_tile_sizes,cols_all_tile_sizes,row_lims,
        #     col_lims, spacings, original_img = return_dict[idx]

        #     print('Len Figure all tile size',len(figures_all_tile_sizes))
        #     print('[0]',figures_all_tile_sizes[0].shape) #7
        #     print('[1]',figures_all_tile_sizes[1].shape) #15
        #     print('[2]',figures_all_tile_sizes[2].shape) #39
        #     print('[3]',figures_all_tile_sizes[3].shape) #81
        #     print('[4]',figures_all_tile_sizes[4].shape) #163
        #     print('Len row_lims',len(row_lims))
        #     print('row_lims',row_lims)
        #     print('Len col_lims',len(col_lims))
        #     print('col_lims',col_lims)
        #     print('Len spacings',len(spacings))
        #     print('spacings',spacings)
        #     for j,entry in enumerate(figures_all_tile_sizes):
        #         save_image_with_q_star('test_{}.png'.format(idx),entry,
        #         row_lims[j], col_lims[j],spacings[j],original_img)

    else:
        print('For testing, checkpoint filename {} has to exist'.format(os.path.abspath(checkpoint_filename)))


def main():
    start_time = time.time()
    args = parse_args()
    if args.toy:
        paths = [
            '/mnt/synology/breast/projects/lisa/toy_data/',
            '/input'
        ]
    else:
        paths = [
            '/mnt/synology/breast/archives/screenpoint3/processed_dataset/',
            '/input'
        ]

    if args.no_rsync:
        rsyncing = False
        print('Working with symbolic links', flush=True)
        data_cmd = ['ln', '-s']
    else:
        rsyncing = True
        print('Rsyncing data', flush=True)
        data_cmd = ['/usr/bin/rsync', '-am', '--stats']

    subprocess.call(data_cmd + paths)
    print('Preparing dataset took {:.2f} seconds.'.format(time.time() - start_time), flush=True)

    # HERE
    qstar = True
    

    # Benchmark best convolution algorithm
    from torch.backends import cudnn

    cudnn.benchmark = True

    if args.cfg_file:
        print('Loading config {}.'.format(args.cfg_file))
        cfg_from_file(args.cfg_file)
    else:
        print('No config given, using standard settings.')

    plot_fct(cfg, rsyncing, qstar=qstar, toy=args.toy)


if __name__ == '__main__': main()
