import os
import torch

from torch import nn

from trainer_qnet import QNetTrainer
from utils.torch_utils import load_checkpoint, get_feature_model, get_optimizer, get_q_model

import models as m
from prepare_qnet import get_val_env_only
from utils.utils import get_nums_from_bbox
from utils.misc_fcts_visualization import save_image_with_orig_plus_current_bb
from utils.config_utils import *
# Need this import vor eval of optimizer
import torch.optim as optim
import matplotlib
import pickle as pkl
import resnet as res

matplotlib.use('agg')


def test(cfg_dict, feat_model_string, rsyncing, toy = False):
    # checkpoint_dir, experiment_name = 'qnet', opti = 'optim.RMSprop', lr = 0.001, mom = 0.1, combi = False
    checkpoint_dir, log_dir, experiment_name = get_q_save_check_tensorB_expName(cfg_dict)
    opti_feat, lr, mom, _ = get_f_train_opti_lr_mom_epochs(cfg_dict)
    inputsize, hiddensize, outputsize = get_q_net_input_hidden_output(cfg_dict)
    _, _, _, double, combi, param_noise, recurrent, hidden_rec, _ , _, _ = get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(
        cfg_dict)
    _, max_steps, replaysize = get_q_hyper_cloneFreq_maxSteps_replaysize(cfg_dict)
    test_tau = get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(cfg_dict)[4]
    cat = get_f_variants_selectiveS_checkPretrained_cat(
        cfg_dict)[2]

    featnet_checkpoint = get_f_save_check_tensorB_expName(cfg_dict)[0]
    print('-------------')
    print('feat_model_string',feat_model_string)
    print('-------------')
    feature_model = get_feature_model(feat_model_string, feat_model_string, load_pretrained=True, opti='optim.Adam',
                                      lr=lr,
                                      mom=mom, checkpoint_pretrained=featnet_checkpoint,cat=cat)

    if feat_model_string == 'auto' or feat_model_string == 'resnet':
        feature_model = res.ResNetFeatures(feature_model)
    else:
        feature_model = m.NetNoDecisionLayer(feature_model)

    if torch.cuda.is_available():
        feature_model.cuda()

    model = get_q_model(combi, recurrent, toy, inputsize, hiddensize, outputsize, feature_model=feature_model, hidden_rec=hidden_rec)
    
    # HERE
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.MSELoss()
    if combi and recurrent <= 0:
        if feat_model_string == 'auto' or feat_model_string == 'resnet':
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
    optimizer = get_optimizer(model_params, opti_feat, lr, mom)

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

        
        if combi and recurrent <= 0:
            if feat_model_string == 'auto' or feat_model_string == 'resnet':
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
        else:
            model_params = model.parameters()
        assert opti_feat in ['optim.Adam', 'optim.SGD', 'optim.RMSprop']
        print('Using optimizer {}'.format(opti_feat))
        if opti_feat == 'optim.Adam':
            optimizer = eval(opti_feat)(model_params, lr=lr)
        else:
            optimizer = eval(opti_feat)(model_params, lr=lr, momentum=mom)
            
        model_path = 'model_best_{}.pth.tar'.format(experiment_name)
        print('Loading model checkpointed at epoch {}'.format(initial_epoch))

        print('Get val env',flush=True)
        val_env = get_val_env_only(cfg_dict, feature_model, rsyncing=rsyncing, toy=toy,f_one=True)
#         val_env = get_val_env_only(cfg_dict, feature_model, rsyncing=rsyncing, toy=toy,f_one=False)
        
        warmup_trainer = QNetTrainer(cfg_dict,
                                 model, val_env, experiment_name=experiment_name,
                                 log_dir='default', checkpoint_dir=checkpoint_dir,
                                 checkpoint_filename=model_path, for_testing=True,
                                 tau_schedule=0, recurrent=recurrent)

        warmup_trainer.compile(loss=criterion, optimizer=optimizer)
        

        print('Evaluate', flush=True)
        val_metrics_arr, trajectory_all_imgs, triggered_all_imgs, Q_s_all_imgs, all_imgs, all_gt, \
        actions_all_imgs = warmup_trainer.evaluate(val_env, 0, save_trajectory=True)

        print('Val_metrics_arr',val_metrics_arr)

        width = 2
        steps_until_detection = []
        print('Save', flush=True)
        if len(trajectory_all_imgs) > 1:
            # print('i:',len(trajectory_all_imgs),flush=True)
            for i, img in enumerate(trajectory_all_imgs):
                
                # print('j:',len(trajectory_all_imgs[i]),flush=True)
                orig_img = all_imgs[i]
                if all_gt[i] is None:
                    orig_r = 0
                    orig_c = 0
                    orig_rn = 0 
                    orig_cn = 0
                else:
                    orig_r, orig_c, orig_rn, orig_cn = get_nums_from_bbox(all_gt[i])
                for j, starts in enumerate(img):
                    # print('k:',len(trajectory_all_imgs[i][j]),flush=True)
                    for k, step in enumerate(starts):
                        r, c, rn, cn = step
                        # print(triggered_all_imgs)
                        # print(Q_s_all_imgs)
                        # print(len(Q_s_all_imgs))
                        # print(len(Q_s_all_imgs[i]))
                        # print(len(Q_s_all_imgs[i][j]))
                        if triggered_all_imgs[i][j] == 1 and k == (len(starts) - 1):
                            steps_until_detection.append(k)
                            if i <10:
                                save_image_with_orig_plus_current_bb(orig_img,
                                                                     '{}/trajectories_test/{}_{}_{}_trigger.png'.format(
                                                                         checkpoint_dir, i, j, k),
                                                                     bbox_flag=True,
                                                                     r=r, c=c,
                                                                     rn=rn, cn=cn, ro=orig_r, co=orig_c,
                                                                     rno=orig_rn,
                                                                     cno=orig_cn, lwidth=width,
                                                                     Q_s=Q_s_all_imgs[i][j][k],
                                                                     eps=-1, action=actions_all_imgs[i][j][k])
                        else:
                            if i <10:
                                save_image_with_orig_plus_current_bb(orig_img,
                                                                     '{}/trajectories_test/{}_{}_{}.png'.format(
                                                                         checkpoint_dir, i, j, k),
                                                                     bbox_flag=True, r=r, c=c, rn=rn, cn=cn,
                                                                     ro=orig_r,
                                                                     co=orig_c, rno=orig_rn, cno=orig_cn,
                                                                     lwidth=width,
                                                                     Q_s=Q_s_all_imgs[i][j][k],
                                                                     eps=-1, action=actions_all_imgs[i][j][k])
        else:
            orig_img = all_imgs[0]
            orig_r, orig_c, orig_rn, orig_cn = get_nums_from_bbox(all_gt[0])
            for j, starts in enumerate(trajectory_all_imgs[0]):
                for k, step in enumerate(starts):
                    r, c, rn, cn = step
                    if triggered_all_imgs[0][j] == 1 and k == (len(starts) - 1):
                        steps_until_detection.append(k)
                        save_image_with_orig_plus_current_bb(orig_img,
                                                             '{}/trajectories_test/0_{}_{}_trigger.png'.format(
                                                                 checkpoint_dir, j, k),
                                                             bbox_flag=True, r=r,
                                                             c=c, rn=rn, cn=cn, ro=orig_r, co=orig_c,
                                                             rno=orig_rn, cno=orig_cn, lwidth=width,
                                                             Q_s=Q_s_all_imgs[0][j][k],
                                                             action=actions_all_imgs[0][j][k])
                    else:
                        save_image_with_orig_plus_current_bb(orig_img,
                                                             '{}/trajectories_test/0_{}_{}_trigger.png'.format(
                                                                 checkpoint_dir, j, k),
                                                             bbox_flag=True, r=r, c=c, rn=rn, cn=cn, ro=orig_r,
                                                             co=orig_c, rno=orig_rn, cno=orig_cn, lwidth=width,
                                                             Q_s=Q_s_all_imgs[0][j][k],
                                                             action=actions_all_imgs[0][j][k])

        pkl.dump(steps_until_detection,
                 open('{}/trajectories_test/steps_until_detection.pkl'.format(checkpoint_dir), 'wb'))
        pkl.dump(val_metrics_arr,
                 open('{}/trajectories_test/val_metrics_arr.pkl'.format(checkpoint_dir), 'wb'))
    else:
        print('For testing, checkpoint filename has to exist')
