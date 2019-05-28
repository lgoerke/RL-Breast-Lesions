import os
import pdb
import sys
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, precision_recall_fscore_support

from utils.dql_utils import get_attribute_from_list, string_action
import utils.replaymemory as replay
from utils.replaymemory import ReplayMemory
from utils.torch_utils import var_to_numpy, tensor_to_var, numpy_to_var, save_checkpoint_and_best, \
    numpy_to_tensor, create_copy
import gc
import copy
import models as m
import resnet as r
from utils.misc_fcts_visualization import save_image_with_orig_plus_current_bb
from utils.config_utils import *
from utils.utils import get_nums_from_bbox, TimeTracker, convert_bbs_to_points, \
    print_current_time_epoch, get_gc_size, memory_usage, get_memory_usage
from scipy.spatial import distance
import collections

class QNetTrainer(object):
    def __init__(self, cfg_dict, model, train_env, log_dir='', checkpoint_dir='', checkpoint_filename=None,
                 experiment_name='qnet', for_testing=False, tau_schedule=0, recurrent=0, replay_mem = None):
        """

        :param train_env:
        :param model:
        :param log_dir:
        :param checkpoint_dir:
        :param checkpoint_filename:
        :param experiment_name:
        :param for_testing:
        :param tau_schedule:
        :param recurrent:
        """
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must inherit from torch.nn.Module')

        self.cfg_dict = cfg_dict

        self.tau_schedule = tau_schedule
        self.recurrent = recurrent

        self.max_steps = get_q_hyper_cloneFreq_maxSteps_replaysize(self.cfg_dict)[1]
        self.gamma = get_q_rewards_gamma_zeta_eta_iota_clipVal(self.cfg_dict)[0]
        self.clip_val = get_q_rewards_gamma_zeta_eta_iota_clipVal(self.cfg_dict)[4]
        self.clone_freq = get_q_hyper_cloneFreq_maxSteps_replaysize(self.cfg_dict)[0]
        self.kappa = get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(self.cfg_dict)[0]
        self.test_tau = get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(self.cfg_dict)[4]
        self.double = \
            get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(self.cfg_dict)[3]
        self.combi = \
        get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(self.cfg_dict)[4]
        self.with_hist = \
            get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(self.cfg_dict)[-1]
        self.model = model

        self.optimizer = None
        self.loss = None
        self.training_ended = False

        self._metrics = []
        self._metrics_names = []
        self._val_metrics_names = []
        self.running_avg_arr = []

        self.train_env = train_env

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_filename)
        self.experiment_name = experiment_name
        self.log_dir = log_dir

        self.for_testing = for_testing

        if not self.for_testing:
            cnt = 0
            print('Check {}'.format(self.checkpoint_filename), flush=True)
            while os.path.exists(self.checkpoint_filename) and not self.for_testing:
                print(self.checkpoint_filename, "exists", flush=True)
                cnt += 1
                self.checkpoint_dir = '{}_{}'.format(self.checkpoint_dir, cnt)
                self.checkpoint_filename = os.path.join(self.checkpoint_dir, checkpoint_filename)
            print('Check {}'.format(self.checkpoint_dir), flush=True)
            if not os.path.isdir('{}'.format(self.checkpoint_dir)):
                print('Create {}'.format(self.checkpoint_dir), flush=True)
                os.makedirs('{}'.format(self.checkpoint_dir))
            cnt = 0
            while os.path.exists(self.log_dir) and not self.for_testing:
                cnt += 1
                self.log_dir = '{}_{}'.format(self.log_dir, cnt)
            if not os.path.isdir('{}'.format(self.log_dir)):
                os.makedirs('{}'.format(self.log_dir))

            print('Write checkpoints to {}'.format(os.path.abspath(self.checkpoint_dir)))
            print('Writing TensorBoard logs to {}'.format(os.path.abspath(self.log_dir)))

        if not self.for_testing:
            self.writer = SummaryWriter(os.path.join(self.log_dir, self.experiment_name))

        if torch.cuda.is_available():
            print('CUDA available. Enabling GPU model.', flush=True)
            self.model = self.model.cuda()

        if replay_mem is None:
            self.D = ReplayMemory(get_q_hyper_cloneFreq_maxSteps_replaysize(self.cfg_dict)[2])
        else:
            self.D = replay_mem
        
        print('+++ Memory has size {} +++'.format(len(self.D)),flush=True)
        
        self.time_tracker = TimeTracker()
        self.changed_lr = False
        

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.save_original_lr()

    def save_original_lr(self):
        self.orig_lr = []
        for param_group in self.optimizer.param_groups:
            self.orig_lr.append(param_group['lr'])
            
    def set_original_lr(self):
        for idx,param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.orig_lr[idx]
            
    def get_lr(self):
        curr_lr = self.optimizer.param_groups[0]['lr']
        if curr_lr < 1e-8:
            curr_lr = 1e-8
            self.optimizer.param_groups[0]['lr'] = curr_lr
        return curr_lr

    @staticmethod
    def get_best_action(Q_s):
        Q_s = torch.squeeze(Q_s)
        ma, _ = torch.max(Q_s, 0)
        a = var_to_numpy(torch.nonzero(Q_s == ma))
        if len(a) == 1:
            a = a[0][0]
        elif len(a) > 1:
            i = np.random.choice(range(len(a)))
            a = a[i][0]
        return a

    def train(self, val_env=None, initial_epoch=1, num_epochs=100, decreasing_eps=500, target_eps=0.1,
              transitions_per_learning=100, lr_schedule=False):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        start_time_before_epoch_loop = self.time_tracker.start_measuring_time()
        # alpha = 1.01
        # delta = -np.log(1 - 0.01 + 0.01 / 7)
        # sigma = 1

        start_training_time = self.time_tracker.start_measuring_time()
        if lr_schedule:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5,
                                                                      verbose=True)
        print('Training {}...'.format(self.experiment_name), flush=True)
        history = []
        self.avg_steps_per_img = 0
        self.total_imgs = 0

        ## Set e-greedy epsilon values
        eps_ary = np.linspace(1, target_eps, num=decreasing_eps)  #
        eps_ary = np.concatenate([eps_ary, np.linspace(target_eps, target_eps, num=num_epochs - eps_ary.size)], axis=0)

        if self.tau_schedule > 0:
            sub_tau_ary = np.linspace(self.train_env.get_tau() - self.test_tau, 0, num=self.tau_schedule)
            sub_tau_ary = np.concatenate(
                [sub_tau_ary, np.zeros((num_epochs - sub_tau_ary.size,))],
                axis=0)

        # initialize target QNet
        # TODO mem

        self.target_qnet = create_copy(self.model)
        if torch.cuda.is_available():
            self.target_qnet.cuda()

        self.tensorboard_counter_step_lvl = 0

        loss = np.inf

#         self._stats_names = ['down', 'up', 'right', 'left', 'bigger', 'smaller', 'trigger', 'reward', 'row', 'col',
#                              'dice']

        self._stats_names = ['reward','dice']

        self._metrics_names = ['loss']

        self._metrics = [lambda x, y: self.loss(x, y)]

        self.running_avg_arr = np.zeros(len(self._metrics))
        self.time_tracker.stop_measuring_time(start_time_before_epoch_loop, 'Code before epoch loop',
                            print_gpu_usage=torch.cuda.is_available())

        for epoch in range(initial_epoch, num_epochs):
            start_time_epoch_loop = self.time_tracker.start_measuring_time()
            print('Starting epoch {}'.format(epoch), flush=True)
            print('...', flush=True)

            if epoch in np.arange(initial_epoch + 4, num_epochs, 5):
                # if epoch in np.arange(initial_epoch, num_epochs, 5):
                save_trajectory = True
            else:
                save_trajectory = False

            if lr_schedule:
                curr_lr = self.get_lr()
                if self.training_ended:
                    break

            self.model.train()
            if lr_schedule:
                if not self.for_testing:
                    self.writer.add_scalar('{}/learning_rate'.format(self.experiment_name), curr_lr, epoch)                   

            # Decrease epsilon for the first X epochs
            epsilon = eps_ary[epoch]
            # TODO mem
            # epsilon = 0.5
            if not self.for_testing:
                self.writer.add_scalar(
                    '{}/train/{}'.format(self.experiment_name, 'epsilon'),
                    epsilon, epoch)

            if self.tau_schedule > 0:
                sub_tau = sub_tau_ary[epoch]
            else:
                sub_tau = 0

            self.time_tracker.stop_measuring_time(start_time_epoch_loop, "One epoch before image loop",
                                print_gpu_usage=torch.cuda.is_available())

            
            if save_trajectory:
                if epoch == 1:
                    trajectory_all_imgs, triggered_all_imgs, Q_s_all_imgs, all_imgs, all_gt, actions_all_imgs, cases_all_imgs = self.train_epoch(
                    save_trajectory, epsilon, sub_tau, transitions_per_learning,True)
                else:
                    trajectory_all_imgs, triggered_all_imgs, Q_s_all_imgs, all_imgs, all_gt, actions_all_imgs, cases_all_imgs = self.train_epoch(
                    save_trajectory, epsilon, sub_tau, transitions_per_learning,False)
            else:
                if epoch == 1:
                    self.train_epoch(save_trajectory, epsilon, sub_tau, transitions_per_learning,True) 
                else:
                    self.train_epoch(save_trajectory, epsilon, sub_tau, transitions_per_learning,False)

            self.time_tracker.stop_measuring_time(start_time_epoch_loop, "One epoch before img saving & val",
                                print_gpu_usage=torch.cuda.is_available())

            if save_trajectory:
                start_saving_trajectories = self.time_tracker.start_measuring_time()
                if not os.path.isdir('{}/trajectories_train_{}'.format(self.checkpoint_dir, epoch)):
                    os.makedirs('{}/trajectories_train_{}'.format(self.checkpoint_dir, epoch))

                width = 2

                if len(trajectory_all_imgs) > 1:
                    for i, img in enumerate(trajectory_all_imgs):
                        orig_img = all_imgs[i]
                        if all_gt[i] is None:
                            orig_r = 0
                            orig_c = 0
                            orig_rn = 0 
                            orig_cn = 0
                        else:
                            orig_r, orig_c, orig_rn, orig_cn = get_nums_from_bbox(all_gt[i])
                        for j, step in enumerate(img):
                            r, c, rn, cn = step
                            if triggered_all_imgs[i] == 1 and j == (len(img) - 1):
                                save_image_with_orig_plus_current_bb(orig_img,
                                                                     '{}/trajectories_train_{}/{}_{}_trigger.png'.format(
                                                                         self.checkpoint_dir, epoch, i, j),
                                                                     bbox_flag=True,
                                                                     r=r, c=c,
                                                                     rn=rn, cn=cn, ro=orig_r, co=orig_c,
                                                                     rno=orig_rn,
                                                                     cno=orig_cn, lwidth=width,
                                                                     Q_s=Q_s_all_imgs[i][j],
                                                                     eps=epsilon, action=actions_all_imgs[i][j],
                                                                     case=cases_all_imgs[i][j])
                            else:
                                save_image_with_orig_plus_current_bb(orig_img,
                                                                     '{}/trajectories_train_{}/{}_{}.png'.format(
                                                                         self.checkpoint_dir, epoch,
                                                                         i, j),
                                                                     bbox_flag=True, r=r, c=c, rn=rn, cn=cn,
                                                                     ro=orig_r,
                                                                     co=orig_c, rno=orig_rn, cno=orig_cn,
                                                                     lwidth=width,
                                                                     Q_s=Q_s_all_imgs[i][j],
                                                                     eps=epsilon, action=actions_all_imgs[i][j],
                                                                     case=cases_all_imgs[i][j])
                else:
                    orig_img = all_imgs[0]
                    if all_gt[0] is None:
                        orig_r = 0
                        orig_c = 0
                        orig_rn = 0 
                        orig_cn = 0
                    else:
                        orig_r, orig_c, orig_rn, orig_cn = get_nums_from_bbox(all_gt[0])
                    for j, step in enumerate(trajectory_all_imgs[0]):
                        r, c, rn, cn = step
                        if triggered_all_imgs[0] == 1 and j == (len(trajectory_all_imgs[0]) - 1):
                            save_image_with_orig_plus_current_bb(orig_img,
                                                                 '{}/trajectories_train_{}/0_{}_trigger.png'.format(
                                                                     self.checkpoint_dir, epoch, j),
                                                                 bbox_flag=True, r=r,
                                                                 c=c, rn=rn, cn=cn, ro=orig_r, co=orig_c,
                                                                 rno=orig_rn, cno=orig_cn, lwidth=width,
                                                                 Q_s=Q_s_all_imgs[0][j],
                                                                 eps=epsilon, action=actions_all_imgs[0][j],
                                                                 case=cases_all_imgs[0][j])
                        else:
                            save_image_with_orig_plus_current_bb(orig_img,
                                                                 '{}/trajectories_train_{}/0_{}.png'.format(
                                                                     self.checkpoint_dir, epoch, j),
                                                                 bbox_flag=True, r=r, c=c, rn=rn, cn=cn, ro=orig_r,
                                                                 co=orig_c, rno=orig_rn, cno=orig_cn, lwidth=width,
                                                                 Q_s=Q_s_all_imgs[0][j],
                                                                 eps=epsilon, action=actions_all_imgs[0][j],
                                                                 case=cases_all_imgs[0][j])
                self.time_tracker.stop_measuring_time(start_saving_trajectories, 'Saving trajectories train',
                                    print_gpu_usage=torch.cuda.is_available())

            validate = False
            # Validate only each 5 epochs to save time in one image case
            if self.train_env.is_one_img() or \
                    get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(
                        self.cfg_dict)[1] < 100:

                if epoch in np.arange(initial_epoch + 4, num_epochs, 5):
                    validate = True
                    mode_list = ['val']
            #                 elif epoch in np.arange(initial_epoch + 50, num_epochs, 50):
            #                     validate = True
            #                     mode_list = ['train_eval', 'val']
            else:
                validate = True
                mode_list = ['val']

            # TODO mem
            # TODO TEsting
            # validate = True
            # mode_list = ['val']
            if validate:
                start_evaluation = self.time_tracker.start_measuring_time()
                print('Starting validation..', flush=True)
                for eval_mode in mode_list:
                    start_one_mode = self.time_tracker.start_measuring_time()
                    env = val_env if eval_mode == 'val' else self.train_env
                    val_metrics, trajectory_all_imgs, triggered_all_imgs, Q_s_all_imgs, all_imgs, all_gt, actions_all_imgs = self.evaluate(
                        env, epoch, save_trajectory=True)
                    start_saving_trajectories = self.time_tracker.start_measuring_time()
                    if not os.path.isdir('{}/trajectories_{}_{}'.format(self.checkpoint_dir, eval_mode, epoch)):
                        os.makedirs('{}/trajectories_{}_{}'.format(self.checkpoint_dir, eval_mode, epoch))

                    width = 2

                    if len(trajectory_all_imgs) > 1:
                        # print('i:',len(trajectory_all_imgs),flush=True)
                        for i, img in enumerate(trajectory_all_imgs):
                            if i >= 10:
                                break
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
                                        save_image_with_orig_plus_current_bb(orig_img,
                                                                             '{}/trajectories_{}_{}/{}_{}_{}_trigger.png'.format(
                                                                                 self.checkpoint_dir, eval_mode, epoch,
                                                                                 i, j, k),
                                                                             bbox_flag=True,
                                                                             r=r, c=c,
                                                                             rn=rn, cn=cn, ro=orig_r, co=orig_c,
                                                                             rno=orig_rn,
                                                                             cno=orig_cn, lwidth=width,
                                                                             Q_s=Q_s_all_imgs[i][j][k],
                                                                             eps=-1, action=actions_all_imgs[i][j][k])
                                    else:
                                        save_image_with_orig_plus_current_bb(orig_img,
                                                                             '{}/trajectories_{}_{}/{}_{}_{}.png'.format(
                                                                                 self.checkpoint_dir, eval_mode, epoch,
                                                                                 i, j, k),
                                                                             bbox_flag=True, r=r, c=c, rn=rn, cn=cn,
                                                                             ro=orig_r,
                                                                             co=orig_c, rno=orig_rn, cno=orig_cn,
                                                                             lwidth=width,
                                                                             Q_s=Q_s_all_imgs[i][j][k],
                                                                             eps=-1, action=actions_all_imgs[i][j][k])
                    else:
                        orig_img = all_imgs[0]
                        if all_gt[0] is None:
                            orig_r = 0
                            orig_c = 0
                            orig_rn = 0 
                            orig_cn = 0
                        else:
                            orig_r, orig_c, orig_rn, orig_cn = get_nums_from_bbox(all_gt[0])
                        for j, starts in enumerate(trajectory_all_imgs[0]):
                            for k, step in enumerate(starts):
                                r, c, rn, cn = step
                                if triggered_all_imgs[0][j] == 1 and k == (len(starts) - 1):
                                    save_image_with_orig_plus_current_bb(orig_img,
                                                                         '{}/trajectories_{}_{}/0_{}_{}_trigger.png'.format(
                                                                             self.checkpoint_dir, eval_mode, epoch, j,
                                                                             k),
                                                                         bbox_flag=True, r=r,
                                                                         c=c, rn=rn, cn=cn, ro=orig_r, co=orig_c,
                                                                         rno=orig_rn, cno=orig_cn, lwidth=width,
                                                                         Q_s=Q_s_all_imgs[0][j][k],
                                                                         action=actions_all_imgs[0][j][k])
                                else:
                                    save_image_with_orig_plus_current_bb(orig_img,
                                                                         '{}/trajectories_{}_{}/0_{}_{}.png'.format(
                                                                             self.checkpoint_dir, eval_mode, epoch, j,
                                                                             k),
                                                                         bbox_flag=True, r=r, c=c, rn=rn, cn=cn,
                                                                         ro=orig_r,
                                                                         co=orig_c, rno=orig_rn, cno=orig_cn,
                                                                         lwidth=width,
                                                                         Q_s=Q_s_all_imgs[0][j][k],
                                                                         action=actions_all_imgs[0][j][k])

                    self.time_tracker.stop_measuring_time(start_saving_trajectories, 'Saving {} trajectories'.format(eval_mode),
                                        print_gpu_usage=torch.cuda.is_available())
                    if eval_mode == 'val':
                        start_time_saving_model = self.time_tracker.start_measuring_time()
                        history.append(val_metrics)
                        print('Validation finished with metrics of {}'.format(val_metrics), flush=True)

                        print('Start saving', flush=True)
                        print('...', flush=True)

                        save_checkpoint_and_best(history, entry_idx=3, smaller_better=False, model=self.model, optimizer=self.optimizer, epoch=epoch, checkpoint_filename=self.checkpoint_filename,
                                     checkpoint_dir=self.checkpoint_dir, experiment_name=self.experiment_name,replay_mem=self.D)
                        self.time_tracker.stop_measuring_time(start_time_saving_model, "Saving model validation",
                                            torch.cuda.is_available())

                    self.time_tracker.stop_measuring_time(start_one_mode, 'Eval of {}'.format(eval_mode),
                                        print_gpu_usage=torch.cuda.is_available())

                self.time_tracker.stop_measuring_time(start_evaluation, 'Whole eval (val + train)',
                                    print_gpu_usage=torch.cuda.is_available())

            if lr_schedule:
                lr_scheduler.step(float(loss))
                # pdb.set_trace()

            self.time_tracker.stop_measuring_time(start_time_epoch_loop, "One epoch the whole deal",
                                print_gpu_usage=torch.cuda.is_available())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print_current_time_epoch(start_time_before_epoch_loop, epoch)

        # Training loop exited normally
        self.training_ended = True
        self.time_tracker.stop_measuring_time(start_training_time, 'Whole training loop', print_gpu_usage=torch.cuda.is_available())

    def calc_and_print_metrics(self, q_vals, targets):
        for _metric_idx, (_metric, _metric_name) in enumerate(zip(self._metrics, self._metrics_names)):
            if _metric_idx == 0:
                curr_metric = _metric(q_vals, targets).data
            else:
                curr_metric = _metric(q_vals, targets)
            self.running_avg_arr[_metric_idx] = (self.tensorboard_counter_step_lvl * self.running_avg_arr[
                _metric_idx] + curr_metric) / (self.tensorboard_counter_step_lvl + 1)

            if not self.for_testing:
                if _metric_idx == 0:
                    self.writer.add_scalars(
                        '{}/train/{}'.format(self.experiment_name, _metric_name),
                        {'value': curr_metric},
                        self.tensorboard_counter_step_lvl)
                    self.writer.add_scalars(
                        '{}/train/{}'.format(self.experiment_name, _metric_name),
                        {'running_avg': self.running_avg_arr[_metric_idx]},
                        self.tensorboard_counter_step_lvl)
                elif _metric_idx < 4:
                    self.writer.add_scalars(
                        '{}/train/predictions'.format(self.experiment_name),
                        {'{}'.format(_metric_name): curr_metric},
                        self.tensorboard_counter_step_lvl)
                    self.writer.add_scalars(
                        '{}/train/predictions_avg'.format(self.experiment_name),
                        {'{}_running_avg'.format(_metric_name): self.running_avg_arr[_metric_idx]},
                        self.tensorboard_counter_step_lvl)
                else:
                    self.writer.add_scalars(
                        '{}/train/targets'.format(self.experiment_name),
                        {'{}'.format(_metric_name): curr_metric},
                        self.tensorboard_counter_step_lvl)
                    self.writer.add_scalars(
                        '{}/train/targets_avg'.format(self.experiment_name),
                        {'{}_running_avg'.format(_metric_name): self.running_avg_arr[_metric_idx]},
                        self.tensorboard_counter_step_lvl)
        del curr_metric

    def get_q_fct(self, env, no_imgs):
        return_dict = {}
        # loop over images
        for idx in range(no_imgs):
            print('{}. image'.format(idx + 1), flush=True)
            env.reset()
            bb_row, bb_col, bb_row_no, bb_col_no = env.get_bbs_plotting()
            # bb_row,bb_col,bb_row_no,bb_col_no,rows_all_tile_sizes, cols_all_tile_sizes, row_lims, col_lims ,spacings= env.get_bbs_plotting()
            # print('bb_row',bb_row,flush=True)
            # print('bb_col',bb_col,flush=True)
            # print('bb_row_no',bb_row_no,flush=True)
            # print('bb_col_no',bb_col_no,flush=True)
            # print('rows_all_tile_sizes',rows_all_tile_sizes,flush=True)
            # print('cols_all_tile_sizes',cols_all_tile_sizes,flush=True)
            # print('row_lims',row_lims,flush=True)
            # print('col_lims',col_lims,flush=True)
            # print('spacings',spacings,flush=True)

            figures_all_tile_sizes = []

            # For all 'resolutions' (5 tiles, 10 tiles...)
            for bb in range(len(bb_row)):
                print('{}. resolution'.format(bb + 1), flush=True)

                row_no_list = bb_row_no[bb]
                col_no_list = bb_col_no[bb]
                row_list = bb_row[bb]
                col_list = bb_col[bb]

                result_figure = []

                # For all rows therein
                for _bb in range(len(row_no_list)):
                    print('{}. row of {}'.format(_bb + 1, len(row_no_list)), flush=True)

                    result_row = []

                    row_no = row_no_list[_bb]
                    col_no = col_no_list[_bb]
                    row = row_list[_bb]
                    col = col_list[_bb]
                    # print('---',flush=True)
                    # print('row_no',row_no,flush=True)
                    # print('col_no',col_no,flush=True)

                    # print('row',row,flush=True)
                    # print('col',col,flush=True)

                    new_resized_im, _, _ = env.reset_with_given_bb(row, col, row_no, col_no)
                    if new_resized_im is None:
                        pass
                    else:
                        Q_s = self.model.forward_all(numpy_to_var(new_resized_im))

                        result_row.append(var_to_numpy(Q_s))

                        a = 2  # go right
                        go_further = True
                        while go_further:
                            new_resized_im, _, done, has_moved, see_lesion, _ = env.step(a)
                            if not has_moved:
                                go_further = False
                            else:
                                Q_s = self.model.forward_all(numpy_to_var(new_resized_im))
                                result_row.append(var_to_numpy(Q_s))
                        # print('result_row',np.array(result_row).shape)
                        result_figure.append(result_row)

                # print('result_figure',np.array(result_figure).shape)
                # print('-====-',flush=True)
                figures_all_tile_sizes.append(np.array(result_figure))

            # return_dict[idx] = [figures_all_tile_sizes,rows_all_tile_sizes,cols_all_tile_sizes, row_lims,
            # col_lims,spacings, env.get_original_img()]
            image = np.copy(env.get_original_img())
        #     print(np.min(image),np.mean(image),np.max(image),flush=True)
            if np.max(image)>1:
        #         print('clip to 0 255',flush=True)
                image = np.clip(image, 0, 255)
                image = image.astype(int)
            else:
        #         print('clip to 0 1',flush=True)
                image = np.clip(image, 0.0, 1.0)
            return_dict[idx] = [figures_all_tile_sizes, image, env.get_original_bb()]
        return return_dict

    @staticmethod
    def get_activation_fcn(env, no_imgs):
        return_dict = {}
        # loop over images
        for idx in range(no_imgs):
            print('{}. image'.format(idx + 1), flush=True)
            env.reset(fcn=True)
            bb_row, bb_col, bb_row_no, bb_col_no = env.get_bbs_plotting()

            figures_all_tile_sizes = []

            # For all 'resolutions' (5 tiles, 10 tiles...)
            for bb in range(len(bb_row)):
                print('{}. resolution'.format(bb + 1), flush=True)
                row_no_list = bb_row_no[bb]
                col_no_list = bb_col_no[bb]
                row_list = bb_row[bb]
                col_list = bb_col[bb]

                result_figure = []

                # For all rows therein
                for _bb in range(len(row_no_list)):
                    print('{}. row of {}'.format(_bb + 1, len(row_no_list)), flush=True)

                    result_row = []

                    row_no = row_no_list[_bb]
                    col_no = col_no_list[_bb]
                    row = row_list[_bb]
                    col = col_list[_bb]

                    new_resized_im, _, _ = env.reset_with_given_bb(row, col, row_no, col_no, fcn=True)
                    if new_resized_im:
                        result_row.append(new_resized_im)

                        a = 2  # go right
                        go_further = True
                        while go_further:
                            new_resized_im, _, done, has_moved, see_lesion, _ = env.step(a, fcn=True)
                            if has_moved:
                                go_further = False
                            else:
                                result_row.append(new_resized_im)
                        result_figure.append(result_row)

                figures_all_tile_sizes.append(np.array(result_figure))

            # return_dict[idx] = [figures_all_tile_sizes,rows_all_tile_sizes,cols_all_tile_sizes, row_lims, col_lims,spacings, env.get_original_img()]
            return_dict[idx] = [figures_all_tile_sizes, env.get_original_img()]
        return return_dict

    def evaluate(self, env, epoch, save_trajectory=False,test_mode=3):
        start_time_eval = self.time_tracker.start_measuring_time()
        self.model.eval()

        self._val_metrics_names = ['per_img_hits', 'overall_hits', 'lesions_acc','score']
        val_metrics_arr = np.zeros(len(self._val_metrics_names))

        per_image_lesions = 0
        lesions_found = 0
        correct = 0
        
        limit = 8
        FP = 0
        TP = 0
        FN = 0
        TN = 0
        y_true = []
        y_pred = []
        
        images_left = True
        total_num_imgs = 0
        num_indi_imgs = 0
        dist_list = []
        
        lesion_found_all_imgs = []
        osc_all_imgs = []
        dice_osc_all_imgs = []
        see_lesion_dice_all = []
        combi_osc_all_imgs = []
        if save_trajectory:
            trajectory_all_imgs = []
            Q_s_all_imgs = []
            triggered_all_imgs = []
            all_imgs = []
            all_gt = []
            actions_all_imgs = []
            
        num_imgs_correct_lesion = 0
        num_imgs_incorrect_lesion = 0
        while images_left:
            start_time_one_img = self.time_tracker.start_measuring_time()
            num_indi_imgs += 1
            
            found_correct_lesion = False
            found_incorrect_lesion = False
            
            lesion_found_before_this_img = lesions_found
            new_resized_im, images_left, see_lesion,_ = env.reset()

            lesion_found_one_img = []
            osc_one_img = []
            dice_osc_one_img = []
            see_lesion_dice_one = []
            combi_osc_one_img = []
            if save_trajectory:
                trajectory_all_starts_one_img = []
                triggered_all_starts_one_img = []
                actions_all_starts_one_img = []
                Q_s_all_starts_one_img = []
                all_imgs.append(env.get_original_img())
                all_gt.append(env.get_original_bb())

            bb_row, bb_col, bb_row_no, bb_col_no = env.get_bbs_testing(test_mode)
            for bb in range(len(bb_row)):
                if self.recurrent > 0:
                    self.model.reset_hidden_state_running()
                if save_trajectory:
                    trajectory_one_start = []
                    Q_s_one_start = []
                    actions_one_start = []

                total_num_imgs += 1
                row_no = bb_row_no[bb]
                col_no = bb_col_no[bb]
                row = bb_row[bb]
                col = bb_col[bb]

                new_resized_im, see_lesion,_ = env.reset_with_given_bb(row, col, row_no, col_no)
                if save_trajectory:
                    trajectory_one_start.append(env.get_current_bb_list())

                # repeat until trigger
                cnt = 0
                done = False
                trigger_flag = False

                while (not done) and cnt < self.max_steps:
                    cnt += 1
                    Q_s = self.model.forward_all(numpy_to_var(new_resized_im))
                    if save_trajectory:
                        Q_s_one_start.append(var_to_numpy(Q_s))
                    # print('Step {} Q_s:'.format(cnt),Q_s,flush=True)

                    a = self.get_best_action(Q_s)
                    new_resized_im, _, done, _, see_lesion, _ = env.step(a)
                    if save_trajectory:
                        trajectory_one_start.append(env.get_current_bb_list())
                        actions_one_start.append(a)

                    if a == 6:
                        new_dice = env.get_current_dice()
                        lesions_found += 1
                        trigger_flag = True
                        if new_dice >= self.test_tau:
                            found_correct_lesion = True
                            correct += 1
                        else:
                            found_incorrect_lesion = True
                
                if save_trajectory:
                    if trigger_flag:
                        triggered_all_starts_one_img.append(1)
                    else:
                        triggered_all_starts_one_img.append(0)
                    trajectory_all_starts_one_img.append(trajectory_one_start)
                    Q_s_one_start.append(np.zeros(Q_s.shape))
                    Q_s_all_starts_one_img.append(Q_s_one_start)
                    actions_one_start.append(-1)
                    actions_all_starts_one_img.append(actions_one_start) 
                    
                is_oscillating, combi_string = env.is_oscillating(limit = limit)
                see_lesion = True if env.get_current_dice() > 0 else False
                if is_oscillating:   
                    dsc = env.get_current_dice()
                    dice_osc_one_img.append(dsc)
                    combi_osc_one_img.append(combi_string)
                    see_lesion = True if env.get_current_dice() > 0 else False
                    see_lesion_dice_one.append(see_lesion)
                    
                osc_one_img.append(is_oscillating)
                lesion_found_one_img.append(trigger_flag)

                points_last_bb = convert_bbs_to_points(env.get_current_bb())
                points_orig_bb = convert_bbs_to_points(env.get_original_bb())
                if not (points_orig_bb is None):
                    distances = distance.cdist(points_last_bb, points_orig_bb)
                    distances = distances.diagonal()
                    dist_list.append(distances)
                    del distances
                del points_last_bb
                del points_orig_bb
               

                
            if save_trajectory:
                trajectory_all_imgs.append(trajectory_all_starts_one_img)
                triggered_all_imgs.append(triggered_all_starts_one_img)
                Q_s_all_imgs.append(Q_s_all_starts_one_img)
                actions_all_imgs.append(actions_all_starts_one_img)
            osc_all_imgs.append(osc_one_img)
            dice_osc_all_imgs.append(dice_osc_one_img)
            combi_osc_all_imgs.append(combi_osc_one_img)
            see_lesion_dice_all.append(see_lesion_dice_one)
            lesion_found_all_imgs.append(lesion_found_one_img)
            
            #hit, DSC > 0.2
            if found_correct_lesion:
                #has lesion
                num_imgs_correct_lesion += 1
                TP += 1
                y_true.append(1)
                y_pred.append(1)
            #hit, DSC < 0.2
            elif found_incorrect_lesion:
                #no lesion
                if env.get_original_bb() is None:
                    FP += 1
                    y_true.append(0)
                    y_pred.append(1)
                #has lesion
                else:
                    num_imgs_incorrect_lesion += 1
                    FN += 1
                    y_true.append(1)
                    y_pred.append(0)
            #no hit
            else:
                #no lesion
                if env.get_original_bb() is None:
                    TN += 1
                    y_true.append(0)
                    y_pred.append(0)
                #has lesion
                else:
                    FN += 1
                    y_true.append(1)
                    y_pred.append(0)
                

            if lesions_found > lesion_found_before_this_img:
                per_image_lesions += 1
            self.time_tracker.stop_measuring_time(start_time_one_img, "Processing one img eval", torch.cuda.is_available())

        dist_list = np.array(dist_list)
        if not self.for_testing:
            self.writer.add_scalars('{}/val/correct'.format(self.experiment_name),
                                    {'tp': TP,
                                     'tn': TN},
                                    epoch)
            self.writer.add_scalars('{}/val/incorrect'.format(self.experiment_name),
                                    {'fp': FP,
                                     'fn': FN},
                                    epoch)
        if not self.for_testing:
            self.writer.add_scalars(
                '{}/val/{}'.format(self.experiment_name, 'distance'),
                {'sum': np.sum(dist_list), 'max': np.max(dist_list), 'avg': np.mean(dist_list),
                 'min': np.min(dist_list)}, epoch)
        print('+++++++++++++++++++++', flush=True)
        print('++++ PERFORMANCE ++++', flush=True)
        print('+++++++++++++++++++++', flush=True)
        print('Per image lesions', per_image_lesions, flush=True)
        print('Overall lesions found', lesions_found, flush=True)
        print('Lesions correct', correct, flush=True)
        print('Number of indi imgs', num_indi_imgs, flush=True)
        print('Total number imgs', total_num_imgs, flush=True)
        print('Number of imgs with at least one correct trigger', num_imgs_correct_lesion, flush=True)
        print('Number of imgs with incorrect triggers only', num_imgs_incorrect_lesion, flush=True)
        print('Score',num_imgs_correct_lesion-num_imgs_incorrect_lesion, flush=True)
        print('Number of oscillations/per img with limit {} : {}'.format(limit,np.sum(np.array(osc_all_imgs))/total_num_imgs),flush=True)
        print('Number of oscillations with limit {} : {}'.format(limit,np.sum(np.array(osc_all_imgs))),flush=True)
        num_see_lesion = 0
        for l in see_lesion_dice_all:
            for entry in l:
                if entry:
                    num_see_lesion += 1
        print('Number of oscillations with lesion in bb',num_see_lesion,flush=True)
        
        extra_hit = 0
        for idx,l in enumerate(lesion_found_all_imgs):
            if (not any(l)) and any(osc_all_imgs[idx]):
                extra_hit += 1
        print('Number of imgs without hit but with oscillation',extra_hit,flush=True) 
        
        extra_hit = 0
#         print(lesion_found_all_imgs)
#         print(osc_all_imgs)
#         print(dice_osc_all_imgs)
        count_lr = 0
        count_ud = 0
        count_sb = 0
        count_tz = 0
        count_tt = 0
        count_other = 0
        for idx,l in enumerate(lesion_found_all_imgs):
            if (not any(l)) and any(osc_all_imgs[idx]):
                l_idx = -1
                for entry in osc_all_imgs[idx]:
                    if entry:
                        l_idx += 1
                        if dice_osc_all_imgs[idx][l_idx] > 0:
                            extra_hit += 1
                            if combi_osc_all_imgs[idx][l_idx] == 'lr':
                                count_lr += 1
                            elif combi_osc_all_imgs[idx][l_idx] == 'ud':
                                count_ud += 1
                            elif combi_osc_all_imgs[idx][l_idx] == 'sb':
                                count_sb += 1
                            elif combi_osc_all_imgs[idx][l_idx] == 'tz':
                                count_tz += 1
                            elif combi_osc_all_imgs[idx][l_idx] == 'tt':
                                count_tt += 1
                            else:
                                count_other += 1
                                print('Combi string is:',combi_osc_all_imgs[idx][l_idx],flush=True)
                            break
        print('Number of imgs without hit but with oscillation, DSC > 0:',extra_hit,flush=True) 
        print('Combi of actions: lr {}, ud {}, sb {}, tz {}, tt {}, other {}'.format(count_lr,\
                            count_ud,count_sb,count_tz,count_tt,count_other),flush=True)
        
        num_dice = 0
        sum_dice = 0
        sum_pos_dice = 0
        num_pos_dice = 0
        for l in dice_osc_all_imgs:
            for entry in l:
                num_dice += 1
                sum_dice += entry
                if entry > 0:
                    sum_pos_dice += entry
                    num_pos_dice += 1
        
        if num_dice > 0:
            print('Average of dice score of oscillations',sum_dice/num_dice ,flush=True)
        else:
            print('Average of dice score of oscillations',0 ,flush=True)
        if num_pos_dice > 0:
            print('Average of dice score of oscillations if see lesion',sum_pos_dice/num_pos_dice ,flush=True)
        else:
            print('Average of dice score of oscillations if see lesion',0 ,flush=True)
        print('+++++++++++++++++++++', flush=True)
        print('TP {}, FP {}, FN {}, TN {}'.format(TP,FP,FN,TN),flush=True)
        try:
            print('F1',f1_score(y_true, y_pred, average=None),flush=True)
            print('Precision, recall, F1',precision_recall_fscore_support(y_true, y_pred, average=None),flush=True)
            print('Precision',(TP/(TP+FP)),flush=True)
            print('Recall',(TP/(TP+FN)),flush=True)
            print('F1',2*((TP/(TP+FP))*(TP/(TP+FN)))/((TP/(TP+FP))+(TP/(TP+FN))),flush=True)
        except ZeroDivisionError:
            pass
        
        try:
            print('-------------------------',flush=True)
            precision = correct/lesions_found
            recall = TP /100
            print('Precision lesion lvl',precision,flush=True)
            print('Recall lesion lvl',recall,flush=True)
            print('F1 lesion lvl',(2*precision*recall)/(precision+recall),flush=True)
        except ZeroDivisionError:
            pass
        
        
        for i, name in enumerate(self._val_metrics_names):
            if name == 'per_img_hits':
                value = per_image_lesions / num_indi_imgs
            elif name == 'overall_hits':
                value = lesions_found / total_num_imgs
            elif name == 'lesions_acc':
                if lesions_found > 0:
                    value = correct / lesions_found
                else:
                    value = 0
            elif name == 'score':
                value = num_imgs_correct_lesion-num_imgs_incorrect_lesion

            val_metrics_arr[i] = value
            if not self.for_testing:
                self.writer.add_scalars(
                    '{}/val/{}'.format(self.experiment_name, name),
                    {'value': value}, epoch)

        
        if not self.for_testing:
            self.writer.add_scalars(
                '{}/val/hits'.format(self.experiment_name),
                {'total': lesions_found},
                epoch)
            self.writer.add_scalars(
                '{}/val/hits'.format(self.experiment_name),
                {'correct': correct},
                epoch)

        self.time_tracker.stop_measuring_time(start_time_eval, 'Evaluation loop', print_gpu_usage=torch.cuda.is_available())
        gc.collect()
        if save_trajectory:
            return val_metrics_arr, trajectory_all_imgs, triggered_all_imgs, Q_s_all_imgs, all_imgs, all_gt, actions_all_imgs
        return val_metrics_arr
    
    def train_epoch(self, save_trajectory, epsilon, sub_tau, transitions_per_learning,warmup_lr=False):
        imgs_per_epoch = 0
        images_left = True
        
        if warmup_lr:
            num_imgs = get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(
        self.cfg_dict)[1]
            lr = get_q_train_opti_lr_mom_epochs_transPerUp(self.cfg_dict)[1]
            lr_ary = np.linspace(start=0.0000001,stop=lr,num=num_imgs)

        if save_trajectory:
            trajectory_all_imgs = []
            triggered_all_imgs = []
            Q_s_all_imgs = []
            all_imgs = []
            all_gt = []
            actions_all_imgs = []
            cases_all_imgs = []

        #             print('GC 02')
        #             for obj in gc.get_objects():
        #                 try:
        # #                     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #                     print(reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
        #                 except:
        #                     pass
        while images_left:
            
            if warmup_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_ary[imgs_per_epoch]
                self.changed_lr = True
                
            if self.changed_lr and not warmup_lr:
                self.set_original_lr()
                self.changed_lr = False
                
                
                
                
#             self.writer.add_scalar('{}/gpu_start_img'.format(self.experiment_name), get_memory_usage() , self.tensorboard_counter_step_lvl)
#             print('GPU',get_memory_usage(),flush=True)
            start_time_one_img = self.time_tracker.start_measuring_time()
            if save_trajectory:
                trajectory_one_start = []
                Q_s_one_start = []
                actions_one_start = []
                cases_one_start = []

            print('-----', flush=True)
            print('Starting image {}'.format(imgs_per_epoch + 1), flush=True)

            imgs_per_epoch += 1
            self.total_imgs += 1
            
            # TODO mem
            if self.total_imgs % self.clone_freq == 0:
                del self.target_qnet
                self.target_qnet = create_copy(self.model)
                if torch.cuda.is_available():
                    self.target_qnet.cuda()
            
            start_reset = self.time_tracker.start_measuring_time()
            # TODO mem
            if self.recurrent > 0:
                self.transition_list = collections.deque(maxlen=self.recurrent)

            self.current_resized_im, images_left, see_lesion,_ = self.train_env.reset()
            # if imgs_per_epoch == 30:
            #     images_left = False
            # else:
            #     images_left = True
            self.time_tracker.stop_measuring_time(start_reset, "Reset env")
            # TODO mem
            # has_seen_lesion = see_lesion
            # lesion_in_starting_bb = see_lesion

            if save_trajectory:
                trajectory_one_start.append(self.train_env.get_current_bb_list())
                all_imgs.append(self.train_env.get_original_img())
                all_gt.append(self.train_env.get_original_bb())

            done = False
            steps_this_img = 0

            while not done:
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                start_time_one_step = self.time_tracker.start_measuring_time()
                # print('Starting step {}:'.format(steps_this_img + 1), flush=True)
                if self.recurrent > 0:
                    self.model.reset_hidden_state_running()

                steps_this_img += 1
                self.tensorboard_counter_step_lvl += 1
                # self.writer.add_scalar('{}/gpu_start_step'.format(self.experiment_name), get_memory_usage() , self.tensorboard_counter_step_lvl)
                
                start_time_action_selection = self.time_tracker.start_measuring_time()

                #TODO mem
                a, case, explo, Q_s = self.action_selection(steps_this_img, epsilon, sub_tau)
#                 a = 5
#                 case = 1
#                 explo = False
#                 Q_s = None

#                 self.time_tracker.stop_measuring_time(start_time_action_selection, 'Action selection',
#                                     print_gpu_usage=torch.cuda.is_available())

                start_time_between_selection_and_update = self.time_tracker.start_measuring_time()

                if save_trajectory and explo:
                    Q_s_one_start.append(var_to_numpy(Q_s))
                elif save_trajectory:
                    Q_s_one_start.append(np.zeros((7, 1)))

                # print('{}'.format(string_action(a)), flush=True)
                start_step = self.time_tracker.start_measuring_time()
                # TODO mem
                self.new_resized_im, reward, done, _, see_lesion, action_hist = self.train_env.step(a, sub_tau=sub_tau)
#                 reward = 1
#                 if steps_this_img == 30:
#                     done =True
#                 else:
#                     done = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.time_tracker.stop_measuring_time(start_step, "==Env step")

                # TODO
                if see_lesion:
                    has_seen_lesion = True
                if a == 6 and save_trajectory:
                    trigger_flag = True

                if save_trajectory:
                    cases_one_start.append(case)
                    actions_one_start.append(a)
                    trajectory_one_start.append(self.train_env.get_current_bb_list())

                start_tensorboard = self.time_tracker.start_measuring_time()
                # TODO mem
                for i, name in enumerate(self._stats_names):
                    if name == string_action(a):
                        value = 1
                    elif name == 'reward':
                        value = reward
                    elif name == 'row':  # row
                        value = self.train_env.get_current_bb()[0]
                    elif name == 'col':  # col
                        value = self.train_env.get_current_bb()[1]
                    elif name == 'dice':  # dice
                        value = self.train_env.get_current_dice()
                    else:
                        value = 0
                    if not self.for_testing:
                        self.writer.add_scalars(
                            '{}/train_stats/{}'.format(self.experiment_name, name),
                            {'value': value}, self.tensorboard_counter_step_lvl)

                self.time_tracker.stop_measuring_time(start_tensorboard, 'Writing to tensorbard train_stats')
                # TODO mem
                if self.recurrent > 0:
                    if steps_this_img >= self.recurrent:
                        self.transition_list.append(replay.Transition(self.current_resized_im, a, reward, self.new_resized_im, action_hist))
                        self.D.push_list(self.transition_list)
                    else:
                        self.transition_list.append(replay.Transition(self.current_resized_im, a, reward, self.new_resized_im, action_hist))
                else:
                    self.D.push(self.current_resized_im, a, reward, self.new_resized_im, action_hist)
                # TODO COPY HERE!!
#                 self.writer.add_image('Dataset', self.current_resized_im, self.tensorboard_counter_step_lvl)
                del self.current_resized_im
                self.current_resized_im = self.new_resized_im
#                 print('Current resized im',self.current_resized_im.shape,np.min(self.current_resized_im),np.max(self.current_resized_im),flush=True)
                del self.new_resized_im

                self.time_tracker.stop_measuring_time(start_time_between_selection_and_update, "++Between action selection and update",
                                    print_gpu_usage=torch.cuda.is_available())
                ###########################
                # Update network          #
                ###########################

                # TODO mem
                self.update_qnet(transitions_per_learning)
                self.time_tracker.stop_measuring_time(start_time_one_step, "One step", print_gpu_usage=torch.cuda.is_available())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # TODO mem
            points_last_bb = convert_bbs_to_points(self.train_env.get_current_bb())
            points_orig_bb = convert_bbs_to_points(self.train_env.get_original_bb())
            distances = distance.cdist(points_last_bb, points_orig_bb)
            distances = distances.diagonal()
            del points_last_bb
            del points_orig_bb


            #                 print('GC 06')
            #                 for obj in gc.get_objects():
            #                     try:
            # #                         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #                         print(reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
            #                     except:
            #                         pass

            # TODO mem
            if not self.for_testing:
                self.writer.add_scalars(
                    '{}/train_stats/{}'.format(self.experiment_name, 'distance'),
                    {'sum': np.sum(distances), 'max': np.max(distances), 'avg': np.mean(distances),
                     'min': np.min(distances)}, self.total_imgs)
            del distances

            self.avg_steps_per_img = (self.avg_steps_per_img * self.total_imgs + steps_this_img) / (self.total_imgs + 1)
            if not self.for_testing:
                self.writer.add_scalar(
                    '{}/train/{}'.format(self.experiment_name, 'steps_per_img'),
                    steps_this_img, self.total_imgs)
                self.writer.add_scalar(
                    '{}/train/{}'.format(self.experiment_name, 'steps_per_img_avg'),
                    self.avg_steps_per_img, self.total_imgs)

            if save_trajectory:
                if trigger_flag:
                    triggered_all_imgs.append(1)
                else:
                    triggered_all_imgs.append(0)
                # print(cases_one_start)
                cases_one_start.append(-1)
                cases_all_imgs.append(cases_one_start)
                Q_s_one_start.append(np.zeros((7, 1)))
                Q_s_all_imgs.append(Q_s_one_start)
                actions_one_start.append(-1)
                actions_all_imgs.append(actions_one_start)
                trajectory_all_imgs.append(trajectory_one_start)

#             self.time_tracker.stop_measuring_time(start_time_one_img, 'Processing one img', print_gpu_usage=torch.cuda.is_available())

            # TODO mem
        if save_trajectory:
            return trajectory_all_imgs, triggered_all_imgs, Q_s_all_imgs, all_imgs, all_gt, actions_all_imgs, cases_all_imgs

    def action_selection(self, steps_this_img, epsilon, sub_tau):
        case = -1
        explo = False
        Q_s = None
        ######################
        # Max steps reached  #
        ######################
        if steps_this_img == self.max_steps:
            # Trigger action
            a = 6
            case = 5
        else:
            ############################
            # 'Normal' epsilon greedy  #
            ############################

            ################
            # Exploration  #
            ################
            if (np.random.uniform(0, 1) < epsilon):
                ######################
                # Guided exploration #
                ######################

                # if 0 <= rnd < kappa: random action
                # if eps < kappa: action with positive reward
                ############
                # Random   #
                ############
                if np.random.uniform(0, 1) < self.kappa:
                    a = np.random.randint(0, 7)
                    case = 1
                #####################
                # Positive Reward   #
                #####################
                else:
                    start_pos_acts = self.time_tracker.start_measuring_time()
                    actions, _, _, got_eta = self.train_env.get_positive_actions(sub_tau=sub_tau)
                    self.time_tracker.stop_measuring_time(start_pos_acts, "Getting positive actions")
                    if len(actions) == 0:
                        # If there are no positive action choose randomly
                        case = 2
#                         a = np.random.randint(0, 7)
                        # choose action which reduces distance
                        a = self.train_env.get_action_smaller_dist()
                    ## TODO don't do this elif to get back to 'normal' --> choose rnd action not trigger action
                    elif got_eta:
                        # Trigger is positive action
                        a = 6
                        case = 3
                    else:
                        # There are more than one positive action
                        a = np.random.choice(actions)
                        case = 3
            #################
            # Exploitation  #
            #################
            else:
                case = 4
                start_explo = self.time_tracker.start_measuring_time()
                with torch.no_grad():
                    Q_s = self.model.forward_all(numpy_to_var(self.current_resized_im))
                a = self.get_best_action(Q_s)
                self.time_tracker.stop_measuring_time(start_explo, "Exploitation")
                explo = True
                Q_s = torch.squeeze(Q_s)
                if not self.for_testing:
                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_down'),
                        {'value': Q_s.data[0]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_up'),
                        {'value': Q_s.data[1]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_right'),
                        {'value': Q_s.data[2]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_left'),
                        {'value': Q_s.data[3]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_bigger'),
                        {'value': Q_s.data[4]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_smaller'),
                        {'value': Q_s.data[5]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_trigger'),
                        {'value': Q_s.data[6]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_max'),
                        {'value': torch.max(Q_s, 0)[0]}, self.tensorboard_counter_step_lvl)

                    self.writer.add_scalars(
                        '{}/train_stats/{}'.format(self.experiment_name, 'q_argmax'),
                        {'value': a}, self.tensorboard_counter_step_lvl)

        return a, case, explo, Q_s

    def update_qnet(self, transitions_per_learning):
        # method_cnt, method_cnt_objs = get_gc_size()
        update = True
#         if self.recurrent > 0:
#             if len(self.D) > (self.recurrent + 5):
#                 if len(self.D) < transitions_per_learning:
#                     transitions = self.D.sample_seq(len(self.D), self.recurrent)
#                 else:
#                     transitions = self.D.sample_seq(transitions_per_learning, self.recurrent)
#             else:
#                 update = False
#         else:
        if len(self.D) == 0:
            update = False
        elif len(self.D) < transitions_per_learning:
            transitions = self.D.sample(len(self.D))
        else:
            transitions = self.D.sample(transitions_per_learning)

        if update:
            start_time_update = self.time_tracker.start_measuring_time()
            # print('Update', flush=True)
            q_vals = torch.zeros(len(transitions), 7).float()
            q_vals = tensor_to_var(q_vals)
            
            targets = torch.zeros(len(transitions), 7).float()

            # go either through list of transitions or transition-sequences
            
            start_loop_targets = self.time_tracker.start_measuring_time()
            for i, transition in enumerate(transitions):
                target = None
                if self.recurrent > 0:
                    if transition[-1].action == 6:
                        target = transition[-1].reward
                else:
                    if transition.action == 6:
                        target = transition.reward

                if target is None:
                    if self.recurrent > 0:
                        with torch.no_grad():
                            self.target_qnet.reset_hidden_state_update()
                            list_next_states = get_attribute_from_list(transition, 'resized_im_next_state')
                            var_transitions = tensor_to_var(numpy_to_tensor(np.array(list_next_states)))
                            # print('---',var_transitions.shape, flush=True)
                            Q_s = self.target_qnet.forward_all(
                                var_transitions)  # .view(len(transition.next_state), 1, -1)))
                            value = np.max(var_to_numpy(Q_s))
                            del list_next_states
                            del var_transitions
                            del Q_s
                    elif self.double:
                        with torch.no_grad():
                            Q_s_prime = self.model.forward_all(numpy_to_var(transition.resized_im_next_state))
                            a_prime = self.get_best_action(Q_s_prime)
                            Q_s = self.target_qnet.forward_all(numpy_to_var(transition.resized_im_next_state))
                            value = np.squeeze(var_to_numpy(Q_s))[a_prime]
                            del Q_s_prime
                            del Q_s
                            del a_prime
                    else:
                        with torch.no_grad():
                            Q_s = self.target_qnet.forward_all(numpy_to_var(transition.resized_im_next_state))
                            value = np.max(var_to_numpy(Q_s))
                            del Q_s

                    if self.recurrent > 0:
                        target = (transition[-1].reward + self.gamma * value)
                        
                    else:
                        target = (transition.reward + self.gamma * value)
                    del value

                if self.recurrent > 0:
                    if self.combi:
                        list_ims = get_attribute_from_list(transition, 'resized_im_state')
                        var_transitions = []
                        for im in list_ims:
                            var_transitions.append(numpy_to_var(im))
                        q_vals[i, :] = self.model.forward_all(var_transitions)
                        del var_transitions
                        del list_ims
                    else:
                        list_states = get_attribute_from_list(transition, 'resized_im_state')
                        var_transitions = tensor_to_var(numpy_to_tensor(np.array(list_states)))
                        q_vals[i, :] = self.model.forward_all(var_transitions)
                        del list_states
                        del var_transitions
                else:
                    if self.with_hist:
                        observation = self.model.get_features(numpy_to_var(transition.resized_im_state))
                        observation = torch.cat((observation, numpy_to_tensor(np.reshape(transition.action_hist,(1,70))).float()),1)
                        q_vals[i, :] = self.model(tensor_to_var(observation))
                    elif self.combi:
                        q_vals[i, :] = self.model.forward_all(numpy_to_var(transition.resized_im_state))
                    else:
                        # TODO this will probably not work here (But I don't use non-combi nets anyways...)
                        q_vals[i, :] = self.model.forward_all(numpy_to_var(transition.resized_im_state))

                targets[i, :] = q_vals[i, :].data
                if self.recurrent > 0:
                    targets[i, transition[-1].action] = numpy_to_tensor(np.array([target]))[0]
                else:
                    targets[i, transition.action] = numpy_to_tensor(np.array([target]))[0]
                del target
            
            del transitions
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.time_tracker.stop_measuring_time(start_loop_targets, "---Getting targets in loop")
            targets = tensor_to_var(targets)
            start_loss = self.time_tracker.start_measuring_time()
#             print('Q',q_vals)
#             print('T',targets)
            loss = self.loss(q_vals, targets)
            if np.isnan(float(loss)):
                raise ValueError('NaN loss detected')

            self.optimizer.zero_grad()
            loss.backward()
            
            if get_clipGrads(self.cfg_dict):
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1,1)

            self.optimizer.step()
            del loss
            if self.recurrent > 0:
                self.target_qnet.reset_hidden_state_update()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.time_tracker.stop_measuring_time(start_loss, "-Loss and backprop")
            start_metrics = self.time_tracker.start_measuring_time()
            self.calc_and_print_metrics(q_vals, targets)
            del targets
            del q_vals
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.time_tracker.stop_measuring_time(start_metrics, "Calculating metrics")
            self.time_tracker.stop_measuring_time(start_time_update, "Updating", print_gpu_usage=torch.cuda.is_available())