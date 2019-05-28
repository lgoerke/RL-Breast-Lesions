import numpy as np
from utils.torch_utils import *
from sklearn.metrics import confusion_matrix
import os
from tensorboardX import SummaryWriter
import torch
from torch import nn
from utils.utils import TimeTracker


class FeatTrainerSuper(object):
    def __init__(self, model, log_dir='', checkpoint_dir='', checkpoint_filename=None, experiment_name='vgg',
                 for_testing=False, print_memory_usage=False):
        """

        :param model:
        :param log_dir:
        :param checkpoint_dir:
        :param checkpoint_filename:
        :param experiment_name:
        :param print_memory_usage:
        """
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must inherit from torch.nn.Module')

        self.model = model

        if torch.cuda.is_available():
            self.print_memory_usage = print_memory_usage
        else:
            self.print_memory_usage = False

        self.optimizer = None

        self.training_ended = False

        self._metrics = []
        self._metrics_names = []

        self.batch_metrics_arr = None
        self.running_avg_arr = None

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_filename)
        self.experiment_name = experiment_name
        self.log_dir = log_dir

        cnt = 0
        while os.path.exists(self.checkpoint_filename) and not for_testing:
            cnt += 1
            self.checkpoint_dir = '{}_{}'.format(self.checkpoint_dir, cnt)
            self.checkpoint_filename = os.path.join(self.checkpoint_dir, checkpoint_filename)
        if not os.path.isdir('{}'.format(self.checkpoint_dir)):
            os.makedirs('{}'.format(self.checkpoint_dir))
        cnt = 0
        while os.path.exists(self.log_dir) and not for_testing:
            cnt += 1
            self.log_dir = '{}_{}'.format(self.log_dir, cnt)
        if not os.path.isdir('{}'.format(self.log_dir)):
            os.makedirs('{}'.format(self.log_dir))

        self.writer = SummaryWriter(os.path.join(self.log_dir, self.experiment_name))

        if torch.cuda.is_available():
            print('CUDA available. Enabling GPU model.', flush=True)
            self.model = self.model.cuda()
            
        self.time_tracker = TimeTracker()

    def get_lr(self):
        curr_lr = self.optimizer.param_groups[0]['lr']
        if curr_lr < 1e-7:
            print('Learning rate is {}, stopping training...'.format(curr_lr), flush=True)
            self.training_ended = True
        return curr_lr

    def calc_and_print_metrics(self, pred, labels, batch_idx, epoch_x_size_loader, training=False , for_testing=False):
#         print(pred)
#         print(labels)
#         print(pred.shape)
#         print(labels.shape)
        scalar_string = 'train' if training else 'val'
        # metric_dict = {}
        metric_dict = np.zeros(len(self._metrics))
        for _metric_idx, (_metric, _metric_name) in enumerate(zip(self._metrics, self._metrics_names)):

            metric_dict[_metric_idx] = get_data(_metric(pred, labels))
            
            if not for_testing:
                if training:
                    self.batch_metrics_arr[_metric_idx] = (batch_idx * self.batch_metrics_arr[
                        _metric_idx] + metric_dict[_metric_idx]) / (
                                                                  batch_idx + 1)
                    self.running_avg_arr[_metric_idx] = ((epoch_x_size_loader + batch_idx) * self.running_avg_arr[
                        _metric_idx] + metric_dict[_metric_idx]) / (epoch_x_size_loader + batch_idx + 1)

                self.writer.add_scalars(
                        '{}/{}/{}'.format(self.experiment_name, scalar_string, _metric_name),
                        {'value': metric_dict[_metric_idx]}, epoch_x_size_loader + batch_idx)

                if training:
                    self.writer.add_scalars(
                        '{}/{}/{}'.format(self.experiment_name, scalar_string, _metric_name),
                        {'running_avg': self.running_avg_arr[_metric_idx]}, epoch_x_size_loader + batch_idx)
                    self.writer.add_scalars(
                        '{}/{}/{}'.format(self.experiment_name, scalar_string, _metric_name),
                        {'batch_avg': self.batch_metrics_arr[_metric_idx]}, epoch_x_size_loader + batch_idx)
                    # During training only evaluate loss
                    break
            else:
                print(_metric_name,metric_dict[_metric_idx],flush=True)
            self.optimizer.zero_grad()
        return metric_dict

    def calc_and_print_conf_mat_mult(self, prediction_list, label_list, epoch,for_testing=False):
        from collections import Counter
        cnt_l = Counter(var_to_numpy(label_list))
        cnt_p = Counter(np.argmax(var_to_numpy(prediction_list),axis=1))
        cmat = confusion_matrix(var_to_numpy(label_list), np.argmax(var_to_numpy(prediction_list),axis=1))
        if not for_testing:
            self.writer.add_scalars('{}/val/distribution_gt'.format(self.experiment_name),
                                    {'cat 0': cnt_l[0],
                                     'cat 1': cnt_l[1],
                                     'cat 2': cnt_l[2],
                                     'cat 3': cnt_l[3],
                                     'cat 4': cnt_l[4]},
                                    epoch)
            self.writer.add_scalars('{}/val/distribution_pred'.format(self.experiment_name),
                                    {'cat 0': cnt_p[0],
                                     'cat 1': cnt_p[1],
                                     'cat 2': cnt_p[2],
                                     'cat 3': cnt_p[3],
                                     'cat 4': cnt_p[4]},
                                    epoch)
            self.writer.add_scalars('{}/val/correct'.format(self.experiment_name),
                                    {'Correct 0': cmat[0,0],
                                     'Correct 1': cmat[1,1],
                                    'Correct 2': cmat[2,2],
                                    'Correct 3': cmat[3,3],
                                    'Correct 4': cmat[4,4]},
                                    epoch)
            self.writer.add_scalars('{}/val/off_by_one'.format(self.experiment_name),
                                    {'0 & 1': cmat[0,1] + cmat[1,0],
                                     '1 & 2': cmat[1,2] + cmat[2,1],
                                    '2 & 3': cmat[2,3] + cmat[3,2],
                                    '3 & 4': cmat[3,4] + cmat[4,3]},
                                    epoch)
            self.writer.add_scalars('{}/val/off_by_two'.format(self.experiment_name),
                                    {'0 & 2': cmat[0,2] + cmat[2,0],
                                     '1 & 3': cmat[1,3] + cmat[3,1],
                                    '2 & 4': cmat[2,4] + cmat[4,2]},
                                    epoch)
            self.writer.add_scalars('{}/val/off_by_more'.format(self.experiment_name),
                                    {'0 & 3': cmat[0,3] + cmat[3,0],
                                     '1 & 4': cmat[1,4] + cmat[4,1],
                                    '0 & 4': cmat[0,4] + cmat[4,0]},
                                    epoch)
        return cnt_l, cnt_p, cmat
        
    
    def calc_and_print_conf_mat(self, prediction_list, label_list, epoch,for_testing=False):
        if not for_testing:
            self.writer.add_scalars('{}/val/distribution_gt'.format(self.experiment_name),
                                    {'has lesion': np.sum(tensor_to_numpy(label_list)),
                                     'no lesion': len(label_list) - np.sum(tensor_to_numpy(label_list))},
                                    epoch)
            self.writer.add_scalars('{}/val/distribution_pred'.format(self.experiment_name),
                                    {'has lesion': np.sum(np.around(tensor_to_numpy(prediction_list))),
                                     'no lesion': len(prediction_list) - np.sum(
                                         np.around(tensor_to_numpy(prediction_list)))},
                                    epoch)
        tn, fp, fn, tp = confusion_matrix(tensor_to_numpy(label_list),
                                          np.around(tensor_to_numpy(prediction_list))).ravel()
        
        if not for_testing:
            self.writer.add_scalars('{}/val/correct'.format(self.experiment_name),
                                    {'tp': tp,
                                     'tn': tn},
                                    epoch)
            self.writer.add_scalars('{}/val/incorrect'.format(self.experiment_name),
                                    {'fp': fp,
                                     'fn': fn},
                                    epoch)
        else:
            print('label',label_list)
            print('prediction',prediction_list)
            print('has lesion ground truth', np.sum(tensor_to_numpy(label_list)),flush=True)
            print('no lesion ground truth', len(label_list) - np.sum(tensor_to_numpy(label_list)),flush=True)
            print('has lesion prediction', np.sum(np.around(tensor_to_numpy(prediction_list))),flush=True)
            print('no lesion prediction', len(prediction_list) - np.sum(
                                         np.around(tensor_to_numpy(prediction_list))),flush=True)
            print('TP: {}, FP: {}, FN: {}, TN: {}'.format(tp,fp,fn,tn),flush=True)
            print(confusion_matrix(tensor_to_numpy(label_list),
                                          np.around(tensor_to_numpy(prediction_list))))
            
