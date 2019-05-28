import time
import torch
import numpy as np

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as acc

from utils.datasets import predictions_to_weights
from utils.torch_utils import tensor_to_var, list_to_variable, save_checkpoint_and_best
import gc

from utils.utils import get_gpu_memory_map
from sklearn.metrics import confusion_matrix

from trainer_feat_super import FeatTrainerSuper


class FeatTrainerPos(FeatTrainerSuper):

    def compile(self, loss_lesion, loss_row, loss_col, optimizer):
        """

        :param loss_lesion:
        :param loss_row:
        :param loss_col:
        :param optimizer:
        :return:
        """
        self.loss_lesion = loss_lesion
        self.loss_row = loss_row
        self.loss_col = loss_col
        self.optimizer = optimizer

    def train(self, train_loader, val_loader=None, initial_epoch=0, num_epochs=100):
        """

        :param train_loader:
        :param val_loader:
        :param initial_epoch:
        :param num_epochs:
        :return:
        """
        start_training_time = time.time()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10, verbose=True)

        print('Training {}...'.format(self.experiment_name), flush=True)
        history = []

        # initialize metrics
        self._metrics_names = ['loss', 'auc', 'accuracy']
        # auc(true,pred), loss(pred,true)
        self._metrics = [lambda x, y: self.loss_lesion(x, y), lambda x, y: auc(y.numpy(), x.numpy()),
                         lambda x, y: acc(y.numpy(), np.around(x.numpy()))]

        for epoch in range(initial_epoch, num_epochs):
            print('Starting epoch {}'.format(epoch), flush=True)
            print('...', flush=True)

            curr_lr = self.get_lr()
            if self.training_ended:
                break

            self.model.train()
            self.writer.add_scalar('{}/learning_rate'.format(self.experiment_name), curr_lr, epoch)

            batch_metrics_arr = np.zeros(len(self._metrics))

            loss = np.inf

            for batch_idx, batch in enumerate(train_loader):

                if self.print_memory_usage:
                    print('Batch before {}'.format(batch_idx), get_gpu_memory_map(), flush=True)

                if len(batch['image_batch']) > 0:
                    try:
                        pred_lesion, pred_row, pred_col, labels_lesion, labels_row, labels_col = self.forward(batch)
                    except RuntimeError:
                        print(len(batch['image_batch']))
                        print(batch['image_batch'][0].shape)
                        # print(batch['label_batch'])
                        print(batch)
                        raise RuntimeError('Another mystery error')
                    if pred_lesion.size()[0] > 1:
                        pred_lesion = torch.squeeze(pred_lesion)
                    elif pred_lesion.size()[0] == 1 and pred_lesion.size()[1] == 1:
                        pred_lesion = pred_lesion.view(1)

                    print('Lesion prediction', pred_lesion, flush=True)
                    print('Lesion label', labels_lesion, flush=True)
                    print('Row prediction', pred_row, flush=True)
                    print('Row label', labels_row, flush=True)
                    print('Col prediction', pred_col, flush=True)
                    print('Col label', labels_col, flush=True)

                    self.optimizer.zero_grad()
                    loss_seq = [self.loss_lesion(pred_lesion, labels_lesion), self.loss_row(pred_row, labels_row),
                                self.loss_col(pred_col, labels_col)]

                    grad_seq = [loss_seq[0].new(1).fill_(1) for _ in range(len(loss_seq))]
                    torch.autograd.backward(loss_seq, grad_seq)
                    self.optimizer.step()

                    if self.print_memory_usage:
                        print('Batch between {}'.format(batch_idx), get_gpu_memory_map(), flush=True)

                    self.calc_and_print_metrics(pred_lesion, labels_lesion, batch_idx, epoch * len(train_loader), training=True)

                    curr_metric = self.loss_row(pred_row, labels_row).data
                    self.writer.add_scalar(
                        '{}/train/row_{}'.format(self.experiment_name, _metric_name),
                        curr_metric, epoch * len(train_loader) + batch_idx)
                    self.optimizer.zero_grad()

                    curr_metric = self.loss_col(pred_col, labels_col).data
                    self.writer.add_scalar(
                        '{}/train/col_{}'.format(self.experiment_name, _metric_name),
                        curr_metric, epoch * len(train_loader) + batch_idx)
                    self.optimizer.zero_grad()


                if self.print_memory_usage:
                    print('Batch after {}'.format(batch_idx), get_gpu_memory_map(), flush=True)
                    print('---', flush=True)

            if self.print_memory_usage:
                print('Before collect', get_gpu_memory_map(), flush=True)
            gc.collect()
            if self.print_memory_usage:
                print('After collect', get_gpu_memory_map(), flush=True)

            # Validate only each 5 epochs to save time
            if epoch in np.arange(initial_epoch, num_epochs, 5):
                print('Starting validation..', flush=True)
                val_metrics = self.evaluate(val_loader, epoch)
                history.append(val_metrics)
                print('Validation finished', flush=True)

                for _val, _metric_name in zip(val_metrics, self._metrics_names):
                    self.writer.add_scalar('{}/val/{}'.format(self.experiment_name, _metric_name), _val, epoch)

                print('Start saving', flush=True)
                print('...', flush=True)
                # Evaluate according to accuracy
                save_checkpoint_and_best(history, entry_idx=2, smaller_better=False, model=self.model, optimizer=self.optimizer, epoch=epoch, checkpoint_filename=self.checkpoint_filename,
                                     checkpoint_dir=self.checkpoint_dir, experiment_name=self.experiment_name)

            lr_scheduler.step(float(loss))

        # Training loop exited normally
        else:
            self.training_ended = True
        total_time = time.time() - start_training_time
        print('Training took {:.0f} seconds'.format(total_time), flush=True)

    def forward(self, batch, **kwargs):
        """

        :param batch:
        :param kwargs:
        :return:
        """
        data = list_to_variable(batch['image_batch'], volatile=kwargs.get('volatile', False))
        labels_lesion = tensor_to_var(torch.Tensor(np.array(batch['label_batch'])))
        labels_row = tensor_to_var(torch.Tensor(np.array(batch['center_row_batch'])))
        labels_col = tensor_to_var(torch.Tensor(np.array(batch['center_col_batch'])))

        pred = self.model(data)

        return pred[:, 0], pred[:, 1], pred[:, 2], labels_lesion, labels_row, labels_col

    def evaluate(self, loader, epoch):
        """

        :param loader:
        :param epoch:
        :return:
        """
        if self.print_memory_usage:
            print('Start of validation', get_gpu_memory_map(), flush=True)
        start_time = time.time()
        self.model.eval()

        val_metrics_arr = np.zeros(len(self._metrics))

        list_prediction_lesion = []
        list_prediction_row = []
        list_prediction_col = []
        list_label_lesion = []
        list_label_row = []
        list_label_col = []

        with torch.no_grad():
            for val_idx, batch in enumerate(loader):
                if self.print_memory_usage:
                    print('Batch {}'.format(val_idx), get_gpu_memory_map(), flush=True)
                if len(batch['image_batch']) > 0:
                    pred_lesion, pred_row, pred_col, labels_lesion, labels_row, labels_col = self.forward(batch)
                    if pred_lesion.size()[0] > 1:
                        pred_lesion = torch.squeeze(pred_lesion)

                    pred_lesion = F.sigmoid(pred_lesion)
                    if self.print_memory_usage:
                        print('Batch between {}'.format(val_idx), get_gpu_memory_map(), flush=True)
                    list_prediction_lesion.append(pred_lesion.data.cpu())
                    list_prediction_row.append(pred_row.data.cpu())
                    list_prediction_col.append(pred_col.data.cpu())

                    list_label_lesion.append(labels_lesion.data.cpu())
                    list_label_row.append(labels_row.data.cpu())
                    list_label_col.append(labels_col.data.cpu())
                    if self.print_memory_usage:
                        print('Batch after {}'.format(val_idx), get_gpu_memory_map(), flush=True)
                        print('---', flush=True)

        self.writer.add_scalars('{}/val/distribution_gt'.format(self.experiment_name),
                                {'has lesion': np.sum(torch.cat(list_label_lesion).numpy()),
                                 'no lesion': len(torch.cat(list_label_lesion)) - np.sum(
                                     torch.cat(list_label_lesion).numpy())},
                                epoch)
        self.writer.add_scalars('{}/val/distribution_pred'.format(self.experiment_name),
                                {'has lesion': np.sum(np.around(torch.cat(list_prediction_lesion).numpy())),
                                 'no lesion': len(torch.cat(list_prediction_lesion)) - np.sum(
                                     np.around(torch.cat(list_prediction_lesion).numpy()))},
                                epoch)
        tn, fp, fn, tp = confusion_matrix(torch.cat(list_label_lesion).numpy(),
                                          np.around(torch.cat(list_prediction_lesion).numpy())).ravel()
        self.writer.add_scalars('{}/val/correct'.format(self.experiment_name),
                                {'tp': tp,
                                 'tn': tn},
                                epoch)
        self.writer.add_scalars('{}/val/incorrect'.format(self.experiment_name),
                                {'fp': fp,
                                 'fn': fn},
                                epoch)

        metric_dict = {}
        for _metric_idx, (_metric, _metric_name) in enumerate(zip(self._metrics, self._metrics_names)):
            if _metric_idx == 0:
                curr_metric_lesion = _metric(torch.cat(list_prediction_lesion), torch.cat(list_label_lesion)).data
            else:
                curr_metric_lesion = _metric(torch.cat(list_prediction_lesion), torch.cat(list_label_lesion))
            val_metrics_arr[_metric_idx] = curr_metric_lesion
            # val_metrics_arr[_metric_idx] = (val_idx * val_metrics_arr[_metric_idx] + curr_value) / (val_idx + 1)
            metric_dict['val_{}'.format(_metric_name)] = '{:.4f}'.format(val_metrics_arr[_metric_idx])

        curr_metric = self.loss_row(torch.cat(list_prediction_row), torch.cat(list_label_row)).data
        self.writer.add_scalar('{}/val/row_loss'.format(self.experiment_name), curr_metric, epoch)
        self.optimizer.zero_grad()

        curr_metric = self.loss_row(torch.cat(list_prediction_col), torch.cat(list_label_col)).data
        self.writer.add_scalar('{}/val/col_loss'.format(self.experiment_name), curr_metric, epoch)
        self.optimizer.zero_grad()

        total_time = time.time() - start_time
        print('Evaluation took {:.0f} seconds.'.format(total_time), flush=True)
        gc.collect()
        return val_metrics_arr

    def predict_dataset(self, dataloader):
        """

        :param dataloader:
        :return:
        """
        start_time = time.time()
        self.model.eval()
        list_prediction_lesion = []
        list_label_lesion = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if len(batch['image_batch']) > 0:
                    pred_lesion, _, _, labels_lesion, _, _ = self.forward(batch)
                    if pred_lesion.size()[0] > 1:
                        pred_lesion = torch.squeeze(pred_lesion)

                    pred_lesion = F.sigmoid(pred_lesion)

                    list_prediction_lesion.append(pred_lesion.data.cpu())

                    list_label_lesion.append(labels_lesion.data.cpu())

        total_time = time.time() - start_time
        print('Prediction took {:.0f} seconds.'.format(total_time), flush=True)
        weights = predictions_to_weights(torch.cat(list_prediction_lesion), torch.cat(list_label_lesion))
        gc.collect()
        return weights
