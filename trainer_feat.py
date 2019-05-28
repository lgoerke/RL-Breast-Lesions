import time
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as acc

from utils.datasets import predictions_to_weights
from utils.torch_utils import tensor_to_var, list_to_variable, save_checkpoint_and_best, var_to_cpu_tensor
import gc

from utils.utils import TimeTracker
from trainer_feat_super import FeatTrainerSuper


class FeatTrainer(FeatTrainerSuper):

    def compile(self, loss, optimizer):
        """

        :param loss:
        :param optimizer:
        :return:
        """
        self.loss = loss
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
        self._metrics = [lambda x, y: self.loss(x, y), lambda x, y: auc(y.numpy(), x.numpy()),
                         lambda x, y: acc(y.numpy(), np.around(x.numpy()))]
        self.running_avg_arr = np.zeros(len(self._metrics))

        for epoch in range(initial_epoch, num_epochs):
            print('Starting epoch {}'.format(epoch), flush=True)
            print('...', flush=True)
            start_time_epoch = self.time_tracker.start_measuring_time()

            curr_lr = self.get_lr()
            if self.training_ended:
                break

            self.model.train()
            self.writer.add_scalar('{}/learning_rate'.format(self.experiment_name), curr_lr, epoch)

            self.batch_metrics_arr = np.zeros(len(self._metrics))

            loss = np.inf

            for batch_idx, batch in enumerate(train_loader):
                start_time_batch = self.time_tracker.start_measuring_time()
                if len(batch['image_batch']) > 0:
                    try:
                        pred, labels = self.forward(batch, volatile=False)
                    except BaseException as e:
                        print('Exception start')
                        print(len(batch['image_batch']))
                        print(batch['image_batch'][0].shape)
                        # print(batch['label_batch'])
                        # print(batch)
                        raise RuntimeError('Another mystery error: {}'.format(e))
                    if pred.size()[0] > 1:
                        pred = torch.squeeze(pred)
                    elif pred.size()[0] == 1 and pred.size()[1] == 1:
                        pred = pred.view(1)
                        
                    if batch_idx == 0:
                        # print('Prediction shape', pred.shape)
                        for i,image in enumerate(batch['image_batch']):
                            if predictions[i]==labels[i]:
                                self.writer.add_image('TP or TN image {}'.format(i), image, epoch)
                            elif predictions[i] == 1:
                                self.writer.add_image('FP image {}'.format(i), image, epoch)
                            elif predictions[i] == 0:
                                self.writer.add_image('FN image {}'.format(i), image, epoch)
                                
                    # print(pred.is_cuda, labels.is_cuda)
                    loss = self.loss(pred, labels)
                    if np.isnan(float(loss)):
                        raise ValueError('NaN loss detected')

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.time_tracker.stop_measuring_time(start_time_batch,"Batch before metrics",self.print_memory_usage)

                    self.calc_and_print_metrics(pred, labels, batch_idx, epoch * len(train_loader), training=True)

                self.time_tracker.stop_measuring_time(start_time_batch,'One batch',self.print_memory_usage)
                # print('After metric',flush=True)
                # print('...',flush=True)


            gc.collect()

            # Validate only each 5 epochs to save time
            # if epoch in np.arange(initial_epoch, num_epochs, 5):
            print('Starting validation..', flush=True)
            val_metrics = self.evaluate(val_loader, epoch)

            history.append(val_metrics)
            print('Validation finished', flush=True)

            for _val, _metric_name in zip(val_metrics, self._metrics_names):
                self.writer.add_scalar('{}/val/{}'.format(self.experiment_name, _metric_name), _val, epoch)

            print('Start saving', flush=True)
            print('...', flush=True)
            print('History',history,flush=True)
            start_time_saving = self.time_tracker.start_measuring_time()
            # Evaluate according to Accuracy 
            save_checkpoint_and_best(history, entry_idx=2, smaller_better=False, model=self.model, optimizer=self.optimizer, epoch=epoch, checkpoint_filename=self.checkpoint_filename,
                                     checkpoint_dir=self.checkpoint_dir, experiment_name=self.experiment_name)
            self.time_tracker.stop_measuring_time(start_time_saving,"Saving the checkpoint",self.print_memory_usage)
            lr_scheduler.step(float(loss))
            self.time_tracker.stop_measuring_time(start_time_epoch,"One epoch",self.print_memory_usage)


        # Training loop exited normally
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
        labels = tensor_to_var(torch.Tensor(np.array(batch['label_batch'])))

        pred = self.model(data)

        return pred, labels

    def forward_feature(self, batch, **kwargs):
        data = list_to_variable(batch['image_batch'], volatile=kwargs.get('volatile', False))
        labels = tensor_to_var(torch.Tensor(np.array(batch['label_batch'])))

        features = self.model.features(data)
        return features, labels

    def evaluate(self, loader, epoch,for_testing=False):
        """

        :param loader:
        :param epoch:
        :return:
        """
        start_time_eval = self.time_tracker.start_measuring_time()
        self.model.eval()

        prediction_list = []
        label_list = []

        with torch.no_grad():
            for val_idx, batch in enumerate(loader):
                start_time_batch = self.time_tracker.start_measuring_time()

                if len(batch['image_batch']) > 0:

                    pred, labels = self.forward(batch)
                    if pred.size()[0] > 1:
                        pred = torch.squeeze(pred)
                    pred = F.sigmoid(pred)

                    prediction_list.append(var_to_cpu_tensor(pred))
                    label_list.append(var_to_cpu_tensor(labels))

                self.time_tracker.stop_measuring_time(start_time_batch,'One batch eval',self.print_memory_usage)

        start_time_metrics = self.time_tracker.start_measuring_time()
        self.calc_and_print_conf_mat(torch.cat(prediction_list), torch.cat(label_list), epoch,for_testing)
        val_metrics_arr = self.calc_and_print_metrics(torch.cat(prediction_list), torch.cat(label_list), 0, epoch,
                                                      False)
        self.time_tracker.stop_measuring_time(start_time_metrics,"Calculation metrics",self.print_memory_usage)
        self.time_tracker.stop_measuring_time(start_time_eval,"Evaluation",self.print_memory_usage)
        gc.collect()
        return val_metrics_arr

    def get_feature_vectors(self, dataloader):
        start_time_prediction = self.time_tracker.start_measuring_time()
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if len(batch['image_batch']) > 0:
                    feats, lbls = self.forward_feature(batch)
                    features.append(feats)
                    labels.append(lbls)

        self.time_tracker.stop_measuring_time(start_time_prediction, "Getting feature vectors", self.print_memory_usage)
        gc.collect()
        return features, labels

    def predict_dataset(self, dataloader):
        """

        :param dataloader:
        :return:
        """
        start_time_prediction = self.time_tracker.start_measuring_time()
        self.model.eval()
        predictions = []
        labels = []
        with torch.no_grad():

            for idx, batch in enumerate(dataloader):
                if len(batch['image_batch']) > 0:
                    pred, lbls = self.forward(batch)
                    if pred.size()[0] > 1:
                        pred = torch.squeeze(pred)
                    pred = F.sigmoid(pred)
                    predictions.append(pred)
                    labels.append(lbls)

        self.time_tracker.stop_measuring_time(start_time_prediction,"Prediction whole dataset",self.print_memory_usage)
        weights = predictions_to_weights(torch.cat(predictions), torch.cat(labels))
        gc.collect()
        return weights
