import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as acc

from utils.datasets import predictions_to_weights
from utils.torch_utils import tensor_to_var, list_to_variable, save_checkpoint_and_best, var_to_cpu_tensor, var_to_numpy, numpy_to_tensor
import gc

from utils.utils import TimeTracker
from trainer_feat_super import FeatTrainerSuper


class FeatTrainerCat(FeatTrainerSuper):

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
        self._metrics_names = ['loss', 'accuracy']
        # auc(true,pred), loss(pred,true)
        self._metrics = [lambda x, y: self.loss(x, y),
                         lambda x, y: acc(var_to_numpy(y), np.argmax(var_to_numpy(x),axis=1))]
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
#                 print('Starting batch {}'.format(batch_idx), flush=True)
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
                       
                    if batch_idx == 0:
                        # print('Prediction shape', pred.shape)
                        predictions = F.softmax(pred,dim=1)
                        predictions = np.argmax(var_to_numpy(predictions),axis=1)
#                         print('Prediction shape', pred.shape,flush=True)
#                         print('Softmax predictions',pred,flush=True)
#                         print('Prediction shape', predictions.shape,flush=True)
#                         print('Discrete predictions',predictions,flush=True)
                        for i,image in enumerate(batch['image_batch']):
                            if predictions[i]==labels[i]:
                                self.writer.add_image('Image {} Correct Prediction {}'.format(i,predictions[i]), image, epoch)
                            else:
                                self.writer.add_image('Image {} Wrong Prediction {}, Label was {}'.format(i,predictions[i],labels[i]), image, epoch)

                    # print(pred.is_cuda, labels.is_cuda)
                    loss = self.loss(pred, labels.long())
                    if np.isnan(float(loss)):
                        raise ValueError('NaN loss detected')

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.time_tracker.stop_measuring_time(start_time_batch,"Batch before metrics",self.print_memory_usage)

                    self.calc_and_print_metrics(pred, labels.long(), batch_idx, epoch * len(train_loader), training=True)
                    
#                 if batch_idx > 3:
#                     break

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
            save_checkpoint_and_best(history, entry_idx=0, smaller_better=True, model=self.model, optimizer=self.optimizer, epoch=epoch, checkpoint_filename=self.checkpoint_filename,
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
#         print('Pred',pred,flush=True)
#         print('Labels',labels,flush=True)
        return pred, labels

    def forward_feature(self, batch, **kwargs):
        data = list_to_variable(batch['image_batch'], volatile=kwargs.get('volatile', False))
        labels = tensor_to_var(torch.Tensor(np.array(batch['label_batch'])))

        features = self.model.features(data)
        return features, labels

    def evaluate(self, loader, epoch, for_testing=False):
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
                if for_testing:
                    print('Starting Val Batch {}'.format(val_idx),flush=True)
                if len(batch['image_batch']) > 0:

                    pred, labels = self.forward(batch)
                    predictions = F.softmax(pred,dim=1)

                    prediction_list.append(predictions)
                    label_list.append(labels.long())

                self.time_tracker.stop_measuring_time(start_time_batch,'One batch eval',self.print_memory_usage)
#                 if val_idx > 3:
#                     break

        start_time_metrics = self.time_tracker.start_measuring_time()
        cnt_l, cnt_p, cmat = self.calc_and_print_conf_mat_mult(torch.cat(prediction_list), torch.cat(label_list), epoch, for_testing=for_testing)
        print(cnt_l)
        print(cnt_p)
        print(cmat)
        print('Ground truth has cat 0: {}, cat 1: {}, cat 2: {}, cat 3: {}, cat 4: {}'.format(cnt_l[0],cnt_l[1],cnt_l[2],cnt_l[3],cnt_l[4]),flush=True)
        print('Prediction has cat 0: {}, cat 1: {}, cat 2: {}, cat 3: {}, cat 4: {}'.format(cnt_p[0],cnt_p[1],cnt_p[2],cnt_p[3],cnt_p[4]),flush=True)
        print('Correct classifications 0: {}, 1: {}, 2: {}, 3: {}, 4: {}'.format(cmat[0,0],cmat[1,1],cmat[2,2],cmat[3,3],cmat[4,4]),flush=True)
        print('Off by one 0&1: {}, 1&2: {}, 2&3: {}, 3&4: {}'.format(cmat[0,1]+cmat[1,0], cmat[2,1]+cmat[1,2], cmat[2,3]+cmat[3,2], cmat[3,4]+cmat[4,3]),flush=True)
        print('Off by two 0&2: {}, 1&3: {}, 2&4: {}'.format(cmat[0,2]+cmat[2,0], cmat[3,1]+cmat[1,3], cmat[2,4]+cmat[4,2]),flush=True)
        print('Off by more 0&3: {}, 1&4: {}, 0&4: {}'.format(cmat[0,3]+cmat[3,0], cmat[4,1]+cmat[1,4], cmat[0,4]+cmat[4,0]),flush=True)
        
        val_metrics_arr = self.calc_and_print_metrics(torch.cat(prediction_list), torch.cat(label_list), 0, epoch,False, for_testing=for_testing)
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

    #TODO
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
                    pred = F.softmax(pred)
                    predictions.append(pred)
                    labels.append(lbls)

        self.time_tracker.stop_measuring_time(start_time_prediction,"Prediction whole dataset",self.print_memory_usage)
        weights = predictions_to_weights(torch.cat(predictions), torch.cat(labels))
        gc.collect()
        return weights

