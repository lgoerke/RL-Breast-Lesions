import torch
from torch.autograd import Variable
import numpy as np
import shutil
import os
import torchvision.models as models
import models as m
# Need this import vor eval of optimizer
import torch.optim as optim
import resnet as res
import copy
import pickle



def list_to_variable(input_list, volatile=False, multiprocessing=False, **kwargs):
    """Makes a Variable of a Tensor on GPU if available"""
    if len(input_list) > 1:
        seq = []
        # print('01',len(input_list),flush=True)
        for t in input_list:
            seq.append(torch.unsqueeze(torch.Tensor(np.array(t)), 0))
        # print('02',len(seq),flush=True)
        tensor = torch.cat(seq)
        # print('03',tensor.size(),flush=True)
        if torch.cuda.is_available():
            tensor = tensor.cuda(**kwargs)
    else:
        try:
            # ary = torch.Tensor(np.array(input_list))
            tensor = torch.unsqueeze(torch.Tensor(np.array(input_list[0])), 0)
            if torch.cuda.is_available():
                tensor = tensor.cuda(**kwargs)
        except BaseException as e:
            print(input_list)
            print(input_list[0].size)
            raise TypeError('And another one: {}'.format(e))
    return Variable(tensor, volatile=volatile)


def numpy_to_tensor(numpyary):
    tensor = torch.from_numpy(numpyary)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def numpy_to_var(numpyary):
    var = Variable(numpy_to_tensor(numpyary))
    if torch.cuda.is_available():
        var = var.cuda()
    return var


def tensor_to_var(tensor, volatile=False, force_on_cpu=False, **kwargs):
    """Makes a Variable of a Tensor on GPU if available"""
    if torch.cuda.is_available() and not force_on_cpu:
        tensor = tensor.cuda(**kwargs)
    return Variable(tensor, volatile=volatile)


def var_to_numpy(variable):
    """Returns a numpy array of the variable data"""
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()


def var_to_cpu_tensor(variable):
    if torch.cuda.is_available():
        return variable.data.cpu()
    else:
        return variable.data


def tensor_to_numpy(tensor):
    """Returns a numpy array of the variable data"""
    if torch.cuda.is_available():
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


def get_data(tensor):
    if type(tensor) == torch.Tensor:
        return tensor.data
    else:
        return tensor


###########################################################
# Optimizer Model Training                                #
###########################################################

def get_optimizer(model_params, opti, lr, mom):
    assert opti in ['optim.Adam', 'optim.SGD', 'optim.RMSprop']
    print('Using optimizer {}'.format(opti))
    if opti == 'optim.Adam':
        optimizer = eval(opti)(model_params, lr=lr)
    else:
        optimizer = eval(opti)(model_params, lr=lr, momentum=mom)
    return optimizer


def create_copy(model):
    if isinstance(model, res.CombiNet):
        copied_model = res.CombiNet(model.feature_model, model.input_size, model.node_size, model.no_actions, model.with_pool)
    elif isinstance(model, res.RCombiNet):
        copied_model = res.RCombiNet(model.feature_model, model.input_size, model.node_size_rec, model.node_size,
                                     model.no_actions)
    elif isinstance(model, m.RCombiNet):
        copied_model = m.RCombiNet(model.feature_model, model.input_size, model.node_size_rec, model.node_size,
                                     model.no_actions)
    elif isinstance(model, m.CombiNet):
        copied_model = m.CombiNet(model.feature_model, model.input_size, model.node_size, model.no_actions)
    elif isinstance(model, m.RQNet):
        raise NotImplementedError('fix in models.py')
    elif isinstance(model, m.QNet):
        raise NotImplementedError('fix in models.py')

    state = model.state_dict()
    state_clone = copy.deepcopy(state)
    copied_model.load_state_dict(state_clone)
    return copied_model


def get_q_model(combi, recurrent, toy, inputsize, hiddensize, outputsize, feature_model, hidden_rec=None, cat=False, simple=False,with_pool=False):
    print('Combi',combi)
    print('Recurrent',recurrent)
    print('Toy',toy)
    print('Loading feature model: {}'.format(feature_model),flush=True)
    print('Cat',cat)
    print('Simple',simple)
    if recurrent > 0:
        if toy:
            model = m.RCombiNet(feature_model, inputsize, hidden_rec, hiddensize, outputsize)
        elif combi:
            model = res.RCombiNet(feature_model, inputsize, hidden_rec, hiddensize, outputsize)
        else:
            model = m.RQNet(inputsize, hiddensize, outputsize)
    elif combi:
        if toy or simple:
            model = m.CombiNet(feature_model, inputsize, hiddensize, outputsize)
            # TODODODODODOD
#             model = res.CombiNet(feature_model, inputsize, hiddensize, outputsize)
        else:    
            if cat:
                model = res.CombiNet(feature_model, inputsize, hiddensize, outputsize,with_pool)
# TODOOD comment that in for sample mammo as backend
            else:
                # TODO comment that in for resnet as backend
#                 model = res.CombiNet(feature_model, inputsize, hiddensize, outputsize)
                # TODOOD comment that in for simple mammo as backend
                model = m.CombiNet(feature_model, inputsize, hiddensize, outputsize)
    else:
        model = m.QNet(inputsize, hiddensize, outputsize)

    return model


def get_feature_model(model_string, experiment_name, load_pretrained=False, opti=None, lr=None, mom=None,
                      checkpoint_pretrained=None,
                      learn_pos=False, force_on_cpu=False,cat=False):
    print('Loading feature model: {}, expname {}, pretrained? {}, checkpoint pretrained {}, learn pos? {} , cat? {}'.format(model_string, experiment_name,load_pretrained,checkpoint_pretrained,learn_pos,cat),flush=True)
    if model_string == 'vgg':
        vgg = models.vgg16(pretrained=False)
        vgg.load_state_dict(torch.load('/mnt/synology/breast/projects/lisa/koel/vgg16-397923af.pth'))
        model = m.ModifiedVGG(vgg, learn_pos=learn_pos)
    elif model_string == 'simple':
        model = m.SimpleNet(learn_pos=learn_pos, cat=cat)
    elif model_string == 'resnet' or model_string == 'resnet_pool':
        if learn_pos:
            num_classes = 3
        elif cat:
            num_classes=5
        else:
            num_classes = 1
        model = res.get_resnet_classification(pretrained=True, num_classes=num_classes)
#         resnet = models.resnet18(pretrained=True)
#         model = m.ModifiedResNet(resnet, learn_pos=learn_pos)
    elif model_string == 'resnet_less':
        model = m.ModiefiedResNetLessFilter(learn_pos=learn_pos)                                
    elif model_string == 'fcresnet':
        if learn_pos:
            raise ValueError('Position learning not implemented for Fully Convolutional NN')
        resnet = models.resnet18(pretrained=True)
        normal_model = m.ModifiedResNet(resnet)
        if load_pretrained:
            normal_model = m.FCResNet(normal_model)

        optimizer = get_optimizer(normal_model.parameters(), opti, lr, mom)

        checkpoint_filename = os.path.join(checkpoint_pretrained, 'warmup_model_{}.pth.tar'.format(experiment_name))
        if os.path.exists(checkpoint_filename):
            model, _, initial_epoch = load_checkpoint(normal_model, optimizer, filename=checkpoint_filename)
        else:
            raise ValueError('Checkpoint {} does not exist.'.format(os.path.abspath(checkpoint_filename)))
        if load_pretrained:
            model = m.FCResNet(model)
    elif model_string == 'auto':
        model = res.get_resnet_auto(given_model=None)

    if load_pretrained and not ( model_string == 'fcresnet' or model_string =='resnet_pool'):
        optimizer = get_optimizer(model.parameters(), opti, lr, mom)

        checkpoint_filename = os.path.join(checkpoint_pretrained, 'warmup_model_{}.pth.tar'.format(experiment_name))
        if os.path.exists(checkpoint_filename):
            model, _, _ = load_checkpoint(model, optimizer, filename=checkpoint_filename)
        else:
            raise ValueError('Checkpoint {} does not exist.'.format(checkpoint_filename))
    elif load_pretrained and model_string == 'resnet_pool':
        optimizer = get_optimizer(model.parameters(), opti, lr, mom)

        checkpoint_filename = os.path.join(checkpoint_pretrained, 'warmup_model_resnet.pth.tar')
        if os.path.exists(checkpoint_filename):
            model, _, _ = load_checkpoint(model, optimizer, filename=checkpoint_filename)
        else:
            raise ValueError('Checkpoint {} does not exist.'.format(checkpoint_filename))
    if torch.cuda.is_available() and not force_on_cpu:
        print('Got model with cuda available', flush=True)
        model.cuda()
    return model


###########################################################
# Serialization                                           #
###########################################################

def save_model(model, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename)
    
def save_memory(mem, checkpoint_dir,experiment_name):
    with open(os.path.join(checkpoint_dir, 'replaymem_{}.pkl'.format(experiment_name)), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(mem, filehandle)
    
def load_memory(checkpoint_dir,experiment_name):
    mem = None
    with open(os.path.join(checkpoint_dir, 'replaymem_{}.pkl'.format(experiment_name)), 'rb') as filehandle:  
        # store the data as binary data stream
        del mem
        mem = pickle.load(filehandle)
    return mem

def save_checkpoint_and_best(history, entry_idx, smaller_better, model, optimizer, epoch, checkpoint_filename,
                             checkpoint_dir, experiment_name,replay_mem = None):
    if checkpoint_filename is not None:
        if smaller_better:
            # Evaluate according to accuracy
            # print('History',np.asarray(history))
            # print('Min History whole axis', np.asarray(history).max(axis=0))
            # print('Min history',np.asarray(history).max(axis=0)[-1])
            # print('Current',history[-1])
            # print('-1 current', history[-1][-1])
            is_best = np.asarray(history).min(axis=0)[entry_idx] >= history[-1][entry_idx]
            # print('Is best?',is_best)
        else:
            # Evaluate according to loss
            is_best = np.asarray(history).max(axis=0)[entry_idx] <= history[-1][entry_idx]
        save_model(model, optimizer, epoch, checkpoint_filename)
        if replay_mem is not None:
            save_memory(replay_mem,checkpoint_dir,experiment_name)
        if is_best:
            print('Model improved, saving better model', flush=True)
            shutil.copyfile(
                checkpoint_filename,
                os.path.join(checkpoint_dir, 'model_best_{}.pth.tar'.format(experiment_name)))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(observation_model, optimizer, filename='checkpoint.pth.tar',load_mem=False,checkpoint_dir=None, force_on_cpu=False, experiment_name='default'):
    print("=> loading checkpoint '{}'".format(filename), flush=True)
    if torch.cuda.is_available() and not force_on_cpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    observation_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']), flush=True)
    if load_mem:
        print('Loading replay memory',flush=True)
        mem = load_memory(checkpoint_dir,experiment_name)
        print('Loaded replay memory',flush=True)
        return observation_model, optimizer, start_epoch, mem
    return observation_model, optimizer, start_epoch
