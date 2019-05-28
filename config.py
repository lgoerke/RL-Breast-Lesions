# encoding: utf-8

import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

## Basic params
__C.SEED = 12345
__C.RESUME = False
__C.LR_SCHEDULE = False
__C.EXPERIMENTAL_ENV = False
__C.ROOT = os.environ.get('ROOT_DIR', "/mnt/synology/breast/archives/screenpoint1/processed_dataset")

## QNet Params
__C.QNET = edict()
__C.QNET.NETWORK = edict()
__C.QNET.NETWORK.INPUTSIZE = 25088
__C.QNET.NETWORK.HIDDENSIZE = 4096
__C.QNET.NETWORK.OUTPUTSIZE = 7

__C.QNET.TRAINING = edict()
__C.QNET.TRAINING.OPTIMIZER = 'optim.RMSprop'
__C.QNET.TRAINING.INIT_LR = 0.0001
__C.QNET.TRAINING.MOMENTUM = 0.0
__C.QNET.TRAINING.NUM_EPOCHS = 10000
__C.QNET.TRAINING.TRANSITIONS_PER_LEARNING = 100
__C.QNET.TRAINING.CLIP_GRADS = False

__C.QNET.SAVE = edict()
__C.QNET.SAVE.CHECKPOINT_DIR = 'qnet_default'
__C.QNET.SAVE.TENSORBOARD_DIR = 'qnet_default/logs'
__C.QNET.SAVE.EXPERIMENT_NAME = 'qnet'

__C.QNET.VARIANTS = edict()
__C.QNET.VARIANTS.ONE_IMG = False
__C.QNET.VARIANTS.MAX_NO_IMGS_TRAIN = -1
__C.QNET.VARIANTS.MAX_NO_IMGS_VAL = -1
__C.QNET.VARIANTS.DOUBLE = False
__C.QNET.VARIANTS.COMBI = False
__C.QNET.VARIANTS.PARAM_NOISE = False
__C.QNET.VARIANTS.RECURRENT = 0 #window size
__C.QNET.VARIANTS.RECURRENT_SIZE = 128
__C.QNET.VARIANTS.DIST_REWARD = False
__C.QNET.VARIANTS.DIST_FACTOR = 1.0
__C.QNET.VARIANTS.HIST = False

__C.QNET.EXPLO = edict()
__C.QNET.EXPLO.KAPPA = 0.5
__C.QNET.EXPLO.DECREASING_EPS = 500
__C.QNET.EXPLO.TARGET_EPS = 0.1
__C.QNET.EXPLO.TAU = 0.6
__C.QNET.EXPLO.TEST_TAU = 0.2
__C.QNET.EXPLO.TAU_SCHEDULE_EPOCHS = 0

__C.QNET.REWARDS = edict()
__C.QNET.REWARDS.GAMMA = 0.9
__C.QNET.REWARDS.ZETA = 1.0
__C.QNET.REWARDS.ETA = 10.0
__C.QNET.REWARDS.IOTA = -1.0
__C.QNET.REWARDS.CLIP_VAL = False

__C.QNET.HYPER = edict()
__C.QNET.HYPER.CLONE_FREQ = 15
__C.QNET.HYPER.MAX_STEPS = 30
__C.QNET.HYPER.REPLAYSIZE = 10000
__C.QNET.HYPER.ADD_BORDER = False


## Feature Network PARAMS
__C.FEATNET = edict()

__C.FEATNET.TRAINING = edict()
__C.FEATNET.TRAINING.OPTIMIZER = 'optim.Adam'
__C.FEATNET.TRAINING.INIT_LR = 0.001
__C.FEATNET.TRAINING.MOMENTUM = 0.1
__C.FEATNET.TRAINING.NUM_EPOCHS = 50

__C.FEATNET.SAVE = edict()
__C.FEATNET.SAVE.CHECKPOINT_DIR = 'vgg_default'
__C.FEATNET.SAVE.TENSORBOARD_DIR = 'vgg_default/logs'
__C.FEATNET.SAVE.EXPERIMENT_NAME = 'vgg'

__C.FEATNET.VARIANTS = edict()
__C.FEATNET.VARIANTS.SELECTIVE = False
__C.FEATNET.VARIANTS.CHECKPOINT_PRETRAINED = 'None'
__C.FEATNET.VARIANTS.CATEGORICAL = False


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key.'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
