def get_seed_resume_lrSchedule_root(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 seed
                1 resume flag
                2 lr_schedule flag
                3 root directory
    """
    return [cfg.SEED, cfg.RESUME, cfg.LR_SCHEDULE, cfg.ROOT]

def get_addBorder(cfg):
    return cfg.QNET.HYPER.ADD_BORDER

def get_clipGrads(cfg):
    return cfg.QNET.TRAINING.CLIP_GRADS

def get_q_net_input_hidden_output(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 inputsize
                1 hiddensize
                2 outputsize
    """
    return [cfg.QNET.NETWORK.INPUTSIZE, cfg.QNET.NETWORK.HIDDENSIZE, cfg.QNET.NETWORK.OUTPUTSIZE]


def get_q_train_opti_lr_mom_epochs_transPerUp(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 optimizer
                1 init_lr
                2 momentum
                3 num epochs
                4 transitions per learning
    """
    return [cfg.QNET.TRAINING.OPTIMIZER, cfg.QNET.TRAINING.INIT_LR, cfg.QNET.TRAINING.MOMENTUM,
            cfg.QNET.TRAINING.NUM_EPOCHS, cfg.QNET.TRAINING.TRANSITIONS_PER_LEARNING]


def get_q_save_check_tensorB_expName(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 checkpoint_dir
                1 tensorboard_dir
                2 experiment name
    """
    return [cfg.QNET.SAVE.CHECKPOINT_DIR, cfg.QNET.SAVE.TENSORBOARD_DIR, cfg.QNET.SAVE.EXPERIMENT_NAME]


def get_q_variants_oneImg_maxNumImgsT_maxNumImgsV_double_combi_paramNoise_recurrent_recSize_distReward_distFactor_hist(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 oneImg flag
                1 max number of imgs
                2 double dqn
                3 backpass also through featnet
                4 parameter noise
                5 recurrent layer (param=window size)
    """
    return [cfg.QNET.VARIANTS.ONE_IMG, cfg.QNET.VARIANTS.MAX_NO_IMGS_TRAIN, cfg.QNET.VARIANTS.MAX_NO_IMGS_VAL, cfg.QNET.VARIANTS.DOUBLE, cfg.QNET.VARIANTS.COMBI,
            cfg.QNET.VARIANTS.PARAM_NOISE, cfg.QNET.VARIANTS.RECURRENT, cfg.QNET.VARIANTS.RECURRENT_SIZE, cfg.QNET.VARIANTS.DIST_REWARD, cfg.QNET.VARIANTS.DIST_FACTOR, cfg.QNET.VARIANTS.HIST]


def get_q_explo_kappa_epochsEps_targetEps_tau_testTau_tauEpochs(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 kappa
                1 num epochs decreasing epsilon
                2 target epsilon
                3 tau
                4 test tau
                5 num epochs increasing tau
    """
    return [cfg.QNET.EXPLO.KAPPA, cfg.QNET.EXPLO.DECREASING_EPS, cfg.QNET.EXPLO.TARGET_EPS, cfg.QNET.EXPLO.TAU,
            cfg.QNET.EXPLO.TEST_TAU, cfg.QNET.EXPLO.TAU_SCHEDULE_EPOCHS]


def get_q_rewards_gamma_zeta_eta_iota_clipVal(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 gamma
                1 zeta
                2 eta
                3 clip value
    """
    return [cfg.QNET.REWARDS.GAMMA, cfg.QNET.REWARDS.ZETA, cfg.QNET.REWARDS.ETA, cfg.QNET.REWARDS.IOTA, cfg.QNET.REWARDS.CLIP_VAL]


def get_q_hyper_cloneFreq_maxSteps_replaysize(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 clone frequency
                1 max num steps
                2 replaymemory size
    """

    return [cfg.QNET.HYPER.CLONE_FREQ, cfg.QNET.HYPER.MAX_STEPS, cfg.QNET.HYPER.REPLAYSIZE]


def get_f_train_opti_lr_mom_epochs(cfg):
    """

    :param cfg:
    :return:    List of params:
                0 optimizer
                1 lr
                2 momentum
                3 number of epochs
    """
    return [cfg.FEATNET.TRAINING.OPTIMIZER, cfg.FEATNET.TRAINING.INIT_LR, cfg.FEATNET.TRAINING.MOMENTUM,
            cfg.FEATNET.TRAINING.NUM_EPOCHS]


def get_f_save_check_tensorB_expName(cfg):
    """

       :param cfg:
       :return:    List of params:
                   0 checkpoint_dir
                   1 tensorboard_dir
                   2 experiment name
       """
    return [cfg.FEATNET.SAVE.CHECKPOINT_DIR, cfg.FEATNET.SAVE.TENSORBOARD_DIR, cfg.FEATNET.SAVE.EXPERIMENT_NAME]


def get_f_variants_selectiveS_checkPretrained_cat(cfg):
    """

    :param cfg:
    :return:       List of params:
                   0 selective sampling flag
                   1 checkpoint dir of pretrained network
    """
    return [cfg.FEATNET.VARIANTS.SELECTIVE, cfg.FEATNET.VARIANTS.CHECKPOINT_PRETRAINED, cfg.FEATNET.VARIANTS.CATEGORICAL]
