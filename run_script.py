# encoding: utf-8
from utils.config_utils import get_f_save_check_tensorB_expName

import time
from pprint import pprint

from config import cfg, cfg_from_file
import argparse


def train_feat_pos(model_string, rsyncing, toy):
    from prepare_feat import prepare_and_start_training as _train_feat
    # Benchmark best convolution algorithm
    from torch.backends import cudnn
    cudnn.benchmark = True

    _train_feat(cfg, model_string, rsyncing, learn_pos=True, toy=toy)


def train_feat(model_string, rsyncing, toy):
    from prepare_feat import prepare_and_start_training as _train_feat
    # Benchmark best convolution algorithm
    from torch.backends import cudnn
    cudnn.benchmark = True

    _train_feat(cfg, model_string, rsyncing=rsyncing, learn_pos=False, toy=toy)


def train_qnet(feat_model_string, rsyncing, toy):
    from prepare_qnet import prepare_and_start_training as _train_qnet
    # Benchmark best convolution algorithm
    from torch.backends import cudnn
    cudnn.benchmark = True

    _train_qnet(cfg, feat_model_string, rsyncing, toy=toy)


def test_qnet(feat_model_string, rsyncing, toy):
    from test_qnet import test as _test_qnet
    # Benchmark best convolution algorithm
    from torch.backends import cudnn
    cudnn.benchmark = True

    _test_qnet(cfg, feat_model_string, rsyncing, toy=toy)

def get_stats(feat_model_string, rsyncing,toy):
    from get_params import get_params
    get_params(cfg, feat_model_string, rsyncing,toy=toy)

def test_vgg():
    pass


def test_feat(feat_model_string,rsyncing, toy):
    from test_feat import test as _test_feat
    _test_feat(cfg, feat_model_string, rsyncing=rsyncing, learn_pos=False, toy=toy, auto=False)


def main():
    mode_list = ['train', 'test', 'train_pos','stats']
    models_list = ['vgg', 'qnet', 'resnet', 'fcresnet','simple','auto']
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='configuration file to use.')
    parser.add_argument('--mode', help='mode (train or test')
    parser.add_argument('--model', help='model to train')
    parser.add_argument('--no-rsync', dest='no_rsync', help='use symbolic links instead', action='store_true')
    parser.add_argument(
        '--toy',
        dest='toy',
        help='use toy data set',
        action='store_true'
    )
    args = parser.parse_args()

    if args.cfg:
        print('Loading config {}.'.format(args.cfg))
        cfg_from_file(args.cfg)
    else:
        print('No config given, using standard settings.')

    pprint(cfg)

    if args.model not in models_list:
        print('Model does not exist. Choose from {}. Exiting.'.format(models_list))
    elif args.mode not in mode_list:
        print('Mode does not exist. Choose from {}. Exiting.'.format(mode_list))
    else:
        rsyncing = False if args.no_rsync else True
        if rsyncing:
            print('Rsynced data! (run_script)',flush=True)
        else:
            print('Using symbolic links! (run_script)',flush=True)

        if args.mode == 'train':
            if args.model == 'vgg':
                train_feat(model_string='vgg', rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'qnet':
                exp_name = get_f_save_check_tensorB_expName(cfg)[2]
                train_qnet(feat_model_string=exp_name, rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'simple':
                train_feat(model_string='simple', rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'resnet':
                train_feat(model_string='resnet', rsyncing=rsyncing, toy=args.toy)         
            elif args.model == 'resnet_less':
                train_feat(model_string='resnet_less', rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'fcresnet':
                train_feat(model_string='fcresnet', rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'auto':
                train_feat('auto', rsyncing=rsyncing,toy=args.toy)
        elif args.mode == 'train_pos':
            if args.model == 'vgg':
                train_feat_pos(model_string='vgg', rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'simple':
                train_feat(model_string='simple', rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'resnet':
                train_feat_pos(model_string='resnet', rsyncing=rsyncing, toy=args.toy)
            else:
                print('Use pos only for feature net training')
        elif args.mode == 'test':
            if args.model == 'vgg':
                test_vgg()
                # Not yet implemented
            elif args.model == 'qnet':
                exp_name = get_f_save_check_tensorB_expName(cfg)[2]
                test_qnet(feat_model_string=exp_name, rsyncing=rsyncing, toy=args.toy)
            elif args.model == 'resnet' or args.model == 'simple':
                exp_name = get_f_save_check_tensorB_expName(cfg)[2]
                test_feat(feat_model_string=exp_name, rsyncing=rsyncing, toy=args.toy)
                # Not yet implemented
        elif args.mode == 'stats':
            exp_name = get_f_save_check_tensorB_expName(cfg)[2]
            get_stats(feat_model_string=exp_name, rsyncing=rsyncing, toy=args.toy)


    total_time = time.time() - start_time
    print('Main program executed in {:.0f} seconds.'.format(total_time))


if __name__ == '__main__':
    main()
