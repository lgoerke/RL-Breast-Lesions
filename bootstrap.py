# encoding: utf-8
import sys
import argparse
import subprocess
import time
import torch
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with reinforcement learning model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--mode',
        dest='mode',
        help='which mode to use (train, test)',
        type=str
    )
    parser.add_argument(
        '--model',
        dest='model',
        help='which model to use (vgg, qnet, resnet)',
        type=str
    )
    parser.add_argument(
        '--toy',
        dest='toy',
        help='use toy data set',
        action='store_true'
    )
    parser.add_argument(
        '--no-rsync',
        dest='no_rsync',
        help='use symbolic links instead',
        action='store_true'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    seed = 4517
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    args = parse_args()
    if args.toy:
        paths = [
            '/mnt/synology/breast/projects/lisa/toy_data/',
            '/input'
        ]
    else:
        paths = [
            '/mnt/synology/breast/archives/screenpoint3/processed_dataset/',
            '/input'
        ]

    if args.no_rsync:
        print('Working with symbolic links', flush=True)
        data_cmd = ['ln', '-s']
    else:
        print('Rsyncing data', flush=True)
        data_cmd = ['/usr/bin/rsync', '-am', '--stats']
        

    subprocess.call(data_cmd + paths)
    print('Preparing dataset took {:.2f} seconds.'.format(time.time() - start_time), flush=True)

    if args.no_rsync:
        if args.toy:
            cmd = 'python src/run_script.py --cfg {} --mode {} --model {} --no-rsync --toy'.format(args.cfg_file, args.mode,
                                                                                             args.model)
        else:
            cmd = 'python src/run_script.py --cfg {} --mode {} --model {} --no-rsync'.format(args.cfg_file, args.mode, args.model)
    else:
        if args.toy:
            cmd = 'python src/run_script.py --cfg {} --mode {} --model {} --toy'.format(args.cfg_file, args.mode, args.model)
        else:
            cmd = 'python src/run_script.py --cfg {} --mode {} --model {}'.format(args.cfg_file, args.mode, args.model)
    print('Running {}'.format(cmd), flush=True)
    subprocess.call(cmd.split(' '))


if __name__ == '__main__':
    main()
