import argparse

import deepspeed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cheatsheet', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--test_interval', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_epochs', type=int, default=1000)
    
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--ckpt_id', default=None)
    parser.add_argument('--save_interval', type=int, default=10)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()