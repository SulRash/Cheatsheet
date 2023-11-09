import argparse

import deepspeed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cheatsheet', action="store_true")
    parser.add_argument('--randomize_sheet', action="store_true")
    parser.add_argument('--one_image', action="store_true")
    parser.add_argument('--one_image_per_class', action="store_true")
    parser.add_argument('--img_per_class', default=0)
    parser.add_argument('--cs_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'mnist'], default='cifar10')

    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'flexivit_base', 'vit_large_patch14_224', 'vit_huge_patch14_224', 'vit_base_patch8_224', 'vit_small_patch8_224', 'deit3_base_patch16_384_in21ft1k'])

    parser.add_argument('--exp_name', type=str, default='Default')
    parser.add_argument('--test_interval', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_epochs', type=int, default=250)
    
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--ckpt_id', default=None)
    parser.add_argument('--save_interval', type=int, default=10)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()