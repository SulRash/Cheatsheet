import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()