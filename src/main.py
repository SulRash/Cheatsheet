from models.cnn import CNN

import torch
import torch.nn as nn
import deepspeed

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


from utils.utils import set_random_seed
from utils.arguments import get_args
from utils.data import get_cifar10, get_dataloaders
from utils.loops import train_cifar, valid_cifar, test_cifar

def main(args):
    if args.dataset == "cifar":
        train_data, valid_data, test_data = get_cifar10()
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(train_data, valid_data, test_data, args.batch_size)
    
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    set_random_seed(args.seed)
    torch.distributed.barrier()

    model = CNN()
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters)
    
    if args.load_dir and args.ckpt_id:
        model.load_checkpoint(args.load_dir, args.ckpt_id)

    for epoch in range(args.train_epochs):

        model.train()
        train_cifar(model, train_dataloader)
        model.eval()
        valid_cifar(model, valid_dataloader)

        if not epoch % args.test_interval:
            test_cifar(model, test_dataloader)
        
        if not epoch % args.save_interval:
            model.save_checkpoint(args.save_dir, epoch)

if __name__ == "__main__":
    args = get_args()
    main(args)