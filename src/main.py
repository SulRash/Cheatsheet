from models.cnn import CNN

from utils.utils import set_random_seed
from utils.arguments import get_args
from utils.data import get_cifar10, get_dataloaders

import torch
import deepspeed

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

def main(args):
    if args.dataset == "cifar":
        train_data, valid_data, test_data = get_cifar10()
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(train_data, valid_data, test_data, args.batch_size)
    
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    set_random_seed(args.seed)

    torch.distributed.barrier()

    model = CNN()

    model, optimizer, _, _ = deepspeed.initialize(args=args, model=model)

    
    for epoch in range(args.train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            loss = model(batch)
            model.backward(loss)
            model.step()
        
        model.eval()
        for step, batch in enumerate(valid_dataloader):
            loss = model(batch)
            print("Validation loss: ", loss)

if __name__ == "__main__":
    args = get_args()
    main(args)