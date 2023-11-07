import torch
import deepspeed
import wandb

import torch.distributed as dist

from models import get_model

from utils.utils import set_random_seed, setup_experiment
from utils.arguments import get_args
from utils.data import get_cifar, get_dataloader
from utils.loops import *

def main(args):

    train_data, test_data, num_classes = get_cifar(args)
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()

    if dist.get_rank() == 0:
        run = wandb.init(
            project='Cheatsheet',
            notes=args.exp_name,
            config=vars(args)
        )

        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

    test_dataloader = get_dataloader(test_data, args.batch_size)

    set_random_seed(args.seed)
    dist.barrier()

    model = get_model(args.model, num_classes, args.cs_size)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, _, train_dataloader, _ = deepspeed.initialize(args=args, model=model, training_data=train_data, model_parameters=parameters)
    
    if args.load_dir and args.ckpt_id:
        model.load_checkpoint(args.load_dir, args.ckpt_id)

    # Actual training loop
    for epoch in range(args.train_epochs):

        if dist.get_rank() == 0:
            model.eval()

            if not epoch % args.test_interval:
                test_acc = test(model, test_dataloader, epoch, args.exp_name, args.dataset, "test")
                train_acc = test(model, train_dataloader, epoch, args.exp_name, args.dataset, "train")

                wandb.log({"epoch": epoch, "train/total_acc": train_acc, "test/total_acc": test_acc})

        dist.barrier()

        model.train()
        train(model, train_dataloader)

        if not epoch % args.save_interval:
            model.save_checkpoint(f"experiments/{args.exp_name}/checkpoints/", epoch)
        

if __name__ == "__main__":
    args = get_args()
    setup_experiment(args)
    main(args)