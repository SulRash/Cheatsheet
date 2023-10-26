import torch
import deepspeed
import wandb

import torch.distributed as dist

from models import get_model

from utils.utils import set_random_seed, setup_experiment
from utils.arguments import get_args
from utils.data import get_cifar, get_dataloaders
from utils.loops import *
from utils.log import *

def main(args):

    train_data, test_data, num_classes = get_cifar(args)
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()

    run = initial_logging(args)

    train_dataloader, test_dataloader = get_dataloaders(train_data, test_data, args.batch_size)

    set_random_seed(args.seed)
    dist.barrier()

    model = get_model(args.model, num_classes, args.cs_size)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters)
    
    if args.load_dir and args.ckpt_id:
        model.load_checkpoint(args.load_dir, args.ckpt_id)

    # Actual training loop
    for epoch in range(args.train_epochs):

        if dist.get_rank() == 0:
            model.eval()

            if not epoch % args.test_interval:
                test_acc = test(model, test_dataloader, epoch, args.exp_name, args.dataset, "test")
                train_acc = test(model, train_dataloader, epoch, args.exp_name, args.dataset, "train")

                wandb.log({"train/epoch": epoch, "train/total_acc": train_acc, "test/total_acc": test_acc})

                images, labels = next(iter(test_dataloader))
                images, labels = images.cuda(), labels.cuda()

                saliencies = compute_saliency_maps(model, images, labels)
                visualize_and_save_saliency(images, labels, saliencies, epoch, args.exp_name)

                log_saliency_maps(args, run)
        
        dist.barrier()

        model.train()
        train(model, train_dataloader)

        if not epoch % args.save_interval:
            model.save_checkpoint(f"experiments/{args.exp_name}/checkpoints/", epoch)
        

if __name__ == "__main__":
    args = get_args()
    setup_experiment(args)
    main(args)