import torch
import deepspeed

import torch.distributed as dist

from models import get_model

from utils.utils import set_random_seed, setup_experiment
from utils.arguments import get_args
from utils.data import get_cifar, get_dataloaders
from utils.loops import *

def main(args):

    train_data, valid_data, test_data, num_classes = get_cifar(
         dataset=args.dataset,
         cheatsheet=args.cheatsheet,
         randomize_sheet=args.randomize_sheet,
         cs_size=args.cs_size,
         val_size=500
    )
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(train_data, valid_data, test_data, args.batch_size)

    set_random_seed(args.seed)
    dist.barrier()

    model = get_model(args.model, num_classes, args.cs_size)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters)
    
    if args.load_dir and args.ckpt_id:
        model.load_checkpoint(args.load_dir, args.ckpt_id)

    # Actual training loop
    for epoch in range(args.train_epochs):

        model.train()
        train(model, train_dataloader)
        
        if dist.get_rank() == 0:
            model.eval()
            new_loss = validation(model, valid_dataloader)

            images, labels = next(iter(valid_dataloader))
            images, labels = images.cuda(), labels.cuda()
            
            saliencies = compute_saliency_maps(model, images, labels)
            visualize_and_save_saliency(images, labels, saliencies, epoch, args.exp_name)

            if not epoch % args.test_interval:
                    test(model, test_dataloader, epoch, args.exp_name, args.dataset)
        
        dist.barrier()


        if not epoch % args.save_interval:
            model.save_checkpoint(f"experiments/{args.exp_name}/checkpoints/", epoch)
        

if __name__ == "__main__":
    args = get_args()
    setup_experiment(args)
    main(args)