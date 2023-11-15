import torch
import deepspeed

import torch.distributed as dist

from models import get_model

from utils.utils import set_random_seed, setup_experiment
from utils.arguments import get_train_args
from utils.data import get_dataset, get_dataloader, get_examples
from utils.loops import *

def main(args):

    train_data, test_data, num_classes = get_dataset(args)
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()

    if dist.get_rank() == 0:
        if args.wandb:
            import wandb

            run = wandb.init(
                name=args.exp_name,
                project='Cheatsheet',
                notes=args.exp_name,
                config=vars(args)
            )

            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("test/*", step_metric="epoch")

            #example_position, example_class = get_examples(train_data)

            #wandb.log({"example_positions": example_position, "example_classes": example_class})

    test_dataloader = get_dataloader(test_data, batch_size=args.batch_size)

    set_random_seed(args.seed)
    dist.barrier()

    model = get_model(args.model, num_classes, args.cs_size)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, _, train_dataloader, _ = deepspeed.initialize(args=args, training_data=train_data, model=model, model_parameters=parameters)
    
    if args.load_dir and args.ckpt_id:
        model.load_checkpoint(args.load_dir, args.ckpt_id)

    # Actual training loop
    for epoch in range(args.train_epochs):

        if dist.get_rank() == 0:
            model.eval()

            if not epoch % args.test_interval:
                test_results, test_pos_chart, test_cls_chart = test(model, test_dataloader, epoch, args.exp_name, args.dataset, "test")
                train_results, train_pos_chart, train_cls_chart = test(model, train_dataloader, epoch, args.exp_name, args.dataset, "train")

                if args.wandb:
                    train_pos_acc = train_results["train"]["Positional Accuracy"]["Total Positional Accuracy"]
                    test_pos_acc = test_results["test"]["Positional Accuracy"]["Total Positional Accuracy"]
                    train_class_acc = train_results["train"]["Class Accuracy"]["Total Class Accuracy"]
                    test_class_acc = test_results["test"]["Class Accuracy"]["Total Class Accuracy"]

                    #train_acc_per_pos = train_results["train"]["Positional Accuracy"]["Accuracy Per Position"]
                    #test_acc_per_pos = test_results["test"]["Positional Accuracy"]["Accuracy Per Position"]
                    #train_acc_per_class = train_results["train"]["Class Accuracy"]["Accuracy Per Class"]
                    #test_acc_per_class = test_results["test"]["Class Accuracy"]["Accuracy Per Class"]

                    wandb.log({
                        "epoch": epoch, 
                        "train/total_pos_acc": train_pos_acc, "test/total_acc": test_pos_acc,
                        "train/total_class_acc": train_class_acc, "test/total_class_acc": test_class_acc,
                        "train/acc_per_pos": train_pos_chart, "test/acc_per_pos": test_pos_chart,
                        "train/acc_per_class": train_cls_chart, "test/acc_per_class": test_cls_chart
                        })

        dist.barrier()

        model.train()
        train(model, train_dataloader)

        if not epoch % args.save_interval:
            model.save_checkpoint(f"experiments/{args.exp_name}/checkpoints/", epoch)
        

if __name__ == "__main__":
    args = get_train_args()
    setup_experiment(args)
    main(args)