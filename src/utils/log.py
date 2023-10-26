import torch.distributed as dist
import wandb

def initial_logging(args):
    run = None
    if dist.get_rank() == 0:
        run = wandb.init(
            project='Cheatsheet',
            notes=args.exp_name,
            config=vars(args)
        )
        deepspeed_artifact = wandb.Artifact(name=f"deepspeed-{args.exp_name}", type="config")
        deepspeed_artifact.add_dir(local_path="src/conf/")
        run.log_artifact(deepspeed_artifact)

        hparams_artifact = wandb.Artifact(name=f"hparams-{args.exp_name}", type="config")
        hparams_artifact.add_dir(local_path=f"experiments/{args.exp_name}/hparams.json", type="config")
        run.log_artifact(deepspeed_artifact)
    dist.barrier()
    return None

def log_saliency_maps(args, run, epoch):
    saliency_artifact = wandb.Artifact(name=f"saliencies-{args.exp_name}", type="results")
    saliency_artifact.add_dir(local_path=f"experiments/{args.exp_name}/saliency_maps/saliency/epoch{epoch}")
    run.log_artifact(saliency_artifact)

    originals_artifact = wandb.Artifact(name=f"saliencies-{args.exp_name}", type="results")
    originals_artifact.add_dir(local_path=f"experiments/{args.exp_name}/saliency_maps/originals/")
    run.log_artifact(originals_artifact)
