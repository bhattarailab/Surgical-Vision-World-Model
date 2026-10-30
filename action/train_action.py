import argparse
from tqdm import tqdm

import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from model.lam import LatentActionModel
# from model.discrete import LatentActionModel
from data import get_dataloaders, get_pitvis_dataloaders
from utils.utils import config_from_yaml, config_from_checkpoint, visualize, log_visualizations
from utils.runs import RunManager

def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Action Model")

    parser.add_argument("--tag", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--config", type=str)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()

    return args

def train():
    args = parse_args()

    config = config_from_yaml(args.config)
    train_config = config['train']

    model = LatentActionModel(config)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with='wandb', gradient_accumulation_steps=train_config['gradient_accumulation_steps'], kwargs_handlers=[ddp_kwargs])
    run_manager = RunManager(config, accelerator, model, args.tag)

    if 'dataset' in config['data'] and config['data']['dataset'] == 'pitvis':
        train_dataloader, val_dataloader = get_pitvis_dataloaders(config, args.num_workers)
    else:
        train_dataloader, val_dataloader = get_dataloaders(config, args.num_workers)
    val_dataset = val_dataloader.dataset
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

    model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer
    )

    overfit = 'overfit' in train_config
    tag = args.tag
    version = run_manager.version
    config["model_parameters"] = sum(p.numel() for p in model.parameters())

    accelerator.init_trackers(
        project_name='benchmark',
        config=config,
        init_kwargs={
            "wandb": {
                "name": f"{tag}/{version}",
                "group": "action"
            }
        }
    )

    if accelerator.is_main_process:
        total_batch_size = train_config['batch_size_per_gpu']*accelerator.num_processes*accelerator.gradient_accumulation_steps
        print('Starting...')
        print('Total batch size:', total_batch_size)
        print('Num iterations per epoch:', len(train_dataloader))
        print('Num gradient updates per epoch:', len(train_dataloader)//config['train']['gradient_accumulation_steps'])

    starting_epoch = 0
    optimization_step = 0
    if args.resume or 'resume_from_checkpoint' in config['train']:
        checkpoint = args.resume if args.resume else config['train']['resume_from_checkpoint']
        accelerator.load_state(checkpoint)

        old_config = config_from_checkpoint(checkpoint)
        starting_epoch = old_config['epoch']
        optimization_step = old_config['optimization_step']
        print('Resuming from', starting_epoch)

    for epoch in range(starting_epoch, train_config['epochs']):
        model.train()

        train_loss_info = {}  # dynamic dict to accumulate losses

        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch+1} Train'):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                output = model(batch)
                loss = output['loss']
                loss_dict = output['loss_dict']  # dict of all components, including reconstruction, kl, etc.

                accelerator.backward(loss)
                batch_size = batch.shape[0]

                train_loss_info['count'] = train_loss_info.get('count', 0) + batch_size
                for k, v in loss_dict.items():
                    train_loss_info[k] = train_loss_info.get(k, 0.0) + v.item() * batch_size

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config['train']['max_grad_norm'])

                    reduced_loss_info = {
                        k: accelerator.reduce(torch.tensor(v, device=accelerator.device))
                        for k, v in train_loss_info.items()
                    }
                    log_dict = {
                        f'train_{k}_loss': reduced_loss_info[k] / reduced_loss_info['count']
                        for k in reduced_loss_info if k != 'count'
                    }
                    log_dict.update({
                        'epoch': epoch + batch_idx / len(train_dataloader),
                        'grad_norm': grad_norm,
                    })
                    accelerator.log(log_dict, step=optimization_step)

                    train_loss_info.clear()
                    optimization_step += 1

                optimizer.step()

        if not (epoch+1) % args.log_every == 0:
            continue

        # Validation
        model.eval()
        val_loss_info = {}

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Epoch {epoch+1} Val'):
                output = model(batch)
                loss_dict = output['loss_dict']

                batch_size = batch.shape[0]

                val_loss_info['count'] = val_loss_info.get('count', 0) + batch_size
                for k, v in loss_dict.items():
                    val_loss_info[k] = val_loss_info.get(k, 0.0) + v.item() * batch_size

        reduced_val_loss_info = {
            k: accelerator.reduce(torch.tensor(v, device=accelerator.device))
            for k, v in val_loss_info.items()
        }
        val_log_dict = {
            f'val_{k}_loss': reduced_val_loss_info[k] / reduced_val_loss_info['count']
            for k in reduced_val_loss_info if k != 'count'
        }
        val_log_dict.update({
            'learning_rate': train_config['learning_rate'],
            'epoch': epoch + 1
        })
        accelerator.log(val_log_dict, step=optimization_step)

        # Save checkpoint
        run_manager.save_checkpoint(
            epoch=epoch+1,
            optimization_step=optimization_step,
            val_loss=val_log_dict['val_total_loss'].item()
        )

        # Visualize Results
        if accelerator.is_main_process:
            wandb_tracker = accelerator.get_tracker('wandb')
            fig, axs = visualize(model, val_dataset, viz_keys=train_config['viz_keys'], num_examples=4, batch_size=4)
            log_visualizations(fig, wandb_tracker, optimization_step, epoch+1)
        
        accelerator.wait_for_everyone()
            

if __name__ == "__main__":
    train()
