"""For config loading, checkpointing, and visualization."""

import os
import json
import yaml
import numpy as np
import torch
from einops import rearrange
from accelerate.utils import extract_model_from_parallel as unwrap_model

def config_from_yaml(path):
    """Load YAML config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(accelerator, epoch, optimization_step, val_loss, save_dir):
    """Save Accelerate checkpoint with metadata."""
    checkpoint_dir = os.path.join(save_dir, "latest")
    accelerator.save_state(checkpoint_dir)

    # Save training metadata alongside
    if accelerator.is_main_process:
        meta = {
            "epoch": epoch,
            "optimization_step": optimization_step,
            "val_loss": float(val_loss) if val_loss is not None else None,
        }
        meta_path = os.path.join(checkpoint_dir, "train_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Save a copy of metadata with epoch number for tracking
        best_meta_path = os.path.join(save_dir, f"meta_epoch_{epoch}.json")
        with open(best_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    accelerator.wait_for_everyone()


def load_checkpoint_meta(checkpoint_dir):
    """Load training metadata from a checkpoint."""
    meta_path = os.path.join(checkpoint_dir, "train_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {"epoch": 0, "optimization_step": 0}


@torch.no_grad()
def generate_frames(model, prompt, n_frames, device):
    """
    Generate frames autoregressively using random latent actions.

    Args:
        model: LatentActionModel (possibly wrapped by DDP/compile)
        prompt: (B, 1, C, H, W) tensor — the starting frame
        n_frames: number of additional frames to generate
        device: torch device

    Returns:
        (B, 1 + n_frames, C, H, W) tensor of generated video
    """
    # Unwrap DDP / compiled model to access attributes
    m = unwrap_model(model)
    m.eval()
    B = prompt.shape[0]
    frames = [prompt[:, 0].to(device)]  # list of (B, C, H, W)
    actions = []  # list of (B, latent_dim)

    for _ in range(n_frames):
        # Stack all frames so far
        x = torch.stack(frames, dim=1)  # (B, T_cur, C, H, W)
        T_cur = x.shape[1]

        # Patchify
        x_flat = rearrange(x, "B T C H W -> (B T) C H W")
        x_flat = m.patch_proj_in(x_flat)
        H_p, W_p = x_flat.shape[-2], x_flat.shape[-1]
        x_flat = rearrange(x_flat, "(B T) D H W -> B T (H W) D", B=B)
        tokens = x_flat + m.pos_embed[:, :T_cur]

        # Random latent actions — one per frame
        z = torch.randn(B, T_cur, m.latent_dim, device=device)
        action_dec = m.action_dec(z)

        # Decode
        x_dec = torch.cat((action_dec[:, :, None, :], tokens), dim=2)
        x_dec = m.decoder(x_dec)[:, :, 1:, :]

        x_dec = rearrange(x_dec, "B T (H W) D -> (B T) D H W", H=H_p)
        x_dec = m.patch_proj_out(x_dec)
        x_dec = torch.sigmoid(rearrange(x_dec, "(B T) C H W -> B T C H W", B=B))

        # Append the last predicted frame
        frames.append(x_dec[:, -1])

    return torch.stack(frames, dim=1).cpu()  # (B, 1+n_frames, C, H, W)


@torch.no_grad()
def encode_actions(model, video, device):
    """
    Extract latent actions (z_mu) from a video clip by running the encoder.

    Args:
        model: LatentActionModel (possibly wrapped)
        video: (B, T, C, H, W) tensor — source video clip
        device: torch device

    Returns:
        z_mu: (B, T-1, latent_dim) — deterministic latent actions
    """
    m = unwrap_model(model)
    m.eval()

    video = video.to(device)
    B, T = video.shape[:2]

    # Patchify + positional embedding
    tokens = m.get_tokens(video)  # B T S D

    # Encoder: prepend action prompt, run transformer
    x = torch.cat((m.action_prompt.repeat(B, T, 1, 1), tokens), dim=2)  # B T (S+1) D
    x = m.encoder(x)  # B T (S+1) D
    action_tokens = x[:, :, 0, :]  # B T D

    # VAE encoder: project to mu/logvar, take mu (deterministic)
    action_enc = m.action_enc(action_tokens)  # B T 2*latent_dim
    z_mu_var = action_enc[:, 1:]  # B T-1 2*latent_dim (skip frame 0)
    z_mu, _ = torch.chunk(z_mu_var, 2, dim=-1)  # B T-1 latent_dim

    return z_mu


@torch.no_grad()
def generate_with_actions(model, initial_frame, actions, device):
    """
    Generate frames autoregressively using *provided* latent actions.
    Used for action transfer: apply actions from one video to a different
    starting frame.

    Args:
        model: LatentActionModel (possibly wrapped)
        initial_frame: (B, 1, C, H, W) — the starting frame
        actions: (B, N, latent_dim) — latent actions to apply
        device: torch device

    Returns:
        (B, 1 + N, C, H, W) — generated video (initial frame + N generated)
    """
    m = unwrap_model(model)
    m.eval()

    B = initial_frame.shape[0]
    n_actions = actions.shape[1]
    frames = [initial_frame[:, 0].to(device)]  # list of (B, C, H, W)

    for t in range(n_actions):
        # Stack all frames accumulated so far
        x = torch.stack(frames, dim=1)  # (B, t+1, C, H, W)
        T_cur = x.shape[1]

        # Patchify
        x_flat = rearrange(x, "B T C H W -> (B T) C H W")
        x_flat = m.patch_proj_in(x_flat)
        H_p = x_flat.shape[-2]
        x_flat = rearrange(x_flat, "(B T) D H W -> B T (H W) D", B=B)
        tokens = x_flat + m.pos_embed[:, :T_cur]

        # Use the provided latent actions (one per decoder timestep)
        z = actions[:, :T_cur].to(device)  # (B, T_cur, latent_dim)
        action_dec = m.action_dec(z)  # (B, T_cur, d_model)

        # Decode
        x_dec = torch.cat((action_dec[:, :, None, :], tokens), dim=2)
        x_dec = m.decoder(x_dec)[:, :, 1:, :]

        x_dec = rearrange(x_dec, "B T (H W) D -> (B T) D H W", H=H_p)
        x_dec = m.patch_proj_out(x_dec)
        x_dec = torch.sigmoid(rearrange(x_dec, "(B T) C H W -> B T C H W", B=B))

        # Take the last predicted frame
        frames.append(x_dec[:, -1])

    return torch.stack(frames, dim=1).cpu()  # (B, 1 + N, C, H, W)


def create_comparison_video(gt_frames, pred_frames):
    """
    Create a side-by-side comparison numpy array for WandB video logging.

    Args:
        gt_frames: (T, C, H, W) float tensor in [0,1]
        pred_frames: (T, C, H, W) float tensor in [0,1]

    Returns:
        (T, C, H, 2*W) uint8 numpy array (side-by-side GT | Pred)
    """
    gt = (gt_frames * 255).clamp(0, 255).byte()
    pred = (pred_frames * 255).clamp(0, 255).byte()
    combined = torch.cat([gt, pred], dim=-1)  # (T, C, H, 2W)
    return combined.numpy()


def create_generation_video(frames):
    """
    Convert generated frames to numpy for WandB video logging.

    Args:
        frames: (T, C, H, W) float tensor in [0,1]

    Returns:
        (T, C, H, W) uint8 numpy array
    """
    return (frames * 255).clamp(0, 255).byte().numpy()


def log_visualizations(model, val_dataset, accelerator, optimization_step, epoch, device):
    """
    Log reconstruction comparison and generation GIFs to WandB.

    Args:
        model: the LAM model
        val_dataset: validation dataset
        accelerator: HF Accelerator instance
        optimization_step: current global step
        epoch: current epoch
        device: torch device
    """
    import wandb

    if not accelerator.is_main_process:
        return

    # Unwrap model for inference
    m = unwrap_model(model)
    m.eval()

    num_examples = 4

    # 1. Reconstruction comparaision
    indices = np.random.choice(len(val_dataset), num_examples, replace=False)
    batch = torch.stack([val_dataset[i] for i in indices]).to(device)  # (N, T, 3, 64, 64)

    output = m(batch)
    pred = output["viz"].get("image")  # (N, T-1, 3, 64, 64)
    gt = output["viz"].get("gt_image")  # (N, T-1, 3, 64, 64)

    wandb_tracker = accelerator.get_tracker("wandb").tracker

    if pred is not None and gt is not None:
        recon_log = {}
        for i in range(min(num_examples, pred.shape[0])):
            vid = create_comparison_video(gt[i], pred[i])  # (T-1, C, H, 2W)
            recon_log[f"reconstruction/example_{i}"] = wandb.Video(vid, fps=4, format="gif")
        wandb_tracker.log(recon_log, step=optimization_step)

    # 2. Generation from prompt (random latent action used)
    gen_log = {}
    for i in range(min(2, num_examples)):
        prompt = batch[i : i + 1, :1]  # (1, 1, C, H, W)
        generated = generate_frames(model, prompt, n_frames=8, device=device)
        vid = create_generation_video(generated[0])  # (17, C, H, W)
        gen_log[f"generation/example_{i}"] = wandb.Video(vid, fps=4, format="gif")
    wandb_tracker.log(gen_log, step=optimization_step)

    # 3. Latent action transfer
    transfer_log = {}
    num_transfer = min(2, num_examples - 1) 
    for i in range(num_transfer):
        source_clip = batch[i : i + 1]  # (1, T, C, H, W)
        # Pick a different clip as the target initial frame
        target_idx = (i + 1) % num_examples
        target_initial = batch[target_idx : target_idx + 1, :1]  # (1, 1, C, H, W)

        # Encode latent actions from source
        z_mu = encode_actions(model, source_clip, device)  # (1, T-1, latent_dim)

        # Generate using source actions + target initial frame
        transferred = generate_with_actions(
            model, target_initial, z_mu, device
        )  # (1, T, C, H, W)

        # Side-by-side: source video | transferred video
        source_vid = (source_clip[0].cpu() * 255).clamp(0, 255).byte()
        transfer_vid = (transferred[0].cpu() * 255).clamp(0, 255).byte()
        combined = torch.cat([source_vid, transfer_vid], dim=-1)  # (T, C, H, 2W)
        transfer_log[f"action_transfer/example_{i}"] = wandb.Video(
            combined.numpy(), fps=4, format="gif"
        )
    wandb_tracker.log(transfer_log, step=optimization_step)

    