import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from safetensors.torch import load_file as load_safetensors

import os
from PIL import Image
import pickle
import tqdm
import pprint
import cv2
import imageio.v2 as imageio
import json

import torch
import torchvision.transforms as T
from pathlib import Path
from functools import partial
from torch import nn
from huggingface_hub import snapshot_download

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Script for video processing with configs and checkpoints")

    # Paths
    parser.add_argument(
        "--config_path",
        type=str,
        default="/scratch/lpanta/logs/dynamics_full/2025-08-26T17-47-31_-dynamics/configs/2025-08-26T17-47-31-project.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/scratch/lpanta/logs/dynamics_full/2025-08-26T17-47-31_-dynamics/checkpoints/best_val/epoch=000040.ckpt/pytorch_model.bin",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="Folder to save results"
    )
    parser.add_argument(
        "--use_reconstructed",
        action="store_true",
        default=False,
        help="Whether to use reconstructed images"
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default="lam_inputs"
    )

    parser.add_argument(
        "--viz_transfer_videos",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--first_to_last",
        action="store_true",
        default=False,
    )

    # Metrics (computed in-memory, no need to read grids back from disk)
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        default=False,
        help="Compute PSNR/SSIM/LPIPS/FVD/ECS in-memory after generation",
    )
    parser.add_argument(
        "--fvd_ckpt",
        type=str,
        default="checkpoints/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
        help="Checkpoint for the content-debiased FVD (videomae) evaluator",
    )
    parser.add_argument(
        "--metrics_batch_size",
        type=int,
        default=8,
        help="Per-batch size for PSNR/SSIM/LPIPS feature extraction",
    )
    parser.add_argument(
        "--fvd_batch_size",
        type=int,
        default=32,
        help="Batch size for videomae FVD feature extraction",
    )

    # Processing parameters
    parser.add_argument("--num_videos", type=int, default=1000, help="Number of videos")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--T", type=int, default=7, help="Temporal length")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of steps")
    parser.add_argument("--slice_each", type=int, default=-1)
    parser.add_argument("--slice_end", type=int, default=-1)
    parser.add_argument("--slice_action", type=int, default=-1)

    # Sampling parameters
    parser.add_argument("--n_rounds", type=int, default=1, help="Number of rounds")
    parser.add_argument("--T_sample", type=int, default=-1, help="Number of rounds")
    parser.add_argument("--min_scale", type=float, default=1)
    parser.add_argument("--max_scale", type=float, default=2.5)

    args = parser.parse_args()

    # Derived parameters
    if args.slice_each == -1:
        args.slice_each = args.T

    if args.T_sample == -1:
        args.T_sample = args.slice_each * args.n_rounds
    else:
        print('Warning! Overriding n_rounds argument as T_sample is provided')
        args.n_rounds = (args.T_sample+args.slice_each-1)//args.slice_each
    args.window_size = (args.n_rounds-1)*args.slice_each+args.T + 1

    return args

args = get_args()
config_path      = args.config_path
ckpt             = args.ckpt
save_folder      = args.save_folder
use_reconstructed = args.use_reconstructed
num_videos  = args.num_videos
batch_size  = args.batch_size
T           = args.T
num_steps   = args.num_steps
n_rounds    = args.n_rounds
T_sample    = args.T_sample
window_size = args.window_size

# ---- Print all args (for logs) ----
print("\n===== Arguments =====")
pprint.pprint(vars(args), sort_dicts=False)
print("=====================\n")

def init_model(config, ckpt, load_ckpt: bool = True):
    config = OmegaConf.load(config)
    config['model']['params']['en_and_decode_n_samples_a_time'] = T
    config['model']['params']['sampler_config']['params']['num_steps'] = num_steps

    config['model']['params']['sampler_config']['params']['guider_config']['params']['min_scale'] = args.min_scale
    config['model']['params']['sampler_config']['params']['guider_config']['params']['max_scale'] = args.max_scale

    model = load_model_from_config(config, ckpt if load_ckpt else None)
    return model

def load_model_from_config(config, ckpt: str = None):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        elif ckpt.endswith("bin"):  # For deepspeed merged checkpoints
            sd = torch.load(ckpt, map_location="cpu")
            for k in list(sd.keys()):  # Remove the prefix
                if "_forward_module" in k:
                    sd[k.replace("_forward_module.", "")] = sd[k]
                    del sd[k]
        elif ckpt.endswith("safetensors"):
            sd = load_safetensors(ckpt)
        else:
            raise NotImplementedError
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    model = model.cuda()
    model.eval()
    return model

def combine_and_save(video1: torch.Tensor, video2: torch.Tensor, out_path: str, fps: int = 25, scale=0.8,
                     label1="source", label2="transferred", prompt_label="target prompt"):
    """
    Combine two videos (T, 3, H, W) side by side with captions.
    Stage 1: video1 + static first frame of video2 ("<prompt_label>")
    Stage 2: video1 + full video2 ("<label2>")

    Args:
        video1: torch.Tensor in [0, 1], shape (T, 3, H, W)
        video2: torch.Tensor in [0, 1], shape (T, 3, H, W)
        out_path: str, output .mp4 path
        fps: int, frames per second
        label1: caption for video1 (left)
        label2: caption for video2 (right, stage 2)
        prompt_label: caption for the static conditioning frame (right, stage 1)
    """
    assert video1.shape == video2.shape, "Videos must have same shape"
    T, C, H, W = video1.shape

    # normalize [-1,1] -> [0,255]
    video1 = ((video1+1)/2 * 255).clamp(0, 255).byte()
    video2 = ((video2+1)/2 * 255).clamp(0, 255).byte()

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264")

    # ---------- Stage 1: video1 + static first frame of video2 ----------
    static_frame2 = video2[0].permute(1, 2, 0).cpu().numpy().copy()
    static_frame2 = cv2.cvtColor(static_frame2, cv2.COLOR_RGB2BGR)
    static_frame2 = cv2.putText(static_frame2, prompt_label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    static_frame2 = cv2.cvtColor(static_frame2, cv2.COLOR_BGR2RGB)

    for t in range(T):
        f1 = video1[t].permute(1, 2, 0).cpu().numpy().copy()

        f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2BGR)
        f1 = cv2.putText(f1, label1, (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)

        combined = np.concatenate([f1, static_frame2], axis=1)
        writer.append_data(combined)

    # ---------- Stage 2: video1 + video2 normally ----------
    for t in range(T):
        f1 = video1[t].permute(1, 2, 0).cpu().numpy().copy()
        f2 = video2[t].permute(1, 2, 0).cpu().numpy().copy()

        f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2BGR)
        f1 = cv2.putText(f1, label1, (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)

        f2 = cv2.cvtColor(f2, cv2.COLOR_RGB2BGR)
        f2 = cv2.putText(f2, label2, (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)

        combined = np.concatenate([f1, f2], axis=1)
        writer.append_data(combined)

    writer.close()
    print(f"Saved combined video to {out_path}")


model = init_model(config_path, ckpt)
if args.action_key == 'lam_inputs':
    model.conditioner.embedders[2].lam.slice = args.slice_action

config = OmegaConf.load(config_path)

# config['model']['params']['en_and_decode_n_samples_a_time'] = T
# model = instantiate_from_config(config.model).cuda().eval()

config['data']['params']['batch_size'] = batch_size
config['data']['params']['window_size'] = window_size
source_sampler = instantiate_from_config(config.data)
target_sampler = instantiate_from_config(config.data)

source_dataloader = source_sampler.val_dataloader()
target_dataloader = target_sampler.val_dataloader()

if 'pitvis' in config['data']['params']:
    df1 = source_dataloader.dataset.df.sample(
        n=num_videos, replace=False, random_state=1
    ).sort_values(by='int_step', kind='stable')
    df2 = df1.sample(
        frac=1, replace=False, random_state=2
    ).sort_values(by='int_step', kind='stable')
    source_dataloader.dataset.df = df1 
    target_dataloader.dataset.df = df2 
else:
    if args.first_to_last:
        videos = np.random.choice(source_dataloader.dataset.videos, num_videos) # same videos for both
        source_dataloader.dataset.videos = videos
        target_dataloader.dataset.videos = videos

    else:
        source_dataloader.dataset.videos = np.random.choice(source_dataloader.dataset.videos, num_videos)
        target_dataloader.dataset.videos = np.random.choice(target_dataloader.dataset.videos, num_videos)

source_dataloader.dataset.fixed_start = True
target_dataloader.dataset.fixed_start = False
source_dataloader.dataset.shuffle = False 
target_dataloader.dataset.shuffle = False 

def save_samples(samples, s):
    samples = torch.clamp(samples.detach().float().cpu(), -1.0, 1.0)

    grid = torchvision.utils.make_grid(rearrange(samples, 'B T C H W -> (B T) C H W'), nrow=T_sample+1)

    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)

    filename = f'{save_folder}/{s}/{i}.png'
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    img = Image.fromarray(grid)
    img.save(filename)

def slice_and_reshape(tensor, start, end, bt=False):
    if bt:
        tensor = rearrange(tensor, '(B T) ... -> B T ... ', T=window_size-1)
    return rearrange(tensor[:,start:end], 'B T ... -> (B T) ...')

def index_repeat_reshape(tensor, index):
    return tensor[:,index].repeat(1, T, 1, 1, 1)

def video_to_lam_inputs(video, size, target_length=None):
    """
    Convert a generated video (SVD space) into the format expected by the LAM
    action encoder (`lam_inputs`), so that actions can be re-inferred from it.

    Generated videos are (B, T, 3, 256, 256) in [-1, 1]; the LAM expects
    (B, T, 3, H_lam, W_lam) in [0, 1], where ``size=(H_lam, W_lam)`` is the LAM
    input resolution. This is dataset-dependent (e.g. 160x240 for surgtoolloc,
    224x224 for pitvis), so it is passed in -- derived from the dataset's own
    ``lam_inputs`` tensor -- rather than hardcoded. If ``target_length`` is given,
    the temporal axis is padded (by repeating the last frame) or truncated to
    match, so the result can be sliced exactly like the dataset's ``lam_inputs``.

    Note: generated frames come from the (square) SVD crop, which may differ from
    the LAM's crop region/aspect, so this re-inference operates on a slightly
    different field of view -- an inherent approximation of evaluating on output.
    """
    B, Tn, C, H, W = video.shape
    x = ((video + 1.0) / 2.0).clamp(0.0, 1.0)
    x = rearrange(x, 'b t c h w -> (b t) c h w')
    x = F.interpolate(x, size=tuple(size), mode='bilinear', align_corners=False, antialias=True)
    x = rearrange(x, '(b t) c h w -> b t c h w', b=B)

    if target_length is not None:
        if x.shape[1] >= target_length:
            x = x[:, :target_length]
        else:
            pad = x[:, -1:].repeat(1, target_length - x.shape[1], 1, 1, 1)
            x = torch.cat([x, pad], dim=1)
    return x

def continued_generation(prompt_frames, target):
    last_frames = prompt_frames
    predicted = []
    for j in range(n_rounds):
        pred_first = j*args.slice_each+1
        pred_end = j*args.slice_each+1+T

        t = {}
        # print(T, pred_end-pred_first, full_video.shape)
        t['video'] = slice_and_reshape(target['video'], 0, T, bt=True) # not used
        t['cond_aug'] = slice_and_reshape(target['cond_aug'], pred_first-1, pred_end-1, bt=True)
        t[args.action_key] = target[args.action_key][:,pred_first-1:pred_end]
        t['num_video_frames']=T
        t['image_only_indicator']=target['image_only_indicator'][:,:T]

        nff_repeat = repeat(last_frames, 'b 1 c h w -> b t c h w', t=T)
        t['cond_frames_without_noise'] = rearrange(nff_repeat, 'b t c h w -> (b t) c h w')
        t['cond_frames'] = t['cond_frames_without_noise']
        if j > 0:
            t['cond_aug'] = 0.02*torch.ones_like(t['cond_aug'])

        t = {k: v.to('cuda') if type(v) == torch.Tensor else v for k, v in t.items()}

        # Target reconstructions
        preds = model.log_images(t, N=batch_size, ucg_keys=['cond_frames_without_noise', 'cond_frames', args.action_key])['samples'].detach().cpu() # B T C H W
        preds = preds[:, :args.slice_each]
        predicted.append(preds)
        last_frames = preds[:, -1].unsqueeze(1)

    full_preds = torch.cat(predicted, dim=1)[:,:args.T_sample]

    return full_preds

device = "cuda" if torch.cuda.is_available() else "cpu"

def to_01(x):
    """(-1, 1) -> (0, 1) on CPU float, matching the range of the saved PNG grids."""
    return (x.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0

def load_metric_models():
    import piq
    from cdfvd import fvd

    lpips_model = piq.LPIPS().eval().to(device)

    ## Content debiased FVD
    evaluator_fvd = fvd.cdfvd('videomae', ckpt_path=args.fvd_ckpt, device=device, compute_feats=True)

    return dict(piq=piq, fvd=fvd, lpips_model=lpips_model, evaluator_fvd=evaluator_fvd)

def calculate_fvd_cmmd(reals, fakes, batch_size, models, device="cuda"):
    """
    reals, fakes: tensors of shape (B, T, C, H, W) in [0, 1].
    """
    piq = models['piq']
    fvd = models['fvd']
    lpips_model = models['lpips_model']
    evaluator_fvd = models['evaluator_fvd']

    B, Tn, C, H, W = reals.shape
    all_psnr, all_ssim, all_lpips = [], [], []
    fid_real = []
    fid_fake = []

    # ---- loop over video batches ----
    for start in tqdm.tqdm(range(0, B, batch_size), total=len(range(0, B, batch_size))):
        end = min(start + batch_size, B)

        x = reals[start:end].to(device).clamp(0, 1)
        y = fakes[start:end].to(device).clamp(0, 1)

        x_frames = x.reshape(-1, C, H, W)
        y_frames = y.reshape(-1, C, H, W)

        # PSNR, SSIM, LPIPS (batched → scalar)
        all_psnr.append(piq.psnr(x_frames, y_frames).item())
        all_ssim.append(piq.ssim(x_frames, y_frames).item())
        all_lpips.append(lpips_model(x_frames, y_frames).item())

    # Aggregate
    psnr = sum(all_psnr) / len(all_psnr)
    ssim = sum(all_ssim) / len(all_ssim)
    lpips = sum(all_lpips) / len(all_lpips)

    # ---- FVD batching (video-level) ----
    x_videos = (reals.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).cpu().numpy()
    y_videos = (fakes.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).cpu().numpy()
    real_stats = fvd.get_videomae_features(evaluator_fvd.real_stats, evaluator_fvd.model, x_videos, batchsize=args.fvd_batch_size, device=evaluator_fvd.device, model_dtype=evaluator_fvd.model_dtype)
    fake_stats = fvd.get_videomae_features(evaluator_fvd.fake_stats, evaluator_fvd.model, y_videos, batchsize=args.fvd_batch_size, device=evaluator_fvd.device, model_dtype=evaluator_fvd.model_dtype)
    fvd_score_ssv2 = evaluator_fvd.compute_fvd_from_stats(fake_stats, real_stats)

    x_feats = real_stats.get_all()
    y_feats = fake_stats.get_all()
    ECS = F.cosine_similarity(torch.from_numpy(x_feats), torch.from_numpy(y_feats), dim=1).mean().numpy()

    print(f"PSNR: {psnr:.3f}")
    print(f"SSIM: {ssim:.3f}")
    print(f"LPIPS: {lpips:.3f}")
    print(f"FVD_ssv2: {fvd_score_ssv2:.3f}")
    print(f"ECS: {ECS:.3f}")

    return dict(PSNR=psnr, SSIM=ssim, LPIPS=lpips, FVD_ssv2=fvd_score_ssv2, ECS=ECS)

if __name__ == '__main__':
    # In-memory accumulators for metrics (kept in [0, 1], matching the PNG grids).
    # gt = ground-truth video, pred = model output; one (gt, pred) pair per stage.
    recon_gt, recon_pred = [], []
    transfer_gt, transfer_pred = [], []
    cycle_gt, cycle_pred = [], []

    for i, (source, target) in tqdm.tqdm(enumerate(zip(source_dataloader, target_dataloader)), total=len(source_dataloader), desc='Sampling'):
        # Reconstruction
        first_frames = rearrange(target['cond_frames_without_noise'], '(B T) ... -> B T ...', T=window_size-1)[:, :1] # B 1 C H W
        frames = rearrange(target['video'], '(B T) ... -> B T ...', T=window_size-1)[:,:T_sample]
        full_video = torch.cat([first_frames, frames], dim=1)

        full_preds = continued_generation(first_frames, target)
        recs_with_prompt = torch.cat([first_frames, full_preds], dim=1)

        B, _, C, H, W = full_video.shape
        combined = torch.empty((B * 2, T_sample + 1, C, H, W))
        combined[0::2] = full_video
        combined[1::2] = recs_with_prompt
        save_samples(combined, 'reconstruction')

        if args.compute_metrics:
            recon_gt.append(to_01(full_video))
            recon_pred.append(to_01(recs_with_prompt))

        # Action Transfer
        source_first_frames = rearrange(source['cond_frames_without_noise'], '(B T) ... -> B T ...', T=window_size-1)[:, :1] # B 1 C H W
        source_frames = rearrange(source['video'], '(B T) ... -> B T ...', T=window_size-1)[:,:T_sample]
        source_full_video = torch.cat([source_first_frames, source_frames], dim=1)
        target[args.action_key] = source[args.action_key] # take gt actions from source

        full_preds = continued_generation(first_frames, target)
        dreams_with_prompt = torch.cat([first_frames, full_preds], dim=1)

        B, _, C, H, W = full_video.shape
        combined = torch.empty((B * 2, T_sample + 1, C, H, W))
        combined[0::2] = source_full_video
        combined[1::2] = dreams_with_prompt
        save_samples(combined, 'transfer')

        if args.compute_metrics:
            transfer_gt.append(to_01(source_full_video))
            transfer_pred.append(to_01(dreams_with_prompt))

        if args.viz_transfer_videos:
            for j in range(len(source_full_video)):
                save_path = os.path.join(save_folder, f'{i}_{j}.mp4')
                combine_and_save(source_full_video[j], dreams_with_prompt[j], save_path, fps=5, scale=.6)

        # Cycle consistency:
        # Re-infer actions from the transferred (target-context) video, transfer
        # them back onto the source context, and compare against the original
        # source. 
        lam_size = source[args.action_key].shape[-2:]
        cycle_action = video_to_lam_inputs(dreams_with_prompt, size=lam_size, target_length=source[args.action_key].shape[1])
        source_cycle_ctx = dict(source)
        source_cycle_ctx[args.action_key] = cycle_action

        full_preds = continued_generation(source_first_frames, source_cycle_ctx)
        cycle_with_prompt = torch.cat([source_first_frames, full_preds], dim=1)

        B, _, C, H, W = source_full_video.shape
        combined = torch.empty((B * 3, T_sample + 1, C, H, W))
        combined[0::3] = source_full_video
        combined[1::3] = dreams_with_prompt
        combined[2::3] = cycle_with_prompt
        save_samples(combined, 'cycle')

        if args.compute_metrics:
            cycle_gt.append(to_01(source_full_video))
            cycle_pred.append(to_01(cycle_with_prompt))

        if args.viz_transfer_videos:
            for j in range(len(source_full_video)):
                save_path = os.path.join(save_folder, f'cycle_{i}_{j}.mp4')
                combine_and_save(source_full_video[j], cycle_with_prompt[j], save_path, fps=5, scale=.6,
                                 label1="source", label2="cycle recon", prompt_label="source prompt")

    if args.compute_metrics:
        model = model.cpu()
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        models = load_metric_models()

        recon_gt = torch.cat(recon_gt)
        recon_pred = torch.cat(recon_pred)
        transfer_gt = torch.cat(transfer_gt)
        transfer_pred = torch.cat(transfer_pred)
        cycle_gt = torch.cat(cycle_gt)
        cycle_pred = torch.cat(cycle_pred)
        print(recon_gt.shape, transfer_gt.shape, cycle_gt.shape)

        # Drop the conditioning first frame ([:, 1:]) before scoring
        print('Reconstruction')
        calculate_fvd_cmmd(recon_gt[:, 1:], recon_pred[:, 1:], args.metrics_batch_size, models, device=device)
        print('Transfer')
        calculate_fvd_cmmd(transfer_gt[:, 1:], transfer_pred[:, 1:], args.metrics_batch_size, models, device=device)
        print('Cycle consistency')
        calculate_fvd_cmmd(cycle_gt[:, 1:], cycle_pred[:, 1:], args.metrics_batch_size, models, device=device)

