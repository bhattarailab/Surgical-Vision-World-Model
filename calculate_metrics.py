import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

from natsort import natsorted
import tqdm

import imageio.v3 as iio
from SurgVLP import surgvlp
from mmengine.config import Config

import piq
from cmmd_pytorch.distance import mmd
from cdfvd import fvd
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

BATCH_SIZE = 32

def load_video_grids(path, n_rows, n_cols, frame_size=(256, 256), padding=2):
    """
    path       : directory with images (0.png, 1.png, ...)
    n_rows     : number of videos per grid (rows)
    n_cols     : number of frames per video (cols)
    frame_size : (H, W) of original frames (before grid)
    padding    : padding used in make_grid (default=2)
    """
    to_tensor = T.ToTensor()  # [C,H,W], float32 in [0,1]
    all_videos = []
    files = natsorted([f for f in os.listdir(path) if f.endswith(".png")])

    h, w = frame_size
    stride_h, stride_w = h + padding, w + padding  # step size between frames

    for i, fname in tqdm.tqdm(enumerate(files), total=len(files)):
        img = Image.open(os.path.join(path, fname)).convert("RGB")
        grid = to_tensor(img)  # [C, H_grid, W_grid]

        videos_by_row = []
        for r in range(n_rows):
            frames = []
            for c in range(n_cols):
                top, left = r * stride_h, c * stride_w
                frame = grid[:, top:top+h, left:left+w]  # crop frame
                frames.append(frame)
            video = torch.stack(frames, dim=0)  # [T,C,H,W]
            videos_by_row.append(video)

        batch = torch.stack(videos_by_row, dim=0)  # [n_rows, T, C, H, W]
        all_videos.append(batch)

    return torch.cat(all_videos, dim=0)  # [B, T, C, H, W]

def load_videos(video_dir, fps=5):
    # Get sorted list of video files
    video_files = sorted(
        [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    )

    all_videos = []
    for vf in tqdm.tqdm(video_files, total=len(video_files)):
        # Read full video as (T, H, W, C)
        frames = iio.imread(vf, plugin="pyav")  # supports mp4 reading
        frames = frames.astype(np.float32) / 255.0  # normalize to [0,1]
        # Rearrange to (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        all_videos.append(frames)

    # Stack into (B, T, C, H, W)
    videos = np.stack(all_videos, axis=0)
    return torch.from_numpy(videos)

lpips_model = piq.LPIPS().eval().cuda()     # Perceptual metric
surgvlp_configs = Config.fromfile('SurgVLP/tests/config_peskavlp.py')['config']
device = "cuda" if torch.cuda.is_available() else "cpu"

fid_model, fid_preprocess = surgvlp.load(surgvlp_configs.model_config, device=device)

## Content debiased FVD
evaluator_fvd = fvd.cdfvd('videomae', ckpt_path="checkpoints/vit_g_hybrid_pt_1200e_ssv2_ft.pth", device = device, compute_feats= True)

rc_tensor = load_video_grids(
    "/scratch/dthapa/outputs/test/reconstruction",
    n_rows=6, # replace by 2*batch_size used in sampling
    n_cols=15,
    frame_size=(256, 256),
    padding=2
)
tr_tensor = load_video_grids(
    "/scratch/dthapa/outputs/test/transfer",
    n_rows=6, # replace by 2*batch_size used in sampling
    n_cols=15,
    frame_size=(256, 256),
    padding=2
)
print(rc_tensor.shape)  # [B, T, C, H, W]
print(tr_tensor.shape)  # [B, T, C, H, W]

gts = rc_tensor[0::2]
target_recs = rc_tensor[1::2]
gts_transfers = tr_tensor[0::2]
transfers = tr_tensor[1::2]

def calculate_fvd_cmmd(reals, fakes, batch_size, device="cuda"):
    """
    reals, fakes: tensors of shape (B, T, C, H, W)
    batch_size: number of videos to process at once
    """

    # reals, fakes = reals[:args.slice_end], fakes[:args.slice_end]

    B, T, C, H, W = reals.shape
    all_psnr, all_ssim, all_lpips = [], [], []
    fid_real = []
    fid_fake = []

        # ---- loop over video batches ----
    for start in tqdm.tqdm(range(0, B, batch_size), total=len(range(0, B, batch_size))):
        end = min(start + batch_size, B)

        x = reals[start:end].to(device).clamp(0, 1)
        y = fakes[start:end].to(device).clamp(0, 1)

        x_frames = x.view(-1, C, H, W)
        y_frames = y.view(-1, C, H, W)

        # # PSNR, SSIM, LPIPS (batched → scalar)
        all_psnr.append(piq.psnr(x_frames, y_frames).item())
        all_ssim.append(piq.ssim(x_frames, y_frames).item())
        all_lpips.append(lpips_model(x_frames, y_frames).item())
        
        # FID updates
        x_frames = fid_preprocess(x_frames).to(device)
        y_frames = fid_preprocess(y_frames).to(device)
        with torch.no_grad():
            output_dict = fid_model(x_frames, mode='video')
            image_embeddings = output_dict['img_emb']
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            fid_real.append(image_embeddings.detach().cpu().numpy())
            output_dict = fid_model(y_frames, mode='video')
            image_embeddings = output_dict['img_emb']
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            fid_fake.append(image_embeddings.detach().cpu().numpy())

    # Aggregate
    psnr = sum(all_psnr) / len(all_psnr)
    ssim = sum(all_ssim) / len(all_ssim)
    lpips = sum(all_lpips) / len(all_lpips)
    fid = mmd(np.vstack(fid_real), np.vstack(fid_fake))


    # ---- FVD batching (video-level) ----
    x_videos = (reals.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).cpu().numpy()
    y_videos = (fakes.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).cpu().numpy()
    import time
    start = time.time()
    real_stats = fvd.get_videomae_features(evaluator_fvd.real_stats, evaluator_fvd.model, x_videos, batchsize = BATCH_SIZE, device=evaluator_fvd.device, model_dtype=evaluator_fvd.model_dtype)
    fake_stats = fvd.get_videomae_features(evaluator_fvd.fake_stats, evaluator_fvd.model, y_videos, batchsize = BATCH_SIZE, device=evaluator_fvd.device, model_dtype=evaluator_fvd.model_dtype)
    end = time.time()
    print('fvd time: ', end-start)
    fvd_score_ssv2 = evaluator_fvd.compute_fvd_from_stats(fake_stats, real_stats)

    x_videos = real_stats.get_all()
    y_videos = fake_stats.get_all()
    ECS = F.cosine_similarity(torch.from_numpy(x_videos), torch.from_numpy(y_videos), dim=1).mean().numpy()

    # ---- Print / log ----
    print(f"PSNR: {psnr:.3f}")
    print(f"SSIM: {ssim:.3f}")
    print(f"LPIPS: {lpips:.3f}")
    print(f"CMMD: {fid:.3f}")
    print(f"FVD_ssv2: {fvd_score_ssv2:.3f}")
    print(f"ECS: {ECS:.3f}")

    return dict(PSNR=psnr, SSIM=ssim, LPIPS=lpips, FID=fid, FVD_ssv2=fvd_score_ssv2, ECS=ECS)

print('Reconsruction')
calculate_fvd_cmmd(gts[:,1:], target_recs[:,1:], 8, device=device)
print('Transfer')
calculate_fvd_cmmd(gts_transfers[:,1:],  transfers[:,1:], 8, device=device)
