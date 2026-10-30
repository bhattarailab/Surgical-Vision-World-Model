import yaml
import json

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

def config_from_yaml(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    return config

def config_from_checkpoint(checkpoint_path):
    with open(checkpoint_path+'/config.json') as f:
        config = json.load(f)
    
    return config

def generate(model, dataset, num_examples, batch_size):
    model.eval()
    num_examples = min(num_examples, len(dataset))
    dataset = dataset.sample(num_examples, seed=np.random.randint(low=0,high=100))
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_original = []
    all_reconstructed = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Visualization'):
            data = data.to(device)
            output = model(data)
            original = output['original']
            reconstructed = output['reconstructed'] # B T-1 C H W

            all_original.append(original)
            all_reconstructed.append(reconstructed)

    all_original = torch.cat(all_original)
    all_reconstructed = torch.cat(all_reconstructed)

    return all_original.to('cpu'), all_reconstructed.to('cpu')

def visualize(model, dataset, num_examples, batch_size):
    all_original, all_reconstructed = generate(model, dataset, num_examples, batch_size)

    B, window_size, C, H, W = all_original.shape
    fig, axs = plt.subplots(nrows = 2*num_examples, ncols=window_size, figsize=(6*window_size, 4*2*num_examples))

    for i in range(num_examples):
        for j in range(window_size):
            ax_orig = axs[2*i, j]
            ax_orig.axis('off')
            ax_orig.imshow(all_original[i][j].detach().cpu().numpy().transpose(1, 2, 0))
            ax_orig.set_title('Original')

            ax_recon = axs[2*i+1, j]
            ax_recon.axis('off')
            if j == 0:
                continue
            ax_recon.imshow(all_reconstructed[i][j-1].detach().cpu().numpy().transpose(1, 2, 0))
            ax_recon.set_title('Reconstructed')
    
    return fig, axs

def overlay_masks_tensor(original, mask, alpha=0.5, mask_color=(0, 0, 1.0)):
    """
    Args:
        original: (B, T, 3, H, W) — normalized RGB images in [0, 1]
        mask:     (B, T, 1, H, W) — binary masks (0 or 1)
        alpha:    float — transparency of the mask overlay
        mask_color: RGB tuple in [0, 1], e.g., (1.0, 0, 0) for red

    Returns:
        overlaid: (B, T, 3, H, W) — tensor with mask overlaid
    """
    # Ensure mask is broadcastable to 3 channels
    mask_rgb = torch.tensor(mask_color, device=original.device).view(1, 1, 3, 1, 1)
    mask = mask.expand(-1, -1, 3, -1, -1)

    # Apply overlay: if mask > 0.5, blend with mask_color, else keep original
    overlaid = torch.where(
        mask > 0.5,
        alpha * mask_rgb + (1 - alpha) * original,
        original
    )

    return overlaid

def log_visualizations(fig, wandb_tracker, step):
    wandb_tracker.log({f"visualizations": fig}, step=step)
    plt.close(fig)
