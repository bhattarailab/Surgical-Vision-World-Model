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

def generate(model, dataset, return_keys, num_examples, batch_size):
    model.eval()
    num_examples = min(num_examples, len(dataset))
    dataset = dataset.sample(num_examples, seed=np.random.randint(low=0, high=100))
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_outputs = {key: [] for key in return_keys}

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Visualization'):
            data = data.to(device)
            output = model(data)
            viz_dict = output['viz']

            for key in return_keys:
                all_outputs[key].append(viz_dict[key])

    for key in return_keys:
        all_outputs[key] = torch.cat(all_outputs[key]).to('cpu')

    return all_outputs

def visualize(model, dataset, viz_keys, num_examples, batch_size):
    all_viz_outputs = generate(model, dataset, viz_keys, num_examples, batch_size)

    num_rows = len(viz_keys)
    example_tensor = next(iter(all_viz_outputs.values()))
    B, T, C, H, W = example_tensor.shape  # assumes all outputs have the same shape

    fig, axs = plt.subplots(
        nrows=num_rows * num_examples, 
        ncols=T, 
        figsize=(6 * T, 4 * num_rows * num_examples)
    )

    if num_rows * num_examples == 1:
        axs = np.expand_dims(axs, axis=0)  # ensure 2D indexing

    for i in range(num_examples):
        for row_idx, key in enumerate(viz_keys):
            for j in range(T):
                ax = axs[i * num_rows + row_idx, j]
                ax.axis('off')
                img = all_viz_outputs[key][i, j].detach().numpy()
                if img.shape[0] == 1:  # grayscale
                    img = img[0]
                    ax.imshow(img, cmap='gray')
                else:
                    img = img.transpose(1, 2, 0)
                    ax.imshow(img)
                ax.set_title(key if j == 0 else "")

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

def log_visualizations(fig, wandb_tracker, step, epoch):
    wandb_tracker.log({"visualizations": fig, "epoch": epoch}, step=step)
    plt.close(fig)
