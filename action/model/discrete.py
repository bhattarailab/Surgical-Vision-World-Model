import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from huggingface_hub import PyTorchModelHubMixin

import numpy as np
from einops import rearrange

from modules.st_transformer import STTransformerDecoder
from vector_quantize_pytorch import VectorQuantize

class LatentActionModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        # MODEL
        model_config = config['model']
        num_heads = model_config['num_heads']
        d_model = model_config['d_model']
        enc_num_layers = model_config['encoder']['num_layers']
        dec_num_layers = model_config['decoder']['num_layers']

        # VQ-VAE
        vae_config = model_config['vq_vae']
        self.codebook_dim = vae_config['codebook_dim']
        self.codebook_size = vae_config['codebook_size']
        self.decay = vae_config['decay']
        self.commitment_weight = vae_config['commitment_weight']
        self.threshold_ema_dead_code = vae_config['threshold_ema_dead_code']

        # DATA
        data_config = config['data']
        video_shape = data_config['video_shape']
        patch_shape = data_config['patch_shape']
        video_shape = (video_shape['T'], video_shape['H'], video_shape['W'])
        patch_shape = (patch_shape['H'], patch_shape['W'])

        self.d_model = d_model
        
        T, H, W = video_shape
        patch_h, patch_w = patch_shape
        assert(H%patch_h == 0 and W%patch_w == 0)

        self.T = T
        self.video_shape = video_shape
        self.H, self.W = H//patch_h, W//patch_w

        super().__init__()

        self.patch_proj_in = nn.Conv2d(
            in_channels=3, 
            out_channels=d_model, 
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_proj_out = nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=3,
            kernel_size=patch_shape,
            stride=patch_shape
        )
        S = H//patch_h * W//patch_w
        self.pos_embed = nn.Parameter(torch.zeros(1, T, S, d_model))
        self.action_prompt = nn.Parameter(torch.zeros((d_model, )))
        nn.init.uniform_(self.pos_embed, a=-1, b=1)
        nn.init.uniform_(self.action_prompt, a=-1, b=1)

        self.action_enc = nn.Linear(in_features=d_model, out_features=self.codebook_dim) 
        self.action_dec = nn.Linear(in_features=self.codebook_dim, out_features=d_model)
        self.quantize = VectorQuantize(dim=self.codebook_dim, codebook_size=self.codebook_size, decay=self.decay, commitment_weight=self.commitment_weight, threshold_ema_dead_code=self.threshold_ema_dead_code)

        self.encoder = STTransformerDecoder(num_layers=enc_num_layers, num_heads=num_heads, d_model=d_model)
        self.decoder = STTransformerDecoder(num_layers=dec_num_layers, num_heads=num_heads, d_model=d_model)


    def get_tokens(self, x):
        B, T = x.shape[:2]
        x = rearrange(x, "B T C H W -> (B T) C H W")
        x = self.patch_proj_in(x) # (B T) d_model H_patches W_patches
        x = rearrange(x, "(B T) D H W -> B T (H W) D", B=B)
        tokens = x + self.pos_embed[:, :T] # B T S D

        return tokens

    def forward(self, x):
        original = x
        B, T, C, H_image, W_image = x.shape
        
        tokens = self.get_tokens(x)

        # add learnable action prompt token
        x = torch.cat((self.action_prompt.repeat(B, T, 1, 1), tokens), dim=2) # B T (S+1) D
        x = self.encoder(x) # B T (S+1) D
        action_tokens = x[:,:,0,:] # B T D

        tokens = tokens[:,:-1]
        original = original[:,1:]

        # vq-vae
        action_enc = self.action_enc(action_tokens) # (12,8,512) -> (12,8,32)  
        action_enc = action_enc[:,1:]
        z_mu, _, commit_loss = self.quantize(action_enc) # B T d_codebook #(12,8,32)
        action_dec = self.action_dec(z_mu)

        # use frames from 0 to n-1 to predict frames 1 to n
        x = torch.cat((action_dec[:, :, None, :], tokens), dim=2) # B T-1 (S+1) D
        x = self.decoder(x)[:,:,1:,:] # drop the action conditioning token

        x = rearrange(x, "B T (H W) D -> (B T) D H W", H=self.H)
        x = self.patch_proj_out(x)
        x = F.sigmoid(rearrange(x, "(B T) C H W -> B T C H W", B=B))

        viz = {
            'gt_image': original.cpu(),
            'image' : x.detach().cpu()
        }

        loss_dict = {}
        total_loss = 0

        loss_term = F.mse_loss(original, x)
        total_loss += loss_term

        loss_dict['reconstruction'] = loss_term

        total_loss += commit_loss

        loss_dict['commit'] = commit_loss
        loss_dict['total'] = total_loss
        return {
            "z_mu": z_mu.detach(),
            "loss": total_loss,
            "loss_dict": loss_dict,
            "viz": viz,
        }

