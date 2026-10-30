import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tqdm
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange

from modules.st_transformer import STTransformerDecoder

class LatentActionModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        # MODEL
        model_config = config['model']
        num_heads = model_config['num_heads']
        d_model = model_config['d_model']
        enc_num_layers = model_config['encoder']['num_layers']
        dec_num_layers = model_config['decoder']['num_layers']

        # VAE
        vae_config = model_config['vae']
        self.beta = vae_config['beta']
        self.latent_dim = vae_config['latent_dim']

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

        self.action_enc = nn.Linear(in_features=d_model, out_features=2*self.latent_dim)
        self.action_dec = nn.Linear(in_features=self.latent_dim, out_features=d_model)

        self.encoder = STTransformerDecoder(num_layers=enc_num_layers, num_heads=num_heads, d_model=d_model)
        self.decoder = STTransformerDecoder(num_layers=dec_num_layers, num_heads=num_heads, d_model=d_model)

        self.first_frame_only = False
        if 'first_frame_only' in model_config and model_config['first_frame_only']:
            self.first_frame_only = True

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
        
        if self.first_frame_only:
            first_frames = tokens[:, :1]
            tokens = first_frames.repeat(1, T, 1, 1)
        
        tokens = tokens[:,:-1]
        original = original[:,1:]

        # vae
        action_enc = self.action_enc(action_tokens)
        z_mu_var = action_enc[:,1:]
        z_mu, z_var = torch.chunk(z_mu_var, 2, dim=-1)
        z_sample = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var) # B T-1 latent_dim
        action_dec = self.action_dec(z_sample)

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

        z_mu, z_var = z_mu.reshape(-1, self.latent_dim), z_var.reshape(-1, self.latent_dim)
        kl_loss = -0.5 * torch.sum(1 + z_var - z_mu ** 2 - z_var.exp(), dim=1).mean() # var is actually log-var here

        total_loss += self.beta * kl_loss

        loss_dict['kl'] = kl_loss
        loss_dict['total'] = total_loss

        return {
            "z_mu": z_mu.detach().cpu(),
            "z_var": z_var.detach().cpu(),
            "loss": total_loss,
            "loss_dict": loss_dict,
            "viz": viz,
        }

    @torch.no_grad()
    def generate_single_frame(self, x, action = None):
        # x: B T C H W
        # action: B T-1 latent_dim

        B, T, C, H_image, W_image = x.shape

        if action is None:
            assert T == 1
            action = torch.zeros((B, 0, self.latent_dim), device=x.device)

        original = x

        next_action = torch.randn((B, 1, self.latent_dim))
        action = torch.cat((action, next_action), dim=1)

        x = rearrange(x, "B T C H W -> (B T) C H W")
        x = self.patch_proj_in(x) # (B T) d_model H_patches W_patches
        H, W = x.shape[-2], x.shape[-1]
        x = rearrange(x, "(B T) D H W -> B T (H W) D", B=B)
        tokens = x + self.pos_embed[:, :T] # B T S D

        action_dec = self.action_dec(action)

        x = torch.cat((action_dec[:, :, None, :], tokens), dim=2) # B T (S+1) D
        x = self.decoder(x)[:,:,1:,:] # drop the action conditioning token

        x = rearrange(x, "B T (H W) D -> (B T) D H W", H=H)
        x = self.patch_proj_out(x)
        x = F.sigmoid(rearrange(x, "(B T) C H W -> B T C H W", B=B))

        original = torch.cat((original, x[:,-1:]), dim=2)
        return original, action

    def generate_video(self, prompt, n_frames):
        # prompt B 1 C H W
        action = None
        x = prompt
        for i in tqdm.tqdm(range(n_frames), desc='Generating Frames', total=n_frames):
            x, action = self.generate_single_frame(x, action)
        
        return x

    def sample(self, index):
        raise NotImplementedError
