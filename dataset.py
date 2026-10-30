import os
from contextlib import redirect_stderr

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge('torch')

import numpy as np
import pandas as pd

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, videos_path_or_list, window_size, stride, transform, lam_transform, noise_mu=-3.0, noise_std=0.5, fixed_start=False, from_list=False):
        if from_list:
            self.videos = videos_path_or_list  # already a list of video paths
        else:
            with open(videos_path_or_list) as f:
                self.videos = f.read().splitlines()

        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.lam_transform = lam_transform
        self.noise_mu = noise_mu
        self.noise_std = noise_std
        self.fixed_start = fixed_start

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        for i in range(10):
            try:
                vid_name = self.videos[index]
                vr = VideoReader(vid_name, num_threads=-1, ctx=cpu(0))
                num_frames = len(vr)

                high = num_frames-(self.window_size-1)*self.stride
                if self.fixed_start:
                    start = 0
                else:
                    start = high-1

                video_data = vr.get_batch(range(start, start+self.window_size*self.stride, self.stride)).permute(0, 3, 1, 2)
                lam_video = self.lam_transform(video_data)
                video_data = self.transform(video_data)
                diffusion_video = video_data*2-1
                
                first_frame = diffusion_video[0]
                pred_frames = diffusion_video[1:]
                T = self.window_size-1
                
                log_sigma = torch.randn(()) * self.noise_std + self.noise_mu
                sigma = log_sigma.exp()
                noised_first_frame = first_frame + torch.randn_like(first_frame) * sigma

                data_dict = {
                    'video': pred_frames,
                    'cond_frames_without_noise': first_frame.repeat((T, 1, 1, 1)),
                    'cond_frames': noised_first_frame.repeat((T, 1, 1, 1)),
                    'lam_inputs': lam_video,
                    'flow_inputs': video_data*256, # (0, 1) -> (0, 256)
                    'cond_aug': sigma*torch.ones((T, )),
                    'num_video_frames': T,
                    'image_only_indicator': torch.zeros(T),
                }
                
                return data_dict
            except Exception as e:
                print(i, 'CANNOT READ', vid_name, e)
                index = np.random.randint(low=0, high=len(self.videos))

class PitVisDataset(torch.utils.data.Dataset):
    def __init__(self, df, window_size, stride, transform, lam_transform, random_offset=True, noise_mu=-3.0, noise_std=0.5, aug_prob=0.0):
        self.df = pd.read_csv(df)

        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.lam_transform = lam_transform
        self.noise_mu = noise_mu
        self.noise_std = noise_std
        self.random_offset = random_offset
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        for i in range(10):
            try:
                # Redirect stderr to null to disable Decord/FFmpeg output
                with redirect_stderr(open(os.devnull, 'w')):
                    row = self.df.iloc[index]
                    vid_name = row['file_name']
                    # print('SOURCE VIDEO NAME:', vid_name)
                    step =  row['int_step']
                    vr = VideoReader(vid_name, num_threads=-1, ctx=cpu(0))

                    start = row['int_time'] * 24
                    if self.random_offset:
                        start += np.random.randint(0, 24)

                    end_frame = start + self.window_size * self.stride
                    full_chunk = vr.get_batch(range(start, end_frame))
                    video_data = full_chunk[::self.stride][:self.window_size].permute(0, 3, 1, 2)

                    # video_data = vr.get_batch(
                    #     range(start, start + self.window_size * self.stride, self.stride)
                    # ).permute(0, 3, 1, 2)

                    lam_video = self.lam_transform(video_data)
                    video_data = self.transform(video_data)*2-1
                    
                    first_frame = video_data[0]
                    pred_frames = video_data[1:]
                    T = self.window_size-1
                    
                    log_sigma = torch.randn(()) * self.noise_std + self.noise_mu
                    sigma = log_sigma.exp()
                    noised_first_frame = first_frame + torch.randn_like(first_frame) * sigma

                    augment = False
                    if np.random.random() < self.aug_prob:
                        augment = True
                        target_row = self.df[self.df['int_step'] == step].sample(n=1).iloc[0]
                        vid_name = target_row['file_name']
                        # print('TARGET VIDEO NAME:', vid_name)
                        vr = VideoReader(vid_name, num_threads=-1, ctx=cpu(0))

                        start = target_row['int_time'] * 24
                        if self.random_offset:
                            start += np.random.randint(0, 24)

                        video_data = vr.get_batch(
                            range(start, start + self.window_size * self.stride, self.stride)
                        ).permute(0, 3, 1, 2)

                        # print('READ SUCCESFUL', video_data.shape)
                        # load lam inputs from source
                        # lam_video = self.lam_transform(video_data) 
                        video_data = self.transform(video_data)*2-1
                        
                        first_frame = video_data[0]
                        pred_frames = video_data[1:]
                        T = self.window_size-1
                        
                        log_sigma = torch.randn(()) * self.noise_std + self.noise_mu
                        sigma = log_sigma.exp()
                        noised_first_frame = first_frame + torch.randn_like(first_frame) * sigma

                    data_dict = {
                        'video': video_data, # only to be used as inputs to the classifier
                        'cond_frames_without_noise': first_frame.repeat((T, 1, 1, 1)), # first frame of the target video
                        'cond_frames': noised_first_frame.repeat((T, 1, 1, 1)),
                        'lam_inputs': lam_video, # frames from the source video
                        'cond_aug': sigma*torch.ones((T, )),
                        'num_video_frames': T,
                        'image_only_indicator': torch.zeros(T),
                        'augment': augment,
                        'step': torch.tensor(step-1, dtype=torch.long)
                    }
                    
                    return data_dict
            except Exception as e:
                print(i, 'CANNOT READ', vid_name, e)
                index = np.random.randint(low=0, high=len(self.df))

class VideoSampler(LightningDataModule):
    def __init__(self, batch_size, num_workers, window_size=8, stride=18, pitvis=False, aug_prob=0.5):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size

        svd_transform = [
            v2.CenterCrop((720, 720)),
            v2.Resize(size=(256, 256), antialias=True),
            v2.ConvertImageDtype(torch.float32),
        ]
        svd_transform = v2.Compose(svd_transform)

        if pitvis:
            lam_transform = [
                v2.CenterCrop((720, 720)),
                v2.Resize(size=(224, 224), antialias=True),
                v2.ConvertImageDtype(torch.float32),
            ]
            lam_transform = v2.Compose(lam_transform)

            self.train_dataset = PitVisDataset('/iopsstor/scratch/cscs/koju/data/pitvis/train.csv', window_size=window_size, stride=stride, transform=svd_transform, lam_transform=lam_transform, noise_mu=-10, noise_std=0, aug_prob=aug_prob)
            self.val_dataset = PitVisDataset('/iopsstor/scratch/cscs/koju/data/pitvis/val.csv', window_size=window_size, stride=stride, transform=svd_transform, lam_transform=lam_transform, noise_mu=-10, noise_std=0, random_offset=False, aug_prob=0)
        else:
            lam_transform = [
                v2.CenterCrop((600, 900)),
                v2.Resize(size=(160, 240), antialias=True),
                v2.ConvertImageDtype(torch.float32),
            ]
            lam_transform = v2.Compose(lam_transform)

            self.train_dataset = VideoDataset('data/train.txt', window_size=window_size, stride=stride, transform=svd_transform, lam_transform=lam_transform)
            self.val_dataset = VideoDataset('data/val.txt', window_size=window_size, stride=stride, transform=svd_transform, lam_transform=lam_transform, noise_mu=-10, noise_std=0, fixed_start=True)
    
    def collate_fn(self, data):
        return {
            'video': torch.cat([d['video'] for d in data]),
            'cond_frames_without_noise': torch.cat([d['cond_frames_without_noise'] for d in data]),
            'cond_frames': torch.cat([d['cond_frames'] for d in data]),
            'lam_inputs': torch.stack([d['lam_inputs'] for d in data]),
            'flow_inputs': torch.stack([d['flow_inputs'] for d in data]),
            'cond_aug': torch.cat([d['cond_aug'] for d in data]),
            'num_video_frames': data[0]['num_video_frames'],
            'image_only_indicator': torch.stack([d['image_only_indicator'] for d in data])
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )
