import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge('torch')

import numpy as np
import pandas as pd

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, videos_path_or_list, window_size, stride, transform, fixed_start=False, from_list=False, frac=1):
        if from_list:
            self.videos = videos_path_or_list  # already a list of video paths
        else:
            with open(videos_path_or_list) as f:
                self.videos = f.read().splitlines()

        if frac!=1:
            self.videos = np.random.choice(self.videos, size=int(len(self.videos) * frac), replace=False)

        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.fixed_start = fixed_start

    def __len__(self):
        return len(self.videos)

    def sample(self, num_samples, seed=42):
        # Returns a new dataset with a sampled subset of videos with starting points fixed
        rng = np.random.default_rng(seed)
        num_samples = min(num_samples, len(self.videos))
        indices = rng.choice(len(self.videos), size=num_samples, replace=False)
        sampled_videos = [self.videos[i] for i in indices]

        return VideoDataset(
            sampled_videos,
            window_size=self.window_size,
            stride=self.stride,
            fixed_start=True,
            transform=self.transform,
            from_list=True,
        )

    def __getitem__(self, index):
        for i in range(10):
            try:
                vid_name = self.videos[index]
                vr = VideoReader(vid_name, num_threads=-1, ctx=cpu(0))
                num_frames = len(vr)

                high = num_frames-(self.window_size-1)*self.stride
                start = np.random.randint(low=0, high=high)

                if self.fixed_start:
                    start = 0

                video_data = vr.get_batch(range(start, start+self.window_size*self.stride, self.stride)).permute(0, 3, 1, 2)

                return self.transform(video_data)
                
            except Exception as e:
                # raise e
                print(i, 'CANNOT READ', vid_name, e)
                index = np.random.randint(low=0, high=len(self.videos))

class PitVisDataset(torch.utils.data.Dataset):
    def __init__(self, df, window_size, stride, transform=None, from_df=False, random_offset=True):
        if from_df:
            self.df = df  # already a DataFrame
        else:
            self.df = pd.read_csv(df)

        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.random_offset = random_offset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        vid_name = row['file_name']
        vr = VideoReader(vid_name, num_threads=-1, ctx=cpu(0))

        start = row['int_time'] * 24
        if self.random_offset:
            start += np.random.randint(0, 24)

        video_data = vr.get_batch(
            range(start, start + self.window_size * self.stride, self.stride)
        ).permute(0, 3, 1, 2)

        if self.transform:
            video_data = self.transform(video_data)

        return video_data

    def sample(self, num_samples, seed=42):
        """Returns a new dataset with a sampled subset of rows."""
        rng = np.random.default_rng(seed)
        num_samples = min(num_samples, len(self.df))
        indices = rng.choice(len(self.df), size=num_samples, replace=False)

        sampled_df = self.df.iloc[indices].reset_index(drop=True)

        return PitVisDataset(
            sampled_df,
            window_size=self.window_size,
            stride=self.stride,
            transform=self.transform,
            from_df=True
        )

def get_dataloaders(config, num_workers):
    data_config = config['data']
    train_config = config['train']
    video_shape = data_config['video_shape']
    
    base_transform = [
        v2.CenterCrop((600, 900)),
        v2.Resize(size=(video_shape['H'], video_shape['W']), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
    val_transform = v2.Compose(base_transform)
    
    train_transform = base_transform.copy()
    if not 'overfit' in train_config:
        train_transform = [v2.RandomHorizontalFlip(p=0.5)] + train_transform
    train_transform = v2.Compose(train_transform)

    frac = data_config['frac'] if 'frac' in data_config else 1
    train_dataset = VideoDataset('../data/train.txt', window_size=video_shape['T'], stride=data_config['stride'], frac=frac, transform=train_transform)
    val_dataset = VideoDataset('../data/val.txt', window_size=video_shape['T'], stride=data_config['stride'], frac=frac, transform=val_transform, fixed_start=True)

    if 'overfit' in train_config:
        train_dataset = train_dataset.sample(train_config['overfit'])
        val_dataset = train_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size_per_gpu'], num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size_per_gpu'], num_workers=num_workers)

    return train_dataloader, val_dataloader

def get_pitvis_dataloaders(config, num_workers):
    data_config = config['data']
    train_config = config['train']
    video_shape = data_config['video_shape']
    
    base_transform = [
        v2.CenterCrop((720, 720)),
        v2.Resize(size=(video_shape['H'], video_shape['W']), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
    val_transform = v2.Compose(base_transform)
    
    train_transform = base_transform.copy()
    if not 'overfit' in train_config:
        train_transform = [v2.RandomHorizontalFlip(p=0.5)] + train_transform
    train_transform = v2.Compose(train_transform)

    train_dataset = PitVisDataset('../data/train.csv', window_size=video_shape['T'], stride=data_config['stride'], transform=train_transform)
    val_dataset = PitVisDataset('../data/val.csv', window_size=video_shape['T'], stride=data_config['stride'], transform=val_transform)

    if 'overfit' in train_config:
        train_dataset = train_dataset.sample(train_config['overfit'])
        val_dataset = train_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size_per_gpu'], num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size_per_gpu'], num_workers=num_workers)

    return train_dataloader, val_dataloader
