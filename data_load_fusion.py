import os
from skimage import io
import glob
from torch.utils.data import Dataset
import cv2
import torch
import json
import numpy as np
import pandas as pd
frame_sample = 57
interval = 8


class FusionSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.subset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.subset)


class ToTensor(object):
    def __call__(self, sample):
        tensor_list = []
        for batch in sample:
            radar = torch.from_numpy(batch[0])
            video = torch.from_numpy(batch[1])
            tensor_list.append([radar, video])
        return tensor_list


class FusionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        stft_path = glob.glob(os.path.join(root_dir, '*', 'STFT', '*.txt'))
        self.batch = [] #[(batch)[[stftmap],[frame]]]
        self.frame_num = [] #[[frame_n corresponding to stft]]
        batch_dir = {}
        self.batch_num = 0
        for i, stft in enumerate(stft_path):
            batch_name = os.path.splitext(os.path.basename(stft))
            batch_name = batch_name[:-5]+batch_name[-5:].split('_')[0]
            if not batch_name in batch_dir.keys():
                batch_dir[batch_name] = self.batch_num
                self.batch_num += 1
                self.batch.append([[],[]])
                name = batch_name + '*'
                dir = os.path.join(stft.split('/')[:-2], 'Frame', name)
                self.batch[batch_dir[batch_name]][1] = glob.glob(dir).sort(key=\
                            lambda x: int(os.path.basename(x).splitext()[0].split('_')[-1]))
            self.batch[batch_dir[batch_name]][0].append(stft)
        for i in range(self.batch_num):
            self.batch[i][0].sort(key=lambda x: int(os.path.basename(x).splitext()[0].split('_')[-1]))
            self.frame_num.append([])
            for j in range(len(self.batch[i][0])):
                self.frame_num[i].append(np.arange(interval*j, interval*j+frame_sample))
        self.transform = transform

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        frames = []
        sample = []
        for i, frame in self.batch[idx][1]:
            frames.append(cv2.imread(frame))
        for i, stftpath in self.batch[idx][0]:
            stft_map = pd.read_csv(stftpath).values
            sample.append([stft_map, np.asarray(frames[self.frame_num[idx][i]])])
        return sample
