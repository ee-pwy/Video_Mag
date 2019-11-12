import os
from skimage import io, transform
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import json


class MagDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_name = glob.glob(os.path.join(root_dir, 'frameA', '*.png'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_name = os.path.basename(self.img_name[idx])
        img_a = io.imread(os.path.join(self.root_dir, 'frameA', frame_name))
        img_b = io.imread(os.path.join(self.root_dir, 'frameB', frame_name))
        img_c = io.imread(os.path.join(self.root_dir, 'frameC', frame_name))
        amplified = io.imread(os.path.join(self.root_dir, 'amplified', frame_name))
        f, _ = os.path.splitext(frame_name)
        meta_path = os.path.join(self.root_dir, 'meta', f + '.json')
        amplification_factor = json.load(open(meta_path))['amplification_factor']

        sample = {'frameA': img_a, 'frameB': img_b, 'frameC': img_c, 'amplified': amplified,
                  'amplification_factor': amplification_factor}

        if self.transform:
            sample = self.transform(sample)

        return sample
