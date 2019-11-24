import os
from skimage import io
import glob
from torch.utils.data import Dataset
import cv2
import torch
import json


class MagSubset(Dataset):
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
        if len(sample) == 5:
            img_a, img_b, img_c, amplified, amplification_factor = sample['frameA'],\
                        sample['frameB'],  sample['frameC'], sample['amplified'], sample['amplification_factor']
            img_a = img_a
            img_b = img_b
            img_c = img_c
            amplified = amplified
            img_a = torch.from_numpy(img_a.transpose((2 ,0, 1)))
            img_b = torch.from_numpy(img_b.transpose((2, 0, 1)))
            img_c = torch.from_numpy(img_c.transpose((2, 0, 1)))
            amplified = torch.from_numpy(amplified.transpose((2, 0, 1)))
            amplification_factor = torch.tensor(amplification_factor)

            return {'frameA': img_a, 'frameB': img_b, 'frameC': img_c, 'amplified': amplified,
                      'amplification_factor': amplification_factor}
        else:
            img_a, img_b, amplification_factor = sample['frameA'], \
                        sample['frameB'], sample['amplification_factor']
            img_a = torch.from_numpy(img_a.transpose((2, 0, 1)))
            img_b = torch.from_numpy(img_b.transpose((2, 0, 1)))
            img_a = img_a
            img_b = img_b
            amplification_factor = torch.tensor(amplification_factor)
            return {'frameA': img_a, 'frameB': img_b, 'amplification_factor': amplification_factor}


class MagDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        self.img_name = glob.glob(os.path.join(root_dir, 'frameA', '*.png'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        while True:
            frame_name = os.path.basename(self.img_name[idx])
            img_a = cv2.imread(os.path.join(self.root_dir, 'frameA', frame_name))
            img_b = cv2.imread(os.path.join(self.root_dir, 'frameB', frame_name))
            img_c = cv2.imread(os.path.join(self.root_dir, 'frameC', frame_name))
            amplified = cv2.imread(os.path.join(self.root_dir, 'amplified', frame_name))
            f, _ = os.path.splitext(frame_name)
            meta_path = os.path.join(self.root_dir, 'meta', f + '.json')
            amplification_factor = json.load(open(meta_path))['amplification_factor']
            if all(i is not None for i in [img_a, img_b, img_c, amplified, amplification_factor]):
                break
            else:
                idx += 1

        sample = {'frameA': img_a, 'frameB': img_b, 'frameC': img_c, 'amplified': amplified,
                  'amplification_factor': amplification_factor}

        if self.transform:
            sample = self.transform(sample)

        return sample
