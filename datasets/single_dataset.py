import os
from PIL import Image
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.pytorch import ToTensorV2


class SingleDataset(Dataset):
    def __init__(self, opt):
        super(SingleDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def name():
        return 'SingleDataset'
