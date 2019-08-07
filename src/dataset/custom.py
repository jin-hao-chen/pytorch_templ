# -*- coding: utf-8 -*-


import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T



class Custom(Dataset):


    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(root) 
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # If number of channel is 3
        ])
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.transforms(Image.open(os.path.join(self.root, self.imgs[idx])))
        labels = None # labels are determined by your situation
        return img, labels

