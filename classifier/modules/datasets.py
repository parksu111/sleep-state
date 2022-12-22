from torch.utils.data import Dataset
from modules.utils import load_json, encode
import numpy as np
from PIL import Image
import os
import pandas as pd
from transforms import get_transforms
import torchvision.transforms as transforms

class TraceDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.df = pd.read_csv(label_dir, dtype={'imname':str,'state':str})
        self.img_dir = img_dir
        self.transforms = get_transforms()
        self.imnames = list(self.df['imname'])
        self.labels = list(self.df['state'])

    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, index):
        impath = os.path.join(self.img_dir, self.imnames[index])
        img = Image.open(impath)
        img = self.transforms(img)
        lbl = self.labels[index]
        lbl = encode(lbl)

        return img,lbl
        
    