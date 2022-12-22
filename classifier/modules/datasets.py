from torch.utils.data import Dataset
from modules.utils import load_json, encode
import numpy as np
from PIL import Image
import os
import pandas as pd
import torchvision.transforms as transforms

class TraceDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.df = pd.read_csv(label_dir, )