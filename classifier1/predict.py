# Import
import os
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from PIL import Image
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score

import argparse
import pathlib

# DATASET
class TestDataset(Dataset):
    def __init__(self, datapath):
        self.data_path = datapath
        self.file_names = os.listdir(self.data_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.file_names[index])
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transforms(image)
        img_name = self.file_names[index]
        return image, img_name

class ResNet_frozen(nn.Module):
    def __init__(self):
        super(ResNet_frozen, self).__init__()
        #res18_modules = list(models.resnet18().children())[:-1]
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,3),
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=pathlib.Path, required=True, help='Path to saved model weights')
    parser.add_argument('--input_dir', type=pathlib.Path, required=True, help='Path to input data')
    parser.add_argument('--output_dir', type=pathlib.Path, required=True, help='Path to output predictions')
    parser.add_argument('--pred_name', type=str, required=True, help='Name of prediction file')

    # read arguments
    args = parser.parse_args()

    # Device
    os.environ['CUDA_VISIBLE_DEVICES']="1"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model weights
    best_model_path = args.model_dir
    model = ResNet_frozen()
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)

    # dataloader
    BATCH_SIZE = 1
    NUM_WORKERS = 1
    SHUFFLE = False
    PIN_MEMORY = True
    DROP_LAST = False

    TEST_DIR = args.input_dir

    test_dataset = TestDataset(datapath = TEST_DIR)
    test_loader = DataLoader(dataset = test_dataset,
                            batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS,
                            shuffle = SHUFFLE,
                            pin_memory = PIN_MEMORY,
                            drop_last = DROP_LAST)

    # make predictions
    model.eval()

    y_preds = []
    img_names = []

    for batch_index, (x, img_id) in enumerate(tqdm(test_loader)):
        x = x.to(DEVICE)
        y_logits = model(x).cpu()
        y_pred = torch.argmax(y_logits, dim=1)
        y_pred = y_pred.tolist()
        img_names.extend(img_id)
        y_preds.extend(y_pred)

    # make predictions into dataframe
    label_decoding = {0:'N',1:'R',2:'W'}
    y_preds_decoded = [label_decoding[x] for x in y_preds]
    predictions = pd.DataFrame(list(zip(img_names, y_preds_decoded)),columns=['fname','state'])

    outpath = args.output_dir
    outname = args.pred_name
    predictions.to_csv(os.path.join(outpath, outname),index=False)