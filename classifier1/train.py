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

# Data directory
DATA_DIR = '/workspace/Competition/SLEEP/EEG/data/train/'
label_dir = os.path.join(DATA_DIR, 'train1_labels.csv')
train_dir = os.path.join(DATA_DIR, 'trace1', 'eeg1')

# Device
os.environ['CUDA_VISIBLE_DEVICES']="0"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Dataset
class TraceDataset(Dataset):
    def __init__(self, datapath, labeldf):
        #self.df = pd.read_csv(labelpath)
        self.df = labeldf
        self.label_encoding = {'N':0, 'R':1, 'W':2}
        self.data_path = datapath
        self.file_names = self.df['fname']
        self.labels = self.df['state']
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])          

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self,index):
        image_path = os.path.join(self.data_path, self.file_names[index]+'.png')
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transforms(image)
        lbl = self.labels[index]
        lbl = self.label_encoding[lbl]
        return image, lbl

# Define model
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

# hyperparams
BATCH_SIZE = 16
NUM_WORKERS = 1
SHUFFLE = True
PIN_MEMORY = True
DROP_LAST = False
EPOCHS = 50
LEARNING_RATE = 5e-5

# Read data
train = pd.read_csv(label_dir)
traindf,valdf = train_test_split(train, test_size=0.2)
traindf = traindf.reset_index(drop=True)
valdf = valdf.reset_index(drop=True)

# Initialize dataset and dataloader
train_dataset = TraceDataset(datapath = train_dir, labeldf = traindf)
val_dataset = TraceDataset(datapath = train_dir, labeldf = valdf)

train_loader = DataLoader(dataset = train_dataset,
                            batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS,
                            shuffle = SHUFFLE,
                            pin_memory = PIN_MEMORY,
                            drop_last = DROP_LAST)

val_loader = DataLoader(dataset = val_dataset,
                            batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS,
                            shuffle = SHUFFLE,
                            pin_memory = PIN_MEMORY,
                            drop_last = DROP_LAST)

# Initialize model
model = ResNet_frozen()
model.to(DEVICE)

# criterion and optmizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# TRAIN
RECORDER_DIR = '/workspace/Competition/SLEEP/EEG/classifier1/results'
best_loss = np.Inf

for epoch in range(EPOCHS):
    model.train()

    train_total_loss = 0
    target_list = []
    pred_list = []

    for batch_index, (x,y) in tqdm(enumerate(train_loader)):
        x,y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        pred_list.extend(y_pred.argmax(dim=1).cpu().tolist())
        target_list.extend(y.cpu().tolist())
    train_mean_loss = train_total_loss / (batch_index+1)
    train_accuracy = accuracy_score(target_list, pred_list)
    train_f1score = f1_score(target_list, pred_list, average='macro')

    model.eval()
    val_total_loss = 0
    target_list = []
    pred_list = []
    with torch.no_grad():
        for batch_index, (x,y) in tqdm(enumerate(val_loader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            #
            val_total_loss += loss.item()
            target_list.extend(y.cpu().tolist())
            pred_list.extend(y_pred.argmax(dim=1).cpu().tolist())
    val_mean_loss = val_total_loss / (batch_index+1)
    val_accuracy = accuracy_score(target_list, pred_list)
    val_f1score = f1_score(target_list, pred_list, average='macro')

    msg1 = f"Epoch {epoch}/{EPOCHS} - Train loss: {train_mean_loss}; Train Accuracy: {train_accuracy}; Train F1: {train_f1score}"
    msg2 = f"Valid loss: {val_mean_loss}; Val Accuracy: {val_accuracy}; Val F1: {val_f1score}"
    print(msg1)
    print(msg2)

    if val_mean_loss < best_loss:                               
        best_loss = val_mean_loss
        check_point = {                                         
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(check_point, os.path.join(RECORDER_DIR,'best.pt')) 

