from torch.nn import functional as F
import torch
imoprt torch.nn as nn

def get_loss(loss_name: str):
    if loss_name == 'crossentropy':
        return F.cross_entropy