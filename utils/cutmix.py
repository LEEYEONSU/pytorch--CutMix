# re-implementation of CutMix
# CutMix, CutMixCrossEntropyLoss
import torch
import numpy as np 
import torch.nn as nn

def cutmix(minibatch, alpha):
    data, targets = minibatch
    indices = torch.randperm(data.size(0))
    data_shuffled = data[indices]
    target_shuffled = targets[indices]

    # Beta distribution sampling
    # lam = torch.from_numpy(np.random.beta(alpha, alpha))
    lam = np.random.beta(alpha, alpha)
    img_H, img_W = data.shape[2:]
     
    # rx = torch.from_numpy(np.random.uniform(0, W))
    # ry = torch.from_numpy(np.random.uniform(0, H))
    rx = np.random.uniform(0, img_W)
    ry = np.random.uniform(0,img_H)
    rh = img_H * np.sqrt(1 - lam)
    rw = img_W * np.sqrt(1 - lam)
    # rh = img_H * torch.sqrt(1 - lam)
    # rw = img_W * torch.sqrt( 1 - lam)
     
    x1 = np.round(np.clip(rx - rw // 2, 0, img_W))
    y1 = np.round(np.clip(ry - rh // 2, 0 , img_H))
    x2 = np.round(np.clip(rx + rw // 2, 0, img_W))
    y2 = np.round(np.clip(ry + rh // 2, 0 , img_H))
     
    lam = 1 - ((x2 - x1)  * (y2 - y1) / (img_H * img_W))
    data[ : , : , int(y1) : int(y2), int(x1) : int(x2)] = data_shuffled[ : , : , int(y1) : int(y2), int(x1) : int(x2) ] 

    return data, targets, target_shuffled, lam