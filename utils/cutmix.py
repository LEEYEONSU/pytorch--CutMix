# re-implementation of CutMix
# CutMix, CutMixCrossEntropyLoss
import torch
import numpy as np 
import torch.nn as nn

from torch.distributions import Beta, Uniform

def cutmix(minibatch, alpha):
    data, targets = minibatch
    indices = torch.randperm(data.size(0))
    data_shuffled = data[indices]
    target_shuffled = targets[indices]

    # Beta distribution sampling
    lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample()
    # lam = np.random.beta(alpha, alpha)
    img_H, img_W = data.size(-2), data.size(-1)
     
    # rx = Uniform(torch.tensor(0), torch.tensor(img_W))
    # ry = Uniform(torch.tensor(0), torch.tensor(img_H))
    rx = Uniform(0, img_W).sample()
    ry = Uniform(0,img_H).sample()
    rw = img_W * torch.sqrt( 1 - lam)
    rh = img_H * torch.sqrt(1 - lam)

    # rx = np.random.uniform(0, img_W)
    # ry = np.random.uniform(0,img_H)
    # rh = img_H * np.sqrt(1 - lam)
    # rw = img_W * np.sqrt(1 - lam)

    # rx, ry, rw, rh = torch.tensor(float(rx)), torch.tensor(float(ry)), torch.tensor(float(rw)), torch.tensor(float(rh))
    x1 = torch.round(torch.clamp(rx - rw // 2, 0, img_W))
    y1 = torch.round(torch.clamp(ry - rh // 2, 0 , img_H))
    x2 = torch.round(torch.clamp(rx + rw // 2, 0, img_W))
    y2 = torch.round(torch.clamp(ry + rh // 2, 0 , img_H))

    # x1 = np.round(np.clip(rx - rw // 2, 0, img_W))
    # y1 = np.round(np.clip(ry - rh // 2, 0 , img_H))
    # x2 = np.round(np.clip(rx + rw // 2, 0, img_W))
    # y2 = np.round(np.clip(ry + rh // 2, 0 , img_H))
     
    lam = 1 - ((x2 - x1)  * (y2 - y1) / (img_H * img_W))
    data[ : , : , int(y1) : int(y2), int(x1) : int(x2)] = data_shuffled[ : , : , int(y1) : int(y2), int(x1) : int(x2) ] 

    return data, targets, target_shuffled, lam