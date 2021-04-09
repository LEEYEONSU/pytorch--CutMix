import os 
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms

from utils.train import *
from utils.function import *
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

#. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
parser = argparse.ArgumentParser(description = 'Pytorch Resnet-32 model for CIFAR-10 Classification')

parser.add_argument('--print_freq', default=32, type=int, help = 'Print frequency')
parser.add_argument('--save_dir', default='./save_model/', type=str, help = 'saving model path')
parser.add_argument('--save_every' , default=10, type=int, help='Saves checkpoints at every num of epochs')
parser.add_argument('--evaluate', default = 0, type = int,help='evaluate model on validation set')

parser.add_argument('--lr', default=0.1, help = 'learning rate')
parser.add_argument('--weight_decay', default = 1e-4, help = 'weight_decay')
parser.add_argument('--momentum', default = 0.9, help = 'momentum')
parser.add_argument('--normalize', default = 'batchnorm', help = 'choose batchnorm or groupnorm')

parser.add_argument('--Epoch', default = 500, type=int,  help = ' Epoch ')
parser.add_argument('--batch_size', default = 128, type=int,  help = 'TRAIN batch size ')
parser.add_argument('--test_batch_size', default = 100, type=int,  help = 'TEST batch size')

#Data augmentation 
parser.add_argument('--cutout', default=False, help='apply cutout')
parser.add_argument('--n_masks', type=int, default=1, help='number of masks to cut out from image')
parser.add_argument('--length', type=int, default=16,  help='length of the masks')

#CutMix
parser.add_argument('--alpha', default=1.0, type=float, help='hyperparameter alpha')
parser.add_argument('--cutmix_prob', default=1.0, type=float, help='cutmix probability')

#Knowledge Distillation
parser.add_argument('--KD', default = True, help = 'Knowledge distillation yes or no ')

args = parser.parse_args()
#. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#

if __name__ == '__main__':
    main(args)
        