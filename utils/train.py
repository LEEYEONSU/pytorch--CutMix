import os 
import time
import torch
import argparse
import torchvision
import numpy as np 
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms

from utils.function import *
from utils.cutmix import cutmix
from model.SE import SEresnet
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

best_prec1 = 0
def main(args):

        global best_prec1
        
        # CIFAR-10 Training & Test Transformation
        print('. . . . . . . . . . . . . . . .PREPROCESSING DATA . . . . . . . . . . . . . . . .')
        TRAIN_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if args.cutout : 
                TRAIN_transform.transforms.append(Cutout(n_masks = args.n_masks, length = args.length))

        VAL_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),      
        ])

        # CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(root = '../data/', 
                                                     train = True, 
                                                     transform = TRAIN_transform, 
                                                     download = True)
        val_dataset = torchvision.datasets.CIFAR10(root = '../data/', 
                                                     train = False, 
                                                     transform = VAL_transform,
                                                     download = True)

        # Data loader 
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                pin_memory = True, 
                                                drop_last = True, 
                                                batch_size = args.batch_size , 
                                                shuffle=True)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                pin_memory = True, 
                                                batch_size = args.batch_size , 
                                                shuffle=False)

        # Device Config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.normalize == 'groupnorm':
                model = SEresnet_gn()

        elif args.normalize == 'groupnorm+ws':
                model = SEresnet_gn_ws()
        else : 
                 model = SEresnet()

        model = model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters() , lr = args.lr , weight_decay = args.weight_decay, momentum = args.momentum)
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones = [250,375], gamma = 0.1)

        if args.evaluate :
                model.load_state_dict(torch.load('./save_model/model.th'))  
                model.to(device)
                validation(args, val_loader, model, criterion)

        #  Epoch = args.Epoch
        for epoch_ in range(0, args.Epoch):
                print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
                train_one_epoch(args, train_loader, model, criterion, optimizer, epoch_)
                lr_schedule.step()

                prec1 = validation(args, val_loader, model, criterion)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                if epoch_ > 0 and epoch_ % args.save_every == 0:
                        save_checkpoint({
                        'epoch': epoch_ + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.pt'))

                save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'model.pt')) 
        
                print('THE BEST MODEL prec@1 : {best_prec1:.3f} saved. '.format(best_prec1 = best_prec1))

def train_one_epoch(args, train_loader, model, criterion, optimizer, epoch_):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.train()

        end = time.time()

        for i, (input_, target) in enumerate(train_loader):

                input_v = input_.to(device)
                target = target.to(device)
                batch = (input_v, target)

                r = torch.rand(1)
                if args.alpha > 0 and args.cutmix_prob > r :
                        input_v, target_a, target_b, lam = cutmix(batch, args.alpha)
                        output = model(input_v)
                        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1 - lam)
                
                else:
                        output = model(input_v)
                        loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), input_.size(0))
                top1.update(prec1.item(), input_.size(0))

                batch_time.update( time.time() - end )
                end = time.time()

                if i % args.print_freq == 0:
                        print('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                epoch_, i,len(train_loader),batch_time=batch_time,loss=losses,top1=top1))

def validation(args, val_loader, model, criterion):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.eval()

        end = time.time()

        with torch.no_grad():
                for i, (input_, target) in enumerate(val_loader):
                        input_v = input_.to(device)
                        target = target.to(device)
                        # target_v = target

                        output = model(input_v)
                        loss = criterion(output, target)

                        # loss = loss.float()

                        prec1 = accuracy(output.data, target)[0]
                        losses.update(loss.item(), input_.size(0))
                        top1.update(prec1.item(), input_.size(0))

                        batch_time.update(time.time() - end)
                        end = time.time()

                        if i % args.print_freq == 0:
                                print('Test: [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                        i, len(val_loader), batch_time=batch_time, loss=losses,
                                        top1=top1))

                print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

        return top1.avg

