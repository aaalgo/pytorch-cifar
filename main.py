#!/usr/bin/env python
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.insert(0, '../picpac/build/lib.linux-x86_64-%d.%d' % sys.version_info[:2])

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR 

import numpy as np
import picpac

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# NOTE THAT PICPAC CHANNEL
mu1 = [0.4465, 0.4822, 0.4914]  
sigma1 = [0.2010, 0.1994, 0.2023]

BATCH = 128
SIZE = 32

normalize = {"type": "normalize", "mean": [x * 255 for x in mu1], "std": [x * 255 for x in sigma1]}
picpac_config = {"db": 'cifar10-train.picpac',
          "loop": True,
          "shuffle": True,
          "reshuffle": True,
          "annotate": False,
          "channels": 3,
          "stratify": True,
          "dtype": "float32",
          "batch": BATCH,
          "order": "NCHW",
          "transforms": [
              {"type": "augment.flip", "horizontal": True, "vertical": False},
              normalize,
              {"type": "clip", "size": SIZE, "shift": 4},
          ]
         }

stream = picpac.ImageStream(picpac_config)

val_config = {"db": 'cifar10-test.picpac',
          "loop": False,
          "channels": 3,
          "dtype": "float32",
          "batch": BATCH,
          "order": "NCHW",
          "transforms": [
                normalize,
                {"type": "clip", "size": SIZE},
          ]
         }
val_stream = picpac.ImageStream(val_config)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch_steps = stream.size() // BATCH
    for batch_idx in range(epoch_steps):
        meta, inputs = stream.next()
        targets = meta.labels.astype(np.int64)
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, epoch_steps, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    val_steps = val_stream.size() // BATCH
    val_stream.reset()
    step = 0
    for meta, inputs in val_stream:
        targets = meta.labels.astype(np.int32)
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(step, val_steps, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        step += 1

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+350):
    scheduler.step()
    train(epoch)
    test(epoch)

