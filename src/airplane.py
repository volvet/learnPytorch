#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:44:07 2021

@author: volvetzhang
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt


#def softmax(x):
#  return torch.exp(x) / torch.exp(x).sum()

def airplane():
  cur_path = os.path.split(os.path.realpath(__file__))[0]
  cifar10 = datasets.CIFAR10(cur_path, train=True, download=False,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4915, 0.4823, 0.4644),
                                                    (0.2410, 0.2435, 0.2616))]))
  cifar10_val = datasets.CIFAR10(cur_path, train=False, download=False,
                                 transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4915, 0.4823, 0.4644),
                                                        (0.2410, 0.2435, 0.2616))]))
  
  label_map = {0: 0, 2: 1}
  cifar2 = [(img, label_map[label])
            for img, label in cifar10
            if label in [0, 2]]
  cifar2_val = [(img, label_map[label])
            for img, label in cifar10_val
            if label in [0, 2]]
  
  
  #print(len(cifar2))
  
  n_out = 2
  model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.Tanh(),
            nn.Linear(512, n_out),
            nn.LogSoftmax(dim=1)
          )
  
  learning_rate = 1e-2
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  loss_fn = nn.NLLLoss()
  n_epoches = 100
  for epoch in range(n_epoches):
    for img, label in cifar2:
      out = model(img.view(-1).unsqueeze(0))
      loss = loss_fn(out, torch.tensor([label]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()   
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
  

def airplane2():
  cur_path = os.path.split(os.path.realpath(__file__))[0]
  cifar10 = datasets.CIFAR10(cur_path, train=True, download=False,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4915, 0.4823, 0.4644),
                                                    (0.2410, 0.2435, 0.2616))]))
  cifar10_val = datasets.CIFAR10(cur_path, train=False, download=False,
                                 transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4915, 0.4823, 0.4644),
                                                        (0.2410, 0.2435, 0.2616))]))
  
  label_map = {0: 0, 2: 1}
  cifar2 = [(img, label_map[label])
            for img, label in cifar10
            if label in [0, 2]]
  cifar2_val = [(img, label_map[label])
            for img, label in cifar10_val
            if label in [0, 2]]
  
  train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
  model = nn.Sequential(
             nn.Linear(3072, 128),
             nn.Tanh(),
             nn.Linear(128, 2),
             nn.LogSoftmax(dim=1))

  learning_rate = 1e-2
  loss_fn = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  n_epoches = 100
  
  for epoch in range(n_epoches):
    for imgs, labels in train_loader:
      outputs = model(imgs.view(imgs.shape[0], -1))      
      loss = loss_fn(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
    
  correct = 0
  total = 0
  with torch.no_grad():
    for imgs, labels in train_loader:
      outputs = model(imgs.view(imgs.shape[0], -1))
      _, predicted = torch.max(outputs, dim=1)
      total += labels.shape[0]
      correct += int((predicted == labels).sum())
      
  print("Acc: %f" % (correct/total))
  
  val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
  correct = 0
  total = 0
  with torch.no_grad():
    for imgs, labels in val_loader:
      outputs = model(imgs.view(imgs.shape[0], -1))
      _, predicted = torch.max(outputs, dim=1)
      total += labels.shape[0]
      correct += int((predicted == labels).sum())
  print("Acc of val: %f" % (correct/total))


if __name__ == '__main__':
  #os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
  torch.manual_seed(123)
  torch.set_printoptions(edgeitems=2)
  airplane2()



