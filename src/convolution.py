# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 12:29:57 2021

@author: Administrator
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import datetime


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), padding=1)
    self.conv2 = nn.Conv2d(16, 8, kernel_size=(3,3), padding=1)
    self.fc1 = nn.Linear(8*8*8, 32)
    self.fc2 = nn.Linear(32, 2)
  
  def forward(self, x):
    out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
    out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
    out = out.view(-1, 8*8*8)
    out = torch.tanh(self.fc1(out))
    out = self.fc2(out)
    return out

def training_loop(n_epoches, optimizer, model, loss_fn, train_loader):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  for epoch in range(1, n_epoches+1):
    loss_train = 0.0
    for imgs, labels in train_loader:
      imgs = imgs.to(device=device)
      labels = labels.to(device=device)
      outputs = model(imgs)
      loss = loss_fn(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()
    if epoch == 1 or epoch % 10 == 0:
      print('{} Epoch{}: training loss{}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))

def validate(model, train_loader, val_loader):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  for name, loader in [('train', train_loader), ('val', val_loader)]:
    correct = 0
    total = 0
    with torch.no_grad():
      for imgs, labels in loader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
        print('Accuracy {}: {:.2f}'.format(name, correct/total))

def test_conv():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
  
  img, _ = cifar10[0]
  #print(img)
  
  label_map = {0: 0, 2: 1}
  #class_names = ['airplane', 'bird']
  cifar2 = [(img, label_map[label])
            for img, label in cifar10
            if label in [0, 2]]
  cifar2_val = [(img, label_map[label])
            for img, label in cifar10_val
            if label in [0, 2]]
  
  #conv = nn.Conv2d(3, 16, kernel_size=(3,3), stride=(1,1))
  #output = conv(img.unsqueeze(0))
  #print(img.unsqueeze(0).shape, output.shape)
  
  model = Net().to(device = device)
  #numel_list = [p.numel() for p in model.parameters()]
  #print(numel_list)
  train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
  val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
  loss_fn = nn.CrossEntropyLoss()
  training_loop(n_epoches=100, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader)
  
  validate(model, train_loader, val_loader)
  


if __name__ == '__main__':
  torch.set_printoptions(edgeitems=2)
  print('convolution')
  test_conv()
