# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:03:29 2020

@author: Administrator
"""

import numpy as np
import torch
from torch.autograd import Variable


def tensor():
  print('Hello, PyTorch.tensor')
  y = torch.eye(3,4)
  print(y)
  print(torch.is_tensor(y))
  print(torch.is_storage(y))
  print(torch.numel(y))
  
  x = [12, 23, 34, 45, 56, 67, 78]
  x1 = np.array(x)
  print(torch.from_numpy(x1))
  print(torch.linspace(2, 10, steps=25))
  print(torch.logspace(-10, 10, steps=15))
  print(torch.rand(10))
  print(torch.rand(4, 5))
  print(torch.randn(10))
  print(torch.randperm(10))
  print(torch.arange(10, 40, 2))
  d = torch.randn(4, 5)
  print(d)
  print(torch.argmin(d))
  print(torch.argmin(d, dim=1))
  print(torch.zeros(4, 5))
  
  x = torch.randn(4, 5)
  print(x)
  print(torch.cat((x, x)))
  print(torch.cat((x,x,x), 1))

  a = torch.randn(4, 4)
  print(a)
  print(torch.chunk(a, 2))
  print(torch.chunk(a, 2, 1))
  
  print(torch.gather(torch.tensor([[11, 12], [23, 24]]), 1,
                     torch.LongTensor([[0, 0], [1, 0]])))
  
  x = torch.randn(4, 5)
  print(x)
  print(x.t())
  print(x.transpose(1, 0))
  
  
def probability():
  print('Hello, PyTorch.probability')
  print(torch.Tensor(4, 4).uniform_(0, 1))
  print(torch.bernoulli(torch.Tensor(4, 4).uniform_(0, 1)))
  x = torch.tensor([10., 10., 13., 10.,
                    34., 45., 65., 67.,
                    87., 89., 87., 34.])
  print(torch.multinomial(x, 5, replacement=True))
  print(torch.normal(mean=0.5, std=torch.arange(1, 0, -0.5)))
  
  a, b = 4, 5
  x1 = Variable(torch.randn(a, b), requires_grad=True)
  x2 = Variable(torch.randn(a, b), requires_grad=True)
  x3 = Variable(torch.randn(a, b), requires_grad=True)
  c = x1 * x2
  d = a + x3
  print(c)
  print(d)
  e = torch.sum(d)
  print(e)
  d = torch.randn(4, 5)
  print(d)
  print(torch.mean(d, dim=0))
  print(torch.mean(d, dim=1))
  print(torch.mode(d))
  
  def forward(x):
    return x * w
  
  def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
  
  x_data = [11., 22., 33.]
  y_data = [21., 14., 64.]
  w = Variable(torch.Tensor([1.0]), requires_grad=True)
  print('predict (before training)', 4, forward(4).data[0])
  for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
      l = loss(x_val, y_val)
      l.backward()
      print('\tgrad: ', x_val, y_val, w.grad.data[0])
      w.data = w.data - 0.01 * w.grad.data
      print(w)
      w.grad.data.zero_()
    print('progress: ', epoch, l.data[0])
  


def model(t_u, w, b):
  return w*t_u + b

def loss_fn(t_p, t_c):
  squared_diffs = (t_p - t_c) ** 2
  return squared_diffs.mean()

def training_loop(epoches, lr, params, t_u, t_c):
  for epoch in range(1, epoches+1):
    if params.grad is not None:
      params.grad.zero_()
    t_p = model(t_u, *params)
    loss = loss_fn(t_p, t_c)
    loss.backward()

    with torch.no_grad():
        params -= lr * params.grad
    if epoch % 500 == 0:
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
  return params



def test_tensor():
  img_t = torch.randn(3, 5, 5)
  weights = torch.tensor([0.2126, 0.7125, 0.0722])
  img_gray_native = img_t.mean(-3)
  print(img_gray_native.shape)
  unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
  print(unsqueezed_weights)
  img_weights = (img_t * unsqueezed_weights)
  img_gray_weighted = img_weights.sum(-3)
  print(img_weights.shape, img_gray_weighted.shape)
  img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
  weight_named = weights.refine_names(..., 'channels')
  #print(img_named)
  weights_aligned = weight_named.align_as(img_named)
  print(weights_aligned)
  
  points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
  print(points.storage())
  points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
  print(points_gpu.storage())

  t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
                    3.0, -4.0, 6.0, 13.0, 21.0])
  t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                    33.9, 21.8, 48.4, 60.4, 68.4])
  t_un = 0.1 * t_u
  t_c = torch.tensor(t_c)
  t_u = torch.tensor(t_u)
  params = torch.tensor([1.0, 0.0], requires_grad=True)
  #print(params.grad)
  #loss = loss_fn(model(t_u, *params), t_c)
  #loss.backward()
  #print(params.grad)
  params = training_loop(10000, 0.01, params, t_un, t_c)
  print(params)
  

def main():
  #tensor()
  #probability()
  test_tensor()

if __name__ == '__main__':
  main()
