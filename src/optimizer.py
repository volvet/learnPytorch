# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:17:38 2021

@author: Administrator
"""

import torch

def model(t_u, w, b):
  return w*t_u + b

def loss_fn(t_p, t_c):
  squared_diffs = (t_p - t_c) ** 2
  return squared_diffs.mean()

def training_loop(epoches, optimizer, params, t_u, t_c):
  for epoch in range(1, epoches + 1):
    t_p = model(t_u, *params)
    loss = loss_fn(t_p, t_c)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
      print('Eopch %d, Loss %f' % (epoch, float(loss)))
      
  return params

def test_optim():
  print(dir(torch.optim))
  t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
                    3.0, -4.0, 6.0, 13.0, 21.0])
  t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                    33.9, 21.8, 48.4, 60.4, 68.4])
  t_un = 0.1 * t_u
  params = torch.tensor([1.0, 0.1], requires_grad=True)
  learning_rate = 1e-5
  optimizer = torch.optim.SGD([params], lr=learning_rate)
  
  params = training_loop(5000, optimizer, params, t_u=t_un, t_c=t_c)
  print(params)
  
  



if __name__ == '__main__':
  test_optim()
  