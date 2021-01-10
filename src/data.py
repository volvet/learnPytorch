# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:57:39 2021

@author: Administrator
"""

import os
import torch
import numpy as np
import cv2


def main():
  print('Hello, pytorch data representation')
  fn = 'lena.png'
  cur_path = os.path.split(os.path.realpath(__file__))[0]
  img = cv2.imread('/'.join([cur_path, fn]))
  img = torch.from_numpy(img)
  img = img.permute(2, 0, 1)
  print(img.shape)
  


if __name__ == '__main__':
  main()