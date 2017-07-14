from __future__ import print_function
import torch
import numpy as np

if torch.cuda.is_available():
    print('Cuda is avaliable')
else:
    print('No cuda')
