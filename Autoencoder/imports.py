from os import error, mkdir
import io
import logging
import datetime
import time

# Setup logging
namespace = str(datetime.datetime.now())
figdir = "figures/"+namespace
mkdir(figdir)
logging.basicConfig(filename=figdir+"/log"+namespace+".txt",level=logging.DEBUG,filemode='w')

from numpy.core.fromnumeric import argmax
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import matplotlib
from itertools import permutations
import pickle

import os
logging.getLogger('PIL').setLevel(logging.WARNING)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


logging.info("Device is"+device)
if device == 'cuda':
    try:
        import cupy as cp
    except:
        import numpy as cp
else:
    import numpy as cp

time.sleep(10)