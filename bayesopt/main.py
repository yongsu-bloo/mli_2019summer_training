### IMPORT MODULES
# basic
import numpy as np
# torch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# utils
import random, time, os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import botorch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-b', "--batch_size", type=int, default=128, help='batch size(default=128)')
    parser.add_argument('--no-reverse', help='not to reverse input seq', action='store_true')
    global args
    args = parser.parse_args()
