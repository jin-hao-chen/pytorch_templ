#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)
import fire
import torch as t
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src import utils


opts = Config()
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = utils.load_model('model').to(device)


def parse_args(func):
    def wrapper_fn(*args, **kwargs):
        opts.parse_args(**kwargs)
        ret = func(**kwargs)
        return ret
    return wrapper_fn

@parse_args
def train(**kwargs):
    print('train')

@parse_args
def eval(**kwargs):
    print('eval')

def help():
    print('help')

def main():
    fire.Fire()


if __name__ == "__main__":
    main()

