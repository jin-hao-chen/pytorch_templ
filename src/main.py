#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)
import fire
import torch as t

from src.config import Config
from src import utils


opts = Config()
model = utils.load_model('model')


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
