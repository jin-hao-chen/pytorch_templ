# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F

from src.models import BaseModule


class Custom(BaseModule):

    
    def __init__(self):
        super(Custom, self).__init__()
        # TODO: your code here
    
    def forward(self, x):
        # TODO: your code here
        out = x
        return out

