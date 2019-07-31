# -*- coding: utf-8 -*-


import os
import time
import warnings
import torch as t
import torch.nn as nn


class BaseModule(nn.Module):


    def __init__(self):
        super(BaseModule, self).__init__()
        self.name = str(self.__class__).split("'")[1].split('.')[-1]
    
    def save(self, root='checkpoints/'):
        model_dir = os.path.join(root, self.name)
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        path = os.path.join(model_dir, current_time + '.pth')
        t.save(self.state_dict(), path)

    def load(self, path, debug=False):
        if not os.path.exists(path):
            warnings.warn("Path %s doesn't exist")
        else:
            self.load_state_dict(t.load(path))
            if debug:
                print("Load weights from %s" % path)

