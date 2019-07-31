# -*- coding: utf-8 -*-


import os
import time
import warnings
import numpy as np
import torch as t
import visdom

from src.config import Config
from src import models


class Visualizer(object):


    def __init__(self, env='main'):
        self.vis = visdom.Visdom(env='main')
        self.index = {}
    
    def plot_scalar(self, name, y):
        """
        Parameters
        ----------
        name : str

        y : array-like but not torch.tensor
        """
        if name not in self.index:
            self.index[name] = 0
        x = self.index[name]
        self.vis.line(Y=np.array(y), X=np.array(x), win=name, update='append', name=name)
        self.index[name] += 1
    
    def plot_image(self, name, img):
        """
        Parameters
        ----------
        name : str

        img : array-like but not torch.tensor
        """
        self.vis.image(np.array(img), win=name)


class Logger(object):


    def __init__(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.root = directory
    
    def log(self, name, s, overwrite=True):
        """
        Parameters
        ----------
        name : str
            log file name
        
        s : str
            content
        """
        path = os.path.join(self.root, name)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(path, 'w' if overwrite else 'a') as f:
            f.write('[%s]: %s\n' % (current_time, s))


def load_model(name):
    model = None
    if not hasattr(models, name):
        warnings.warn("Model `%s` doesn't exist in your defined model set, \
            please check src/models/__init__.py" % name)
    else:
        model = getattr(models, name)
    return model


logger = Logger(Config.log_dir)
