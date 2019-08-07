# -*- coding: utf-8 -*-


import os
import time
import warnings
import numpy as np
import torch as t
import visdom
import scipy
import scipy.io as io
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial


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


def adjust_lr(optimizer, epoch, lr_decay, initial_lr):
    lr = initial_lr / (1.0 + epoch * lr_decay)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    retur lr


def pil2cv(image):
    """Convert pillow image to cv image
    """
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def cv2pil(image):
    """Convert cv image to pillow image
    """
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def cv2array(image):
    """Convert ndarray type image format from BGR to RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def pil2array(image):
    """Convert pillow image to ndarray
    """
    return np.array(image)

def tensor2numpy(tensor):
    """Convert tensor to ndarray
    """
    return tensor.cpu().detach().numpy()

def array2tensor(array, device='auto'):
    """Convert ndarray to tensor on ['cpu', 'gpu', 'auto']
    """
    assert device in ['cpu', 'gpu', 'auto'], "Invalid device"
    if device != 'auto':
        return t.tensor(array).float().to(t.device(device))
    if device == 'auto':
        return t.tensor(array).float().to(t.device('cuda' if t.cuda.is_available() else 'cpu'))

def load_mat(path):
    return io.loadmat(path)

logger = Logger(Config.log_dir)

