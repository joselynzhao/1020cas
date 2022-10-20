#-*-coding:utf-8-*-

from random import random
from dnnlib import camera
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='6'
import numpy as np
import torch
import copy
import torch.distributed as dist
import torchvision
import click
import dnnlib
import legacy
import pickle

from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from torchvision import transforms, utils
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from training.networks import Encoder
import inspect
import collections
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss





def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()



def num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


def stack_imgs(imgs):
    img = torch.stack(imgs, dim=2)
    return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)


def proc_img(img):
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

# handle camera
def get_camera_metrices(cameras,device):  # 直接把camera1 和camera2 穿进去。
    cam_0 = cameras['camera_0']
    cam_1 = cameras['camera_1']
    cam_2 = cameras['camera_2']
    cam_0 = cam_0[:,0,:,:].squeeze()
    cam_1 = cam_1[:,0,:,:].squeeze()
    cam_0 = cam_0.to(device).to(torch.float32)
    cam_1 = cam_1.to(device).to(torch.float32)
    cam_2 = cam_2.to(device).to(torch.float32)
    return cam_0,cam_1,cam_2,None
