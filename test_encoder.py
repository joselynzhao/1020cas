# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from random import random
from dnnlib import camera
import os
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

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


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

    
class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        source, target = (source + 1) / 2, (target + 1) / 2
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss 






def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



@click.command()
@click.option("--data", type=str, default=None)
@click.option("--g_ckpt", type=str, default=None)
@click.option("--e_ckpt", type=str, default=None)
@click.option("--max_steps", type=int, default=1000000)
@click.option("--batch", type=int, default=8)
@click.option("--lr", type=float, default=0.0001)
@click.option("--local_rank", type=int, default=0)
@click.option("--vgg", type=float, default=1.0)
@click.option("--l2", type=float, default=1.0)
@click.option("--adv", type=float, default=0.05)   
@click.option("--tensorboard", type=bool, default=True)
@click.option("--outdir", type=str, required=True)
def main(data, outdir, g_ckpt, e_ckpt,
         max_steps, batch, lr, local_rank, vgg,
         l2, adv, tensorboard):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(ori_main, args=(world_size, data, outdir, g_ckpt, e_ckpt, max_steps, batch, lr, vgg, l2, adv, tensorboard), nprocs=world_size, join=True)


def ori_main(rank, word_size, data, outdir, g_ckpt, e_ckpt,
         max_steps, batch, lr, vgg,
         l2, adv, tensorboard):
    local_rank = rank
    setup(rank, word_size)

    random_seed = 22
    np.random.seed(random_seed)
    use_image_loss = False

    num_gpus = 8
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda', local_rank)

    with dnnlib.util.open_url(e_ckpt) as f:
        resume_data = legacy.load_network_pkl(f)
        E = copy.deepcopy(resume_data['E'].module).eval().requires_grad_(False).to(device)
    # load the dataset
    training_set_kwargs = dict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=False, xflip=False)
    data_loader_kwargs  = dict(pin_memory=True, num_workers=1, prefetch_factor=1)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    #training_set_sampler  = misc.InfiniteSampler(dataset=training_set, rank=local_rank, num_replicas=num_gpus, seed=random_seed)  # for now, single GPU first.
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, shuffle=False, batch_size=batch//num_gpus, **data_loader_kwargs)
    #training_set_iterator = iter(training_set_iterator)
    #print('Num images: ', len(training_set))
    #print('Image shape:', training_set.image_shape)
    for input_batch in tqdm(training_set_iterator):
        img,_,_ = input_batch
        rec_ws, rec_cm  = E(img.to(device).float()/127.5-1)
        print(rec_cm)
    cleanup()

if __name__ == "__main__":
    main()
