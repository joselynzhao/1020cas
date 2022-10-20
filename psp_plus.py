#-*-coding:utf-8-*-

"""
This file defines the core research contribution
"""
import matplotlib

matplotlib.use('Agg')
import math
import dnnlib
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
import os
from training.networks import Generator
from torch_utils import misc
import glob
import copy
import legacy
import lpips
from models.encoders.psp_encoders import GradualStyleEncoder1
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')

from training.stylenerf import MatchConv

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class Mynerf(nn.Module):
    def __init__(self,g_ckpt):
        super(Mynerf,self).__init__()
        self.generater = self.init_generater(g_ckpt)
        self.encoder = GradualStyleEncoder1(50, 3, self.generater.mapping.num_ws, 'ir_se')
        self.match = MatchConv(512,512)
        self.ws_avg = self.generater.w_avg[None,None,:]

    def init_generater(self,g_ckpt):
        # load the pre-trained model
        if os.path.isdir(g_ckpt):  # 基本模型信息
            g_ckpt = sorted(glob.glob(g_ckpt + '/*.pkl'))[-1]
        print('Loading networks from "%s"...' % g_ckpt)
        with dnnlib.util.open_url(g_ckpt) as fp:
            network = legacy.load_network_pkl(fp)
            G = network['G_ema'].requires_grad_(False)
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs)
            misc.copy_params_and_buffers(G, G2, require_all=True)
        G = copy.deepcopy(G2).eval().requires_grad_(False)
        return G


    def forward(self,img1,img2,camera1,camera2):
        rec_ws1,c1,c2,c3 = self.encoder(img1)
        rec_ws1+=self.ws_avg
        gen_img = self.generater.get_final_output(styles=rec_ws1, camera_matrices=camera1,c=(c1,c2,c3))





