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

import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

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
            loss += weight * self.criterion(source, target)

        return loss


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

# --data=./output/car_dataset_3w_test/images --g_ckpt=car_model.pkl --outdir=../car_stylenrf_output/psp_case1/debug
@click.command()
@click.option("--data", type=str, default=None)
@click.option("--g_ckpt", type=str, default=None)
@click.option("--e_ckpt", type=str, default=None)
@click.option("--max_steps", type=int, default=1000000)
@click.option("--batch", type=int, default=4)
@click.option("--lr", type=float, default=0.0001)
@click.option("--local_rank", type=int, default=0)
@click.option("--lambda_w", type=float, default=1.0)
@click.option("--lambda_img", type=float, default=1.0)
@click.option("--adv", type=float, default=0.05)
@click.option("--tensorboard", type=bool, default=True)
@click.option("--outdir", type=str, required=True)
def main(data, outdir, g_ckpt, e_ckpt,
         max_steps, batch, lr, local_rank, lambda_w,
         lambda_img, adv, tensorboard):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(ori_main, args=(
    world_size, data, outdir, g_ckpt, e_ckpt, max_steps, batch, lr, lambda_w, lambda_img, adv, tensorboard), nprocs=world_size,
                                join=True)



def ori_main(rank, word_size, data, outdir, g_ckpt, e_ckpt,
             max_steps, batch, lr, lambda_w,
             lambda_img, adv, tensorboard):
    local_rank = rank
    setup(rank, word_size)

    random_seed = 22
    np.random.seed(random_seed)
    use_image_loss = False

    num_gpus = torch.cuda.device_count()  # 自动获取显卡数量
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda', local_rank)

    # load the pre-trained model
    if os.path.isdir(g_ckpt):  #基本模型信息
        import glob
        g_ckpt = sorted(glob.glob(g_ckpt + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % g_ckpt)
    with dnnlib.util.open_url(g_ckpt) as fp:
        network = legacy.load_network_pkl(fp)
        G = network['G_ema'].requires_grad_(False).to(device)
    # 直接使用G D
    # G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    #　拷贝使用 G
    from training.networks import Generator
    from torch_utils import  misc
    with torch.no_grad():
        G2 = Generator(*G.init_args,**G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G,G2,require_all=True)
    G = copy.deepcopy(G2).eval().requires_grad_(False).to(device)

    from models.encoders.psp_encoders import GradualStyleEncoder
    E = GradualStyleEncoder(50, 3, G.mapping.num_ws, 'ir_se').to(device)  # psp encoder
    if num_gpus >1:
       E = DDP(E, device_ids=[rank], output_device=rank, find_unused_parameters=True) # broadcast_buffers=False

    # E_optim = optim.Adam(E.parameters(), lr=lr*0.1, betas=(0.9, 0.99))
    E_optim = optim.Adam(E.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(E_optim, step_size=5000, gamma=0.1)
    requires_grad(E, True)

    # load the dataset
    # data_dir = os.path.join(data, 'images')
    training_set_kwargs = dict(class_name='training.dataset.ImageFolderDataset_psp_case1', path=data, use_labels=False, xflip=True)
    data_loader_kwargs  = dict(pin_memory=True, num_workers=1, prefetch_factor=1)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler  = misc.InfiniteSampler(dataset=training_set, rank=local_rank, num_replicas=num_gpus, seed=random_seed)  # for now, single GPU first.
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch//num_gpus, **data_loader_kwargs)
    training_set_iterator = iter(training_set_iterator)
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)


    start_iter = 0
    pbar = range(max_steps)
    pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01)

    e_loss_val = 0
    loss_dict = {}
    # vgg_loss   = VGGLoss(device=device)
    truncation = 0.5  # 固定的
    ws_avg = G.mapping.w_avg[None, None, :]

    if SummaryWriter and tensorboard:
        logger = SummaryWriter(logdir='./checkpoint')

    for idx in pbar:
        i = idx + start_iter
        if i > max_steps:
            print("Done!")
            break

        E_optim.zero_grad()  # zero-out gradients
        # get data infor
        img_1,img_2,camera1,camera2,w = next(training_set_iterator)
        # handle image
        img_1 = img_1.to(device).to(torch.float32) / 127.5 - 1
        img_2 = img_2.to(device).to(torch.float32) / 127.5 - 1

        # handle w  # 暂时用不到
        w = w.to(device).to(torch.float32)

        # handle camera
        def get_camera_metrices(cameras):  # 直接把camera1 和camera2 穿进去。
            cam_0 = cameras['camera_0']
            cam_1 = cameras['camera_1']
            cam_2 = cameras['camera_2']
            cam_0 = cam_0[:,0,:,:].squeeze()
            cam_1 = cam_1[:,0,:,:].squeeze()
            cam_0 = cam_0.to(device).to(torch.float32)
            cam_1 = cam_1.to(device).to(torch.float32)
            cam_2 = cam_2.to(device).to(torch.float32)
            return cam_0,cam_1,cam_2,None
        camera1 = get_camera_metrices(camera1)
        camera2 = get_camera_metrices(camera2)

        rec_ws_1 = E(img_1)
        rec_ws_1 +=ws_avg
        gen_img1 = G.get_final_output(styles=rec_ws_1, camera_matrices=camera1)
        gen_img2 = G.get_final_output(styles=rec_ws_1, camera_matrices=camera2)
        rec_ws_2 = E(gen_img2)
        rec_ws_2 += ws_avg

        # define loss
        loss_dict['loss_w'] = F.smooth_l1_loss(rec_ws_1, rec_ws_2).mean() * lambda_w
        loss_dict['img1_lpips'] = loss_fn_alex(gen_img1.cpu(), img_1.cpu()).mean().to(device) * lambda_img
        loss_dict['img2_lpips'] = loss_fn_alex(gen_img2.cpu(), img_2.cpu()).mean().to(device) * lambda_img

        E_loss = sum([loss_dict[l] for l in loss_dict])
        E_loss.backward()
        E_optim.step()
        scheduler.step()

        desp = '\t'.join([f'{name}: {loss_dict[name].item():.4f}' for name in loss_dict])
        pbar.set_description((desp))

        if SummaryWriter and tensorboard:
            logger.add_scalar('E_loss/total', e_loss_val, i)
            # logger.add_scalar('E_loss/vgg', vgg_loss_val, i)
            # logger.add_scalar('E_loss/l2', l2_loss_val, i)
            # logger.add_scalar('E_loss/adv', adv_loss_val, i)

        if i % 200 == 0:
            os.makedirs(f'{outdir}/sample', exist_ok=True)
            with torch.no_grad():

                sample = torch.cat([img_1.detach(),gen_img1.detach(),img_2.detach(), gen_img2.detach()])
                utils.save_image(
                    sample,
                    f"{outdir}/sample/{str(i).zfill(6)}.png",
                    nrow=int(batch),
                    normalize=True,
                    range=(-1, 1),
                )

        if i % 1000 == 0:
            os.makedirs(f'{outdir}/checkpoints', exist_ok=True)
            snapshot_pkl = os.path.join(f'{outdir}/checkpoints/', f'network-snapshot-{i // 1000:06d}.pkl')
            snapshot_data = {}  # dict(training_set_kwargs=dict(training_set_kwargs))
            snapshot_data['E'] = E
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
    cleanup()


if __name__ == "__main__":
    main()
