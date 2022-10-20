# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-#-*-coding:utf-8-*-# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import re
import time
import glob
import copy
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
import cv2
from renderer import Renderer
from training.my_utils import *

# ----------------------------------------------------------------------------
# hpcl
torch.cuda.current_device()
torch.cuda._initialized = True
# torch.cuda.empty_cache()

# ----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'
data_path={
    'hpcl':'./output/create_dataset/test_dataset/trunc075/images',
    'jdt':'/workspace/datasets/car_zj/trunc075/images'
}

# 不需要指定任何参数，只需要修改待测试encoder的列表，主要调整seed（选择测试样列）
@click.command()
@click.pass_context
# @click.option('--network', 'network_pkl', help='Network pickle filename', default='./car_model.pkl')
@click.option('--encoder', 'encoder_pkl', help='Network pickle filename',
              default='./output/psp_mvs_one_img/debug/checkpoints/network-snapshot-000000.pkl')
@click.option("--which_server", type=str, default='hpcl')
# @click.option('--insert_layer',type=int, default=3)
# @click.option('--group_name',type=str, default='01')
@click.option("--batch", type=int, default=8)
# @click.option("--which_c", type=str, default='p2')
@click.option("--local_rank", type=int, default=0)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
# @click.option('--outdir', help='Where to save the output images', type=str, metavar='DIR',
#               default='../output/test_encoders/show_psp_case2_encoder_v4')
def generate_images(
        ctx: click.Context,
        # network_pkl: str,
        encoder_pkl: str,
        # data: str,
        # insert_layer:int,
        batch: int,
        # which_c: str,
        local_rank:int,
        class_idx: Optional[int],
        which_server:str):

    data = data_path[which_server]
    random_seed = 22
    np.random.seed(random_seed)

    num_gpus = torch.cuda.device_count()  # 自动获取显卡数量
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda')
    with dnnlib.util.open_url(encoder_pkl) as f:
        print('Loading encoder from "%s"...' % encoder_pkl)
        encoder = legacy.load_network_pkl(f)
        E = encoder['E'].to(device)
        G = encoder['G'].to(device)
        # uploaded encoder and generator

    # load the dataset
    # data_dir = os.path.join(data, 'images')
    training_set_kwargs = dict(class_name='training.dataset.ImageFolderDataset_psp_case1', path=data, use_labels=False,
                               xflip=True)
    data_loader_kwargs = dict(pin_memory=True, num_workers=1, prefetch_factor=1)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=local_rank, num_replicas=num_gpus,
                                                seed=random_seed)  # for now, single GPU first.
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                        batch_size=batch // num_gpus, **data_loader_kwargs)
    training_set_iterator = iter(training_set_iterator)
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)

    # store_dir = os.path.join(f'./output/test_encoders/In_testset/Group_{group_name}')  # 用于服务器测试
    store_dir = './output/test_encoders/trunc075'
    os.makedirs(store_dir, exist_ok=True)
    # seed_list = [11]

    ws_avg = G.mapping.w_avg[None, None, :]
    img_1, img_2, camera1, camera2, w = next(training_set_iterator)
    img_1 = img_1.to(device).to(torch.float32) / 127.5 - 1
    w = w.to(device).to(torch.float32)

    camera1 = get_camera_metrices(camera1, device)
    views = source_views = camera1[2]  # mode uvr infos

    rec_ws, feature = E(img_1)
    rec_ws += ws_avg
    gen_img1 = G.get_final_output(styles=w, features=feature, views=views, source_views=source_views)

    from torchvision import utils
    root_path = encoder_pkl.split('.')[1]
    root_path = root_path.split('/')
    file_name = '-'.join(root_path[2:])
    # os.makedirs(f"{store_dir}/sample", exist_ok=True)
    with torch.no_grad():
        sample = torch.cat([img_1.detach(),gen_img1.detach()])
        utils.save_image(
            sample,
            f"{store_dir}/{file_name}.png",
            nrow=int(batch),
            normalize=True,
            range=(-1, 1),
        )

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
# running orders
# python3 test_encoder_for_gen_image.py --encoder=./output/psp_mvs_one_img/debug/checkpoints/network-snapshot-000000.pkl --which_server=jdt
# 输出目录：./output/test_encoder/trunc075/$encoder_name
