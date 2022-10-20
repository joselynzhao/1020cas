#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
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



# ----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'


# 无参数运行。
@click.command()
@click.pass_context
# @click.option('--network', 'network_pkl', help='Network pickle filename', default='./car_model.pkl')
@click.option('--encoder', 'encoder_pkl', help='Network pickle filename', default='./output/psp_mvs_one_img/debug/checkpoints/network-snapshot-000000.pkl')
@click.option('--img_dir', help='test image path', default='./real_test_images/real_test_images3')
# @click.option('--batch',type=int, default=5)
# @click.option('--which_c', help='which_c', default='p2')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')

# @click.option('--outdir', help='Where to save the output images', type=str, metavar='DIR',
#               default='real_test_images1')
def generate_images(
        ctx: click.Context,
        encoder_pkl: str,
        img_dir: str,
        class_idx: Optional[int],
):
    local_rank = 0
    random_seed = 22
    np.random.seed(random_seed)

    num_gpus = torch.cuda.device_count()  # 自动获取显卡数量
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda')
    # batch = 4

    with dnnlib.util.open_url(encoder_pkl) as f:
        print('Loading encoder from "%s"...' % encoder_pkl)
        encoder = legacy.load_network_pkl(f)
        E = encoder['E'].to(device)
        G = encoder['G'].to(device)


    # store_dir = encoder_pkl.split('/')[-1].split('.')[0]
    store_dir = os.path.join('./output/test_encoders',img_dir.split('/')[-1])
    os.makedirs(store_dir, exist_ok=True)



    def get_test_image(image_path):
        image = np.array(PIL.Image.open(image_path))
        if image.shape[-1] != resolution:
            image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).to(device).to(torch.float32) / 127.5 - 1
        image = image.unsqueeze(0)
        return image

    def get_test_camera(camera):
        cam_0 = camera['camera_0']
        cam_1 = camera['camera_1']
        cam_2 = camera['camera_2']
        cam_0 = cam_0[0:1, :, :]
        cam_1 = cam_1[0:1, :, :]
        cam_0 = torch.from_numpy(cam_0).to(device).to(torch.float32)
        cam_1 = torch.from_numpy(cam_1).to(device).to(torch.float32)
        cam_2 = torch.from_numpy(cam_2).to(device).to(torch.float32)
        cam_2 = cam_2.unsqueeze(0)
        return cam_0, cam_1, cam_2, None

    # 获取pose
    import random
    seed = random.uniform(0,1)
    u_samples = []
    n_steps = 16

    u_samples.append(seed)
    for i in range(n_steps - 1):
        k = seed + (i + 1) / n_steps
        if k >= 1:
            k = k - 1
        u_samples.append(k)

    resolution = 256
    pose_list = np.array([0, 2, 5, 8, 10])
    select_pose = [u_samples[i] for i in pose_list]
    batch_size = len(select_pose)

    cam_0s = []
    cam_1s = []
    cam_2s = []
    # batch_size = 5
    gen = G.synthesis
    for u in select_pose:
        cam_0,cam_1,cam_2,_ = gen.get_camera(batch_size=1, mode=[u, 0.5, 0.26], device=device)
        cam_2 = torch.from_numpy(np.array(cam_2)).unsqueeze(0)
        cam_0s.append(cam_0)
        cam_1s.append(cam_1)
        cam_2s.append(cam_2)
    cam_0s = torch.cat(cam_0s, 0)
    cam_1s = torch.cat(cam_1s, 0)
    cam_2s = torch.cat(cam_2s, 0)

    camera_matrics = cam_0s,cam_1s,cam_2s,None

    # 获取图像信息
    images = []
    ws_avg = G.mapping.w_avg[None, None, :]
    files = os.listdir(img_dir)
    for img in files:
        img_path = os.path.join(img_dir,img)
        image = get_test_image(img_path)
        # 用单重图像作为输入计算ws，再repea

        rec_ws,c = E(image)
        c = c.repeat(batch_size, 1, 1,1)
        rec_ws = rec_ws.repeat(batch_size, 1, 1)
        rec_ws += ws_avg
        # img_c = which_c, c, insert_layer, match, in_net
        views = source_views = camera_matrics[2]#.to(torch.float32)
        gen_images = G.get_final_output(styles=rec_ws,features=c, views=views, source_views=source_views)
        out_one = torch.cat([image,gen_images],0)
        images.append(out_one.detach())
    images = torch.cat(images,0)

    from torchvision import utils
    root_path = encoder_pkl.split('.')[1]
    root_path = root_path.split('/')
    file_name = '-'.join(root_path[2:])
    # os.makedirs(f"{store_dir}/sample", exist_ok=True)
    with torch.no_grad():
        # sample = torch.cat([images.detach(),gen_images.detach()])
        utils.save_image(
            images,
            f"{store_dir}/{file_name}.png",
            nrow=int(batch_size+1),
            normalize=True,
            range=(-1, 1),
        )


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter


# running orders
# python3 test_encoder_for_real_image.py --img_dir=./real_test_images/real_test_images3 --encoder=./output/psp_mvs_one_img/debug/checkpoints/network-snapshot-000000.pkl
# 输出目录：./output/test_encoder/$img_dir/$encoder_name
