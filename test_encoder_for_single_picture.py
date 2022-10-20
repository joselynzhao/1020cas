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
os.environ["CUDA_VISIBLE_DEVICES"]='7'
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


# ----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
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




# ----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 不需要指定任何参数，只需要修改待测试encoder的列表，主要调整seed（选择测试样列）
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default='./car_model.pkl')
@click.option('--encoder', 'encoder_pkl', help='Network pickle filename', default='./output/psp_case1/v4/checkpoints/network-snapshot-000015.pkl')
@click.option('--img_dir', help='test image path', default='./output/create_dataset/test_dataset/trunc075')
@click.option('--group_name',type=str, default='01')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, metavar='DIR',
              default='../output/test_encoders/show_psp_case2_encoder_v4')
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        encoder_pkl: str,
        img_dir: str,
        outdir: str,
        class_idx: Optional[int],
        group_name:str
):
    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    from training.networks import Generator
    from torch_utils import misc
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device)  # type: ignore
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G = copy.deepcopy(G2).eval().requires_grad_(False).to(device)

    # group_number = 8
    encoder_list = [
        # './output/psp_case2/v3/checkpoints/network-snapshot-000050.pkl',
        # './output/psp_case2/v31/checkpoints/network-snapshot-000079.pkl',
        # './output/psp_case2/v34/checkpoints/network-snapshot-000050.pkl',
        # './output/psp_case2/v34/checkpoints/network-snapshot-000079.pkl',
        # './output/psp_case2_encoder/v01/checkpoints/network-snapshot-000070.pkl',
        # './output/psp_case2_encoder/v02/checkpoints/network-snapshot-000070.pkl',
        # './output/psp_case2_encoder/v03/checkpoints/network-snapshot-000070.pkl',
        # './output/psp_case2_encoder/v04/checkpoints/network-snapshot-000070.pkl',
        # './output/psp_case2_encoder/v05/checkpoints/network-snapshot-000070.pkl',
        './output/psp_mvs_one_img/v05/checkpoints/network-snapshot-000002.pkl',
        # './output/psp_case2_encoder/v4/checkpoints/network-snapshot-000015.pkl'

    ]
    need_c = [1, 1, 1, 1,1,1,1]  # 标志是不是需要加入c_feature的模型
    which_C_list = ['p2', 'c2','c2', 'c2','c2', 'c2','c2']  # 各个模型用的是哪一个c特征。
    seed_list = [0,2,4, 8, 9, 11]  # 选的固定的几个测试图像
    # store_dir = os.path.join(f'./output/test_encoders/In_testset/Group_{group_name}')  # 用于服务器测试
    store_dir = os.path.join(f'./output/test_encoders/test_set/Group_{group_name}')
    os.makedirs(store_dir, exist_ok=True)
    # seed_list = [11]

    Encoders = []
    Generators =[]
    for idx,encoder_path in enumerate(encoder_list):
        with dnnlib.util.open_url(encoder_path) as f:
            print('Loading encoder from "%s"...' % encoder_path)
            encoder = legacy.load_network_pkl(f)
            E = encoder['E'].to(device)
            Encoders.append(E)
            if need_c[idx]:
                EG = encoder['G'].to(device)
                Generators.append(EG)
            else:
                Generators.append(G)


    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

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

    resolution=256
    pose_list = [0,2,5,8,10]
    batch_size = len(pose_list)
    for seed in seed_list:
        test_id  = f'0300{seed:02d}'
        print(f"camparing seed {test_id}")
        images =[]
        cam_0s = []
        cam_1s = []
        cam_2s = []
        for pose in pose_list:
            image_path = os.path.join(img_dir,'images',f'{test_id}_{pose:02d}.png')
            image = get_test_image(image_path)
            images.append(image)
            camera_path = os.path.join(img_dir,'cameras',f'{test_id}_{pose:02d}.npz')
            camera = np.load(camera_path)
            cam_0, cam_1, cam_2, _ = get_test_camera(camera)
            cam_0s.append(cam_0)
            cam_1s.append(cam_1)
            cam_2s.append(cam_2)
        images = torch.cat(images,0)
        cam_0s = torch.cat(cam_0s,0)
        cam_1s = torch.cat(cam_1s,0)
        cam_2s = torch.cat(cam_2s,0)
        camera_matrics = cam_0s,cam_1s,cam_2s,None
        views = source_views = camera_matrics[2][:, :2]

        ws_avg = G.mapping.w_avg[None, None, :]
        out_images = []
        for idx,E in enumerate(Encoders):
            G = Generators[idx]
            which_c = which_C_list[idx]
            is_c = need_c[idx]
            # 用单重图像作为输入计算ws，再repeat
            image_input = images[0:1, :, :, :]
            if is_c:
                rec_ws,c = E(image_input,which_c)
                c = c.repeat(batch_size, 1, 1, 1)
                rec_ws = rec_ws.repeat(batch_size, 1, 1)
                rec_ws += ws_avg
                gen_images,_ = G.get_final_output(styles=rec_ws,features=c,views = views,source_views=source_views)
            else:
                output = E(image_input)
                if isinstance(output, tuple):
                    rec_ws,_= output  # 返回ws +p的情况 stylenerf
                else:
                    rec_ws = output # 仅返回w的情况 psp case1
                rec_ws = rec_ws.repeat(batch_size,1,1)
                rec_ws += ws_avg
                gen_images = G.get_final_output(styles=rec_ws, camera_matrices=camera_matrics)
            out_images.append(gen_images.detach())
        out_images.append(images.detach())
        sample = torch.cat(out_images,0)
        from torchvision import utils
        # os.makedirs(f"{store_dir}/sample", exist_ok=True)
        with torch.no_grad():
            # sample = torch.cat([images.detach(),gen_images.detach()])
            timestamp = time.strftime('%Y%m%d.%H%M%S', time.localtime(time.time()))
            utils.save_image(
                sample,
                f"{store_dir}/{seed}-{timestamp}.png",
                nrow=int(batch_size),
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
