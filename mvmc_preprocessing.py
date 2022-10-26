
import os
import numpy as np
import torch
log_dir = '../dataset/mvmc'
out_dir ='../dataset/mvmv'
import json
import shutil


def get_world_mat(camera):
    r = torch.from_numpy(np.array(camera['R']))
    t = torch.from_numpy(np.array(camera['T']))
    t = t.squeeze(-1)
    x = torch.eye(4,4)
    x[:3,:3]=r
    x[:3,-1]=t
    return x,camera['fov']

def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan(sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)
    if invert:
        mat = torch.inverse(mat)
    return mat

def move_images(src_file,dst_file):
    try:
        os.rename(src_file,dst_file)
    except Exception as e:
        print(e)
        print("rename file failed")


def generate_dataset(dataset,name):
    os.makedirs(os.path.join(out_dir,name,'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir,name,'cameras_init'), exist_ok=True)
    os.makedirs(os.path.join(out_dir,name,'cameras_opt'), exist_ok=True)
    for idx,ID in enumerate(dataset):  # 重新编号ID
        json_path = os.path.join(log_dir,ID,"annotations.json")
        with open(json_path,'r') as f:
            info = f.read()
            json_data = json.loads(info)
            id_number = json_data['id']
            annotations = json_data['annotations']  # list
            # datas =[]
            for view_id,one in enumerate(annotations):
                camera_initial = one['camera_initial']
                world_mat_init,fov_init=get_world_mat(camera_initial)
                camera_mat_init = get_camera_mat(fov_init).squeeze()
                camera_optimizerd = one['camera_optimized']
                world_mat_opt,fov_opt = get_world_mat(camera_optimizerd)
                camera_mat_opt  = get_camera_mat(fov_opt).squeeze()

                azim=one['camera_ots']['azim']
                elev=one['camera_ots']['elev']
                roll=one['camera_ots']['roll']
                azim = (180-azim)/360 if azim <=180 else (540-azim)/360
                # azim = np.pi * azim / 180 if azim <=180 else (azim-360) * np.pi /180
                mode = np.array([azim,elev,roll])
                name_code= f'{idx:0>6d}_{view_id:02d}'
                dst_name = os.path.join(out_dir,name,"images",f'{name_code}.jpg')
                src_name = os.path.join(log_dir,ID,"images", one['filename'])
                shutil.copyfile(src_name,dst_name)
                # move_images(src_name,dst_name)
                np.savez(os.path.join(out_dir,name,'cameras_init',f'{name_code}.npz'),camera_0=camera_mat_init, camera_1=world_mat_init,camera_2=mode)
                np.savez(os.path.join(out_dir,name,'cameras_opt',f'{name_code}.npz'),camera_0=camera_mat_opt, camera_1=world_mat_opt,camera_2=mode)

IDs = os.listdir(log_dir)
print(len(IDs))
training_set = IDs[:-10]
testing_set = IDs[-10:]
print(len(training_set), len(testing_set))
generate_dataset(training_set,"training_set")
generate_dataset(testing_set,"testing_set")







    # image_path = os.path.join(log_dir,ID,'images')c