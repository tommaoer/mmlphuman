#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import json


def build_rotation(quat):
    quat = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)
    rot = torch.empty((quat.shape[0], 3, 3), device=quat.device, dtype=quat.dtype)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)
    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - x * w)
    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    rot = build_rotation(rotations)
    rot_sorted = torch.gather(rot, dim=2, index=sorted_idx[:, None, :].repeat(1, 3, 1))
    return rot_sorted[:, :, 0]


def flip_align_view(normal, viewdir):
    dotprod = torch.sum(normal * -viewdir, dim=-1, keepdims=True)
    non_flip = dotprod >= 0
    return normal * torch.where(non_flip, 1, -1), non_flip

def safe_state(silent, seed=0):
    old_f = sys.stdout
    class F:
        def __init__(gaussians, silent):
            gaussians.silent = silent

        def write(gaussians, x):
            if not gaussians.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(gaussians):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))


def serialize_to_list(data, decimals=5):
    if isinstance(data, list):
        for i, v in enumerate(data):
            data[i] = serialize_to_list(v, decimals)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = serialize_to_list(v, decimals)
    elif isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.integer): data = np.round(data.astype(np.float64), decimals)
        data = data.tolist()
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        if not np.issubdtype(data.dtype, np.integer): data = np.round(data.astype(np.float64), decimals)
        data = data.tolist()
    return data


def storePly(out_path, xyz, rgb):
    from plyfile import PlyElement, PlyData

    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(out_path)

def fetchPly(path):
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return positions, colors, normals


def dump_poses_to_json(out_path, gaussians):
    from scene.gaussian_model import GaussianModel
    assert isinstance(gaussians, GaussianModel)

    frame_ids = sorted([int(k) for k in gaussians.all_poses])
    data_list = []
    for frame_id in frame_ids:
        data = {
            'frame_id': frame_id,
            'pose': gaussians.all_poses[str(frame_id)],
            'Th': gaussians.all_Th[str(frame_id)],
            'Rh': gaussians.all_Rh[str(frame_id)],                
        }
        data_list.append(data)

    serialize_to_list(data_list)
    with open(out_path, 'w') as file:
        json.dump(data_list, file)  
