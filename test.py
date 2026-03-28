
import os
from os import path
import torch
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import numpy as np
import random
import pickle
import copy
import json
from argparse import ArgumentParser
import copy
from scipy.spatial.transform import Rotation
import imageio.v3 as iio
from torch.utils.data import DataLoader

from scene.dataset import get_dataset_type, data_to_cam
from scene.gaussian_model import GaussianModel
from scene.net_vis import load_model
from utils.config_utils import Config
from utils.image_utils import encode_bytes
from utils.smpl_utils import init_smpl_pose

def render_frame(gaussians: GaussianModel, cam, background):
    if getattr(gaussians, 'use_deferredgs', False):
        return gaussians.render_deferred(cam, background=background)
    return gaussians.render(cam, background=background)

def save_deferred_buffers(info, out_dir, frame_id):
    os.makedirs(path.join(out_dir, 'albedo'), exist_ok=True)
    os.makedirs(path.join(out_dir, 'normal'), exist_ok=True)
    os.makedirs(path.join(out_dir, 'roughness'), exist_ok=True)
    os.makedirs(path.join(out_dir, 'specular'), exist_ok=True)
    os.makedirs(path.join(out_dir, 'alpha'), exist_ok=True)

    alpha = torch.clamp(info['alpha'], 0, 1)
    confident = alpha > 0.6

    albedo = torch.clamp(info['albedo'], 0, 1)
    normal_src = info.get('normal_geom', info['normal'])
    normal = torch.clamp(normal_src * 0.5 + 0.5, 0, 1)
    albedo[~confident.squeeze(-1)] = 0.0
    normal[~confident.squeeze(-1)] = 0.5

    albedo = (albedo * 255).byte().contiguous().cpu().numpy()
    normal = (normal * 255).byte().contiguous().cpu().numpy()
    roughness = (torch.clamp(info['roughness'], 0, 1).repeat(1, 1, 3) * 255).byte().contiguous().cpu().numpy()
    specular = (torch.clamp(info['specular'], 0, 1).repeat(1, 1, 3) * 255).byte().contiguous().cpu().numpy()
    alpha = (alpha.repeat(1, 1, 3) * 255).byte().contiguous().cpu().numpy()

    iio.imwrite(path.join(out_dir, 'albedo', f'{frame_id:08d}.png'), albedo)
    iio.imwrite(path.join(out_dir, 'normal', f'{frame_id:08d}.png'), normal)
    iio.imwrite(path.join(out_dir, 'roughness', f'{frame_id:08d}.png'), roughness)
    iio.imwrite(path.join(out_dir, 'specular', f'{frame_id:08d}.png'), specular)
    iio.imwrite(path.join(out_dir, 'alpha', f'{frame_id:08d}.png'), alpha)

def fovx_to_intrinsic(fovx, H, W):
    focal = W / 2 / np.tan(fovx/2)
    K = np.zeros((3, 3))
    K[0, 0] = focal
    K[1, 1] = focal
    K[2, 2] = 1
    K[0, 2], K[1, 2] = W/2, H/2
    return K.astype(np.float32)

def load_amass_pose_list(pose_path):
    data = np.load(pose_path)
    pose_list = []
    poses = data['poses'].astype(np.float32)
    trans = data['trans'].astype(np.float32)
    N = len(poses)

    # AMASS poses are noisy
    OPTIMIZE_AMASS = False
    if OPTIMIZE_AMASS:
        foo = poses[:,3:]
        foo[:, 13 * 3 + 2] -= 0.25
        foo[:, 12 * 3 + 2] += 0.25
        foo[:, 19 * 3: 20 * 3] = 0.
        foo[:, 20 * 3: 21 * 3] = 0.
        foo[:, 14 * 3] = 0.

        poses[:,3:] = foo

        # smooth
        win_size = 1
        poses_clone = np.copy(poses)
        trans_clone = np.copy(trans)
        frame_num = poses_clone.shape[0]
        poses[win_size: frame_num-win_size] = 0
        trans[win_size: frame_num-win_size] = 0
        for i in range(-win_size, win_size + 1):
            poses[win_size: frame_num-win_size] += poses_clone[win_size+i: frame_num-win_size+i]
            trans[win_size: frame_num-win_size] += trans_clone[win_size+i: frame_num-win_size+i]
        poses[win_size: frame_num-win_size] /= (2 * win_size + 1)
        trans[win_size: frame_num-win_size] /= (2 * win_size + 1)

    for i in range(N):
        pose_list.append(dict(pose=poses[i], Th=trans[i], Rh=np.eye(3, dtype=np.float32)))

    return pose_list

def load_thuman_pose_list(pose_path):
    smpl_params = np.load(pose_path, allow_pickle=True)
    smpl_params = dict(smpl_params)

    pose_list = []
    N = len(smpl_params['global_orient'])
    for frame_id in range(N):
        pose = np.concatenate([smpl_params['global_orient'][frame_id],
                    smpl_params['body_pose'][frame_id],
                    np.zeros(3,dtype=np.float32),
                    np.zeros(6,dtype=np.float32),
                    smpl_params['left_hand_pose'][frame_id],
                    smpl_params['right_hand_pose'][frame_id],], axis=0)
        Th = smpl_params['transl'][frame_id]
        Rh = np.eye(3, dtype=np.float32)
        pose_list.append(dict(pose=pose, Th=Th, Rh=Rh))
    return pose_list

def testing_novel_cam_pose_speed(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background):

    # warm up
    pose = pose_list[0]
    gaussians.smpl_poses = torch.as_tensor(pose['pose'])
    gaussians.Th = torch.as_tensor(pose['Th'])
    gaussians.Rh = torch.as_tensor(pose['Rh'])
    image, alpha, info = render_frame(gaussians, cam, background)
    torch.cuda.synchronize()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    iter_start.record()

    for frame_id in frame_ids:
        pose = pose_list[frame_id]
        gaussians.smpl_poses = torch.as_tensor(pose['pose'])
        gaussians.Th = torch.as_tensor(pose['Th'])
        gaussians.Rh = torch.as_tensor(pose['Rh'])

        image, alpha, info = render_frame(gaussians, cam, background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous()
        torch.cuda.synchronize()

    iter_end.record()
    torch.cuda.synchronize()

    run_time = iter_start.elapsed_time(iter_end)
    fps = len(frame_ids) / run_time * 1000
    print('Running time:', run_time)
    print('FPS:', fps)

def testing_novel_cam_pose(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background, save_buffers=False):

    os.makedirs(path.join(out_dir), exist_ok=True)
    for frame_id in tqdm(frame_ids):
        pose = pose_list[frame_id]
        pose = copy.deepcopy(pose)

        gaussians.smpl_poses = torch.as_tensor(pose['pose']).cpu()
        gaussians.Th = torch.clone(torch.as_tensor(pose['Th']).cpu())
        gaussians.Rh = torch.as_tensor(pose['Rh']).cpu()
        image, alpha, info = render_frame(gaussians, cam, background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
        iio.imwrite(path.join(out_dir, f'{frame_id:08d}.png'), image)
        if save_buffers and getattr(gaussians, 'use_deferredgs', False):
            save_deferred_buffers(info, out_dir, frame_id)


def testing_dataset(gaussians: GaussianModel, out_dir, dataset, background, save_buffers=False):
    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    for k in ['gt', 'result', 'mask']:
        os.makedirs(path.join(out_dir, k), exist_ok=True)

    for cam in tqdm(test_dataloader):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']
        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

        image, alpha, info = render_frame(gaussians, cam, background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()

        image_gt = cam['image']
        image_gt[~cam['mask']] = background
        image_gt = (image_gt * 255).byte().contiguous().cpu().numpy()
        mask = cam['mask'].byte().contiguous().cpu().numpy() * 255

        iio.imwrite(path.join(out_dir, f'gt/{frame_id:08d}.png'), image_gt)
        iio.imwrite(path.join(out_dir, f'result/{frame_id:08d}.png'), image)
        iio.imwrite(path.join(out_dir, f'mask/{frame_id:08d}.png'), mask)
        if save_buffers and getattr(gaussians, 'use_deferredgs', False):
            save_deferred_buffers(info, out_dir, frame_id)


@torch.no_grad()
def testing(args: Config):
    init_smpl_pose()

    gaussians = load_model(args.model_dir)
    gaussians.is_test = args.test.is_test
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(args.background)).float().cuda()
    if args.test.envmap_path is not None:
        relight_cfg = gaussians.load_envmap_lighting(args.test.envmap_path, args.test.envmap_intensity)
        print(f'Loaded envmap relighting: {args.test.envmap_path}')
        print(json.dumps(relight_cfg, indent=2)[:1000])
    if getattr(args.test, 'save_light_envmap', False):
        envmap_path = path.join(args.out_dir, 'optimized_light_envmap.png')
        gaussians.export_deferred_envmap(envmap_path)
        print(f'Saved optimized light envmap to: {envmap_path}')

    if args.test.relight_json is not None:
        was_deferredgs = bool(getattr(gaussians, 'use_deferredgs', False))
        relight_cfg = gaussians.load_deferred_lighting(args.test.relight_json)
        print(f'Loaded relighting config: {args.test.relight_json}')
        if not was_deferredgs:
            print('Warning: checkpoint has no deferredGS flag; initialized deferred params from legacy checkpoint for relighting.')
        print(json.dumps(relight_cfg, indent=2))

    # Dataset
    test_frame_ids = np.arange(args.test.begin_ith_frame, args.test.begin_ith_frame+args.test.frame_interval*args.test.num_frame, args.test.frame_interval).tolist()
    test_cam_ids = np.array(args.test.cam_ids).tolist()

    if args.test.cam_path is not None and args.test.pose_path is not None:
        with open(args.test.cam_path, 'r') as file:
            cam = json.load(file)
        cam['w2c'] = torch.as_tensor(np.array(cam['w2c']).reshape(4,4)).float().cuda()
        K = fovx_to_intrinsic(cam['fovx'] / 180 * np.pi, cam['height'], cam['width'])
        cam['K'] = torch.as_tensor(K).cuda()

        if 'smpl_params.npz' in args.test.pose_path:
            pose_list = load_thuman_pose_list(args.test.pose_path)
        else:
            pose_list = load_amass_pose_list(args.test.pose_path)

        if args.test.test_speed:
            testing_novel_cam_pose_speed(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background)
        else:
            testing_novel_cam_pose(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background, save_buffers=args.test.save_deferred_buffers)
    else:
        DatasetType = get_dataset_type(args.data_dir)
        testset = DatasetType(
            datadir=args.data_dir,
            frame_ids=test_frame_ids,
            cam_ids=test_cam_ids,
            background=np.array(args.background),
            image_scaling=args.image_scaling,
        )

        testing_dataset(gaussians, args.out_dir, testset, background, save_buffers=args.test.save_deferred_buffers)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing")

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')

    parser.add_argument('--cam_path', type=str, default=None)
    parser.add_argument('--pose_path', type=str, default=None)
    parser.add_argument('--relight_json', type=str, default=None)
    parser.add_argument('--envmap_path', type=str, default=None)
    parser.add_argument('--envmap_intensity', type=float, default=1.0)
    parser.add_argument('--save_light_envmap', action='store_true')
    parser.add_argument('--save_deferred_buffers', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_speed', action='store_true')
    pargs = parser.parse_args(sys.argv[1:])

    args = OmegaConf.load(pargs.config)
    args.data_dir, args.out_dir, args.model_dir, args.test.cam_path, args.test.pose_path = pargs.data_dir, pargs.out_dir, pargs.model_dir, pargs.cam_path, pargs.pose_path
    args.test.relight_json = pargs.relight_json
    args.test.envmap_path = pargs.envmap_path
    args.test.envmap_intensity = pargs.envmap_intensity
    args.test.save_light_envmap = pargs.save_light_envmap
    args.test.save_deferred_buffers = pargs.save_deferred_buffers
    args.test.is_test, args.test.test_speed = pargs.test, pargs.test_speed
    torch.backends.cuda.matmul.allow_tf32 = True

    testing(args)
