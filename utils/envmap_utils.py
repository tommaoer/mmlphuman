import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio


def _srgb_to_linear(image):
    threshold = 0.04045
    return np.where(image <= threshold, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def load_envmap_tensor(envmap_path, device='cuda'):
    if envmap_path is None:
        return None
    if not os.path.exists(envmap_path):
        raise FileNotFoundError(f'Envmap not found: {envmap_path}')

    env = iio.imread(envmap_path).astype(np.float32)
    if env.ndim == 2:
        env = np.repeat(env[..., None], 3, axis=-1)
    if env.shape[-1] > 3:
        env = env[..., :3]

    ext = os.path.splitext(envmap_path)[-1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        env = np.clip(env / 255.0, 0.0, 1.0)
        env = _srgb_to_linear(env)

    return torch.from_numpy(env).to(device=device)


def sample_latlong(envmap, dirs):
    dirs = F.normalize(dirs, dim=-1)
    x, y, z = dirs.unbind(-1)
    u = (torch.atan2(x, z) / (2 * np.pi) + 0.5) % 1.0
    v = torch.acos(torch.clamp(y, -1.0, 1.0)) / np.pi

    grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).view(1, -1, 1, 2)
    tex = envmap.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(tex, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return sampled.view(3, -1).transpose(0, 1)
