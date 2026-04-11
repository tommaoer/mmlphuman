import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio


def _srgb_to_linear(image):
    threshold = 0.04045
    return np.where(image <= threshold, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image):
    threshold = 0.0031308
    return np.where(image <= threshold, image * 12.92, 1.055 * np.power(np.clip(image, 0.0, None), 1.0 / 2.4) - 0.055)


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


def save_envmap_png(envmap, path):
    if torch.is_tensor(envmap):
        image = envmap.detach().float().cpu().numpy()
    else:
        image = envmap.astype(np.float32)
    image = np.clip(linear_to_srgb(image), 0.0, 1.0)
    iio.imwrite(path, (image * 255.0 + 0.5).astype(np.uint8))


def blur_envmap_tensor(envmap, kernel_size=9):
    if kernel_size <= 1:
        return envmap
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    tex = envmap.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    # Lat-long wrap on width, clamp on height.
    tex = torch.cat([tex[..., -pad:], tex, tex[..., :pad]], dim=-1)
    tex = F.pad(tex, (0, 0, pad, pad), mode='replicate')
    tex = F.avg_pool2d(tex, kernel_size=kernel_size, stride=1)
    return tex[0].permute(1, 2, 0).contiguous()


def _sh_basis_9(dirs):
    x, y, z = dirs.unbind(-1)
    return torch.stack([
        0.282095 * torch.ones_like(x),                # l=0,m=0
        0.488603 * y,                                 # l=1,m=-1
        0.488603 * z,                                 # l=1,m=0
        0.488603 * x,                                 # l=1,m=1
        1.092548 * x * y,                             # l=2,m=-2
        1.092548 * y * z,                             # l=2,m=-1
        0.315392 * (3.0 * z * z - 1.0),              # l=2,m=0
        1.092548 * x * z,                             # l=2,m=1
        0.546274 * (x * x - y * y),                  # l=2,m=2
    ], dim=-1)


def project_envmap_sh9(envmap):
    # envmap: [H,W,3] (linear radiance)
    h, w = envmap.shape[:2]
    device = envmap.device
    dtype = envmap.dtype

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij'
    )
    theta = (ys + 0.5) / h * np.pi
    phi = (xs + 0.5) / w * (2.0 * np.pi) - np.pi
    sin_t = torch.sin(theta)

    dirs = torch.stack([
        torch.sin(phi) * sin_t,   # x
        torch.cos(theta),         # y
        torch.cos(phi) * sin_t,   # z
    ], dim=-1).reshape(-1, 3)
    basis = _sh_basis_9(dirs)  # [HW,9]

    d_omega = (2.0 * np.pi / w) * (np.pi / h)
    weights = (sin_t.reshape(-1, 1) * d_omega)  # [HW,1]
    radiance = envmap.reshape(-1, 3)            # [HW,3]
    coeff = torch.einsum('nk,nc->kc', basis * weights, radiance)  # [9,3]
    return coeff


def sample_irradiance_sh9(envmap, dirs):
    coeff = project_envmap_sh9(envmap)  # [9,3]
    basis = _sh_basis_9(F.normalize(dirs, dim=-1))  # [N,9]
    # Lambertian convolution in SH domain: band factors.
    scale = torch.tensor(
        [np.pi, 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0,
         np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0],
        device=coeff.device,
        dtype=coeff.dtype,
    )[:, None]  # [9,1]
    coeff_irr = coeff * scale
    return torch.einsum('nk,kc->nc', basis, coeff_irr)
