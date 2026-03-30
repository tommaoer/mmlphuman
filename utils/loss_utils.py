

import torch
import torch.nn.functional as F
import lpips

from torch.nn.functional import l1_loss
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from scene.gaussian_model import GaussianModel

loss_fn_vgg = lpips.LPIPS(net='vgg').eval().cuda()
for p in loss_fn_vgg.parameters():
    p.requires_grad = False

def psnr(img1, img2):
    img1 = img1.permute(2,0,1)[None]
    img2 = img2.permute(2,0,1)[None]
    loss = peak_signal_noise_ratio(img1, img2, data_range=1.0)
    return loss

def ssim_loss(img1, img2, bbox=None):
    if bbox is not None:
        img1 = img1[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        img2 = img2[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    img1 = img1.permute(2,0,1)[None]
    img2 = img2.permute(2,0,1)[None]
    loss = 1.0 - structural_similarity_index_measure(img1, img2, data_range=1.0)
    return loss

def lpips_loss(img1, img2):
    img1 = img1.permute(2,0,1)[None]
    img2 = img2.permute(2,0,1)[None]
    img1 = img1 * 2.0 - 1.0
    img2 = img2 * 2.0 - 1.0
    loss = loss_fn_vgg(img1, img2).mean()
    return loss

def dxyz_smooth_loss(gaussians: GaussianModel):
    dxyz_vt = gaussians.get_dxyz_vt
    N, N_nbr = dxyz_vt.shape[0], gaussians.nbr_vt.shape[1]
    dxyz_nbrs = torch.index_select(dxyz_vt, 0, gaussians.nbr_vt.reshape(-1)).reshape(N, N_nbr, -1)
    loss = torch.linalg.vector_norm(dxyz_nbrs - dxyz_vt.unsqueeze(1), dim=-1, ord=2).mean()
    return loss

def gaussian_scaling_loss(scaling, threshold=0.01):
    scale_sub = scaling - threshold
    loss = torch.where(scale_sub > 0, scaling, torch.tensor(0, device=scaling.device)).mean()
    return loss

def normal_unit_loss(normal):
    return (torch.linalg.vector_norm(normal, dim=-1) - 1.0).abs().mean()

def normal_cosine_loss(normal_a, normal_b, mask=None):
    normal_a = F.normalize(normal_a, dim=-1)
    normal_b = F.normalize(normal_b, dim=-1)
    cos = (normal_a * normal_b).sum(dim=-1).clamp(-1.0, 1.0)
    loss_map = 1.0 - cos
    if mask is not None:
        m = mask.squeeze(-1) if mask.dim() == 3 else mask
        valid = m > 0.5
        if valid.any():
            return loss_map[valid].mean()
    return loss_map.mean()

def image_tv_loss(image, mask=None, eps=1e-6):
    if image.shape[0] < 2 or image.shape[1] < 2:
        return torch.tensor(0.0, device=image.device, dtype=image.dtype)
    dx = image[1:, :, :] - image[:-1, :, :]
    dy = image[:, 1:, :] - image[:, :-1, :]
    tv_x = torch.sqrt((dx * dx).sum(dim=-1) + eps)
    tv_y = torch.sqrt((dy * dy).sum(dim=-1) + eps)

    if mask is not None:
        m = mask.squeeze(-1) if mask.dim() == 3 else mask
        mx = (m[1:, :] > 0.5) & (m[:-1, :] > 0.5)
        my = (m[:, 1:] > 0.5) & (m[:, :-1] > 0.5)
        loss_x = tv_x[mx].mean() if mx.any() else torch.tensor(0.0, device=image.device, dtype=image.dtype)
        loss_y = tv_y[my].mean() if my.any() else torch.tensor(0.0, device=image.device, dtype=image.dtype)
        return 0.5 * (loss_x + loss_y)
    return 0.5 * (tv_x.mean() + tv_y.mean())

def depth_to_normal(depth, cam_k):
    h, w = depth.shape[:2]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing='ij'
    )
    fx = cam_k[0, 0]
    fy = cam_k[1, 1]
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]

    x = (xx - cx) / fx * depth
    y = (yy - cy) / fy * depth
    xyz = torch.stack([x, y, depth], dim=-1)

    n = torch.zeros_like(xyz)
    if h < 3 or w < 3:
        return F.normalize(xyz, dim=-1)
    dx = xyz[1:-1, 2:, :] - xyz[1:-1, :-2, :]
    dy = xyz[2:, 1:-1, :] - xyz[:-2, 1:-1, :]
    n_mid = torch.cross(dx, dy, dim=-1)
    n_mid = F.normalize(n_mid, dim=-1)
    n[1:-1, 1:-1, :] = n_mid
    n[0] = n[1]
    n[-1] = n[-2]
    n[:, 0] = n[:, 1]
    n[:, -1] = n[:, -2]
    return F.normalize(n, dim=-1)
