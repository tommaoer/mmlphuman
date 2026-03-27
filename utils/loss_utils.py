

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

def normal_cosine_loss(normal_pred, normal_gt, mask=None, eps=1e-6):
    normal_pred = F.normalize(normal_pred, dim=-1, eps=eps)
    normal_gt = F.normalize(normal_gt, dim=-1, eps=eps)
    loss = 1.0 - (normal_pred * normal_gt).sum(dim=-1).clamp(-1.0, 1.0)
    if mask is not None:
        valid = mask.bool()
        if valid.sum().item() > 0:
            loss = loss[valid]
    return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=1.0).mean()

def image_tv_loss(img, mask=None):
    dx = img[:, 1:, :] - img[:, :-1, :]
    dy = img[1:, :, :] - img[:-1, :, :]
    dx_abs = dx.abs()
    dy_abs = dy.abs()
    if mask is not None:
        mx = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).float()
        my = (mask[1:, :] & mask[:-1, :]).unsqueeze(-1).float()
        if mx.sum().item() > 0:
            loss_x = (dx_abs * mx).sum() / (mx.sum() * dx_abs.shape[-1])
        else:
            loss_x = dx_abs.mean()
        if my.sum().item() > 0:
            loss_y = (dy_abs * my).sum() / (my.sum() * dy_abs.shape[-1])
        else:
            loss_y = dy_abs.mean()
        return torch.nan_to_num(loss_x + loss_y, nan=0.0, posinf=1.0, neginf=1.0)
    return dx_abs.mean() + dy_abs.mean()

def albedo_chromaticity_loss(albedo, rgb, mask=None, eps=1e-6):
    albedo = albedo / (albedo.mean(dim=-1, keepdim=True) + eps)
    rgb = rgb / (rgb.mean(dim=-1, keepdim=True) + eps)
    diff = (albedo - rgb).abs().mean(dim=-1)
    if mask is not None:
        valid = mask.bool()
        if valid.sum().item() > 0:
            diff = diff[valid]
    return torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=1.0).mean()
