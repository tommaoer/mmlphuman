

import torch
import torch.nn.functional as F

from torch.nn.functional import l1_loss
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from scene.gaussian_model import GaussianModel

lpips_model = None

def _prepare_metric_images(img1, img2):
    img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    img1 = img1.permute(2,0,1)[None]
    img2 = img2.permute(2,0,1)[None]
    return img1, img2

def psnr(img1, img2):
    img1, img2 = _prepare_metric_images(img1, img2)
    loss = peak_signal_noise_ratio(img1, img2, data_range=1.0)
    return loss

def ssim_loss(img1, img2, bbox=None):
    if bbox is not None:
        img1 = img1[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        img2 = img2[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    img1, img2 = _prepare_metric_images(img1, img2)
    loss = 1.0 - structural_similarity_index_measure(img1, img2, data_range=1.0)
    return loss

def lpips_loss(img1, img2):
    global lpips_model
    img1, img2 = _prepare_metric_images(img1, img2)
    if lpips_model is None: 
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
        for p in lpips_model.parameters(): p.requires_grad = False
    loss = lpips_model(img1, img2)
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


def total_variation_loss(image, mask=None, eps=1e-6):
    # image: [H, W, C], mask: [H, W] boolean/float (optional)
    dh = torch.abs(image[1:, :, :] - image[:-1, :, :])
    dw = torch.abs(image[:, 1:, :] - image[:, :-1, :])

    if mask is not None:
        mask = mask.float()
        mh = (mask[1:, :] * mask[:-1, :]).unsqueeze(-1)
        mw = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        dh = dh * mh
        dw = dw * mw
        denom = mh.sum() + mw.sum()
        return (dh.sum() + dw.sum()) / torch.clamp(denom, min=eps)

    denom = dh.numel() + dw.numel()
    return (dh.sum() + dw.sum()) / max(denom, 1)


def _edge_weight(guide, mask=None, sigma=10.0):
    dh = torch.mean(torch.abs(guide[1:, :, :] - guide[:-1, :, :]), dim=-1, keepdim=True)
    dw = torch.mean(torch.abs(guide[:, 1:, :] - guide[:, :-1, :]), dim=-1, keepdim=True)
    wh = torch.exp(-sigma * dh)
    ww = torch.exp(-sigma * dw)
    if mask is not None:
        mask = mask.float()
        wh = wh * (mask[1:, :] * mask[:-1, :]).unsqueeze(-1)
        ww = ww * (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
    return wh, ww


def bilateral_smooth_loss(pred, guide, mask=None, sigma=10.0, eps=1e-6):
    pdh = torch.abs(pred[1:, :, :] - pred[:-1, :, :])
    pdw = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
    wh, ww = _edge_weight(guide, mask=mask, sigma=sigma)
    loss_h = (pdh * wh).sum()
    loss_w = (pdw * ww).sum()
    denom = wh.sum() + ww.sum()
    return (loss_h + loss_w) / torch.clamp(denom, min=eps)


def base_smooth_loss(pred, mask=None, eps=1e-6):
    return total_variation_loss(pred, mask=mask, eps=eps)


def depth_to_world_normal(depth, K, w2c, mask=None):
    # depth: [H, W], K: [3,3], w2c: [4,4]
    h, w = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing='ij'
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    x = (xs - cx) * z / torch.clamp(fx, min=1e-8)
    y = (ys - cy) * z / torch.clamp(fy, min=1e-8)
    pts = torch.stack([x, y, z], dim=-1)  # [H, W, 3] in camera space

    dx = pts[:, 1:, :] - pts[:, :-1, :]
    dy = pts[1:, :, :] - pts[:-1, :, :]
    dx = F.pad(dx, (0, 0, 0, 1, 0, 0), mode='replicate')
    dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='replicate')
    n_cam = torch.cross(dx, dy, dim=-1)
    n_cam = F.normalize(n_cam, dim=-1, eps=1e-6)

    c2w = torch.linalg.inv(w2c)
    R = c2w[:3, :3]
    n_world = torch.einsum('ij,hwj->hwi', R, n_cam)
    n_world = F.normalize(n_world, dim=-1, eps=1e-6)

    valid = torch.isfinite(n_world).all(dim=-1) & (z > 1e-6)
    if mask is not None:
        valid = valid & mask.bool()
    return n_world, valid


def cosine_normal_loss(pred_normal, ref_normal, mask=None, eps=1e-6):
    # both normals in [-1,1], [H,W,3]
    pred = F.normalize(pred_normal, dim=-1, eps=eps)
    ref = F.normalize(ref_normal, dim=-1, eps=eps)
    cos = torch.sum(pred * ref, dim=-1)
    loss = 1.0 - torch.clamp(cos, -1.0, 1.0)
    if mask is not None:
        m = mask.float()
        return (loss * m).sum() / torch.clamp(m.sum(), min=eps)
    return loss.mean()
