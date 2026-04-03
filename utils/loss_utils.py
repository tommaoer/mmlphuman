

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points

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

def normal_smooth_loss(gaussians: GaussianModel, K=6):
    if not gaussians.use_deferred:
        return torch.tensor(0.0, device=gaussians.get_xyz.device)

    normals = gaussians.get_world_normal()
    xyz = gaussians.get_xyz
    K = min(K + 1, xyz.shape[0])
    if K <= 1:
        return torch.tensor(0.0, device=xyz.device)

    _, idxs, _ = knn_points(xyz[None], xyz[None], K=K)
    nbr = idxs[0][:, 1:]
    nbr_normals = normals[nbr]
    return torch.linalg.vector_norm(nbr_normals - normals.unsqueeze(1), dim=-1, ord=2).mean()

def gaussian_scaling_loss(scaling, threshold=0.01):
    scale_sub = scaling - threshold
    loss = torch.where(scale_sub > 0, scaling, torch.tensor(0, device=scaling.device)).mean()
    return loss
