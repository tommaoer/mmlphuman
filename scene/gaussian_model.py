
import torch
import numpy as np
from torch import nn
import os
import json

from scipy.spatial.transform import Rotation
import torch.nn.functional as F
from torch.func import vmap, functional_call, stack_module_state
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from pytorch3d.ops import knn_points
from gsplat import rasterization, quat_scale_to_covar_preci, spherical_harmonics

from scene.mlp import MLP, vmap_mlp
from utils.smpl_utils import smpl, interpolate_skinningfield, rigid_transform_tensor, rigid_transform_numba
from utils.config_utils import Config
from utils.sh_utils import RGB2SH, SH2RGB

class GaussianModel:
    @staticmethod
    def _estimate_point_normals_from_xyz(xyz, k=16):
        with torch.no_grad():
            _, idxs, _ = knn_points(xyz[None], xyz[None], K=k+1)
            nbr = xyz[idxs[0][:, 1:]]  # [N, k, 3]
            mean = nbr.mean(dim=1, keepdim=True)
            centered = nbr - mean
            cov = torch.einsum('nki,nkj->nij', centered, centered) / max(1, k - 1)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            normal = eigvecs[:, :, 0]
            outward = xyz - xyz.mean(dim=0, keepdim=True)
            flip = torch.sign((normal * outward).sum(dim=-1, keepdim=True)).clamp_min(0.0) * 2.0 - 1.0
            normal = F.normalize(normal * flip, dim=-1)
        return normal


    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = F.normalize

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = torch.logit

    def __init__(self):

        self._xyz = torch.empty(0)
        self.xyz_offset = torch.empty(0)
        self.dxyz_vt = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._sh0 = torch.empty(0)
        self._shN = torch.empty(0)
        self._albedo = torch.empty(0)
        self._normal = torch.empty(0)
        self._roughness = torch.empty(0)
        self._specular = torch.empty(0)
        self.sh_degree = 0

        self.xyz_vt = torch.empty(0)
        self.xyz_ft = torch.empty(0)

        # basis property definition
        self.num_vt_basis = 15     # Control point basis number
        self.num_basis = 15        # Gaussian property basis number

        self.encoder_feat_params = None
        self.encoder_feat_model_meta = None

        self.dxyz_bs = torch.empty(0)
        self.sh0_bs = torch.empty(0)
        self.shN_bs = torch.empty(0)
        self.scaling_bs = torch.empty(0)
        self.rotation_bs = torch.empty(0)
        self.opacity_bs = torch.empty(0)
        self.albedo_bs = torch.empty(0)
        self.normal_bs = torch.empty(0)
        self.roughness_bs = torch.empty(0)
        self.specular_bs = torch.empty(0)

        self.use_deferredgs = False
        self.deferred_light_sh = torch.empty(0)
        self.deferred_light_dc = torch.empty(0)
        self.use_direct_envmap = False
        self.envmap_diffuse_mode = 'direct'
        self.deferred_envmap_input = torch.empty(0)
        self.deferred_envmap = torch.empty(0)

        # lbs weights
        self._weights = None

        # pose
        self._Rh = torch.empty(0)
        self._Th = torch.empty(0)
        self.Ac_inv = torch.empty(0)
        self._smpl_poses = torch.empty(0)
        self.smpl_poses_cuda = torch.empty(0)
        self.t_joints = torch.empty(0)
        self.joint_parents = torch.empty(0)

        self.all_poses = torch.empty(0)

        # cache
        self.cache_dict = {}

        # optimizer
        self.optimizers = None
        self.schedulers = None

        # knn
        self.nbr_gs = torch.empty(0)
        self.nbr_gs_invdist = torch.empty(0)
        self.nbr_vt = torch.empty(0)
        self.nbr_gsft = torch.empty(0)
        self.nbr_vtft = torch.empty(0)
        self.nbr_gsft_wght = torch.empty(0)
        self.nbr_vtft_wght = torch.empty(0)

        # misc
        self.scene_scale = None
        self.is_dxyz_bs = False     # whether to use control point basis
        self.is_gsparam_bs = False  # whether to use Gaussian property basis

        self.is_test = False        # whether to use PCA 

        self.setup_functions()

    def capture(self):
        data = {
            '_xyz': self._xyz,
            'xyz_offset': self.xyz_offset,
            'dxyz_vt': self.dxyz_vt,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            '_opacity': self._opacity,
            '_sh0': self._sh0,
            '_shN': self._shN,
            '_albedo': self._albedo,
            '_normal': self._normal,
            '_roughness': self._roughness,
            '_specular': self._specular,
            'sh_degree': self.sh_degree,

            '_weights': self.get_weights,

            't_joints': self.t_joints,
            'all_poses': self.all_poses,
            'joint_parents': self.joint_parents,

            'nbr_gs_invdist': self.nbr_gs_invdist,
            'nbr_gs': self.nbr_gs,
            'nbr_vt': self.nbr_vt,
            'nbr_gsft': self.nbr_gsft,
            'nbr_vtft': self.nbr_vtft,
            'nbr_gsft_wght': self.nbr_gsft_wght,
            'nbr_vtft_wght': self.nbr_vtft_wght,

            'xyz_vt': self.xyz_vt,
            'xyz_ft': self.xyz_ft,

            'num_vt_basis': self.num_vt_basis,
            'num_basis': self.num_basis,

            'encoder_feat_params': self.encoder_feat_params,
            'encoder_feat_model_meta': self.encoder_feat_model_meta,

            'dxyz_bs': self.dxyz_bs,
            'sh0_bs': self.sh0_bs,
            'shN_bs': self.shN_bs,
            'scaling_bs': self.scaling_bs,
            'rotation_bs': self.rotation_bs,
            'opacity_bs': self.opacity_bs,
            'albedo_bs': self.albedo_bs,
            'normal_bs': self.normal_bs,
            'roughness_bs': self.roughness_bs,
            'specular_bs': self.specular_bs,

            'is_dxyz_bs': self.is_dxyz_bs,
            'is_gsparam_bs': self.is_gsparam_bs,
            'use_deferredgs': self.use_deferredgs,
            'deferred_light_sh': self.deferred_light_sh,
            'deferred_light_dc': self.deferred_light_dc,
        }
        return data
    
    def restore(self, data):
        def loader(s):
            if s in data: return data[s]
            else: print(f'NO DATA {s}!')
            return None

        self._xyz = data['_xyz']
        self.xyz_offset = data['xyz_offset']
        self.dxyz_vt = data['dxyz_vt']
        self._opacity = data['_opacity']
        self._rotation = data['_rotation']
        self._scaling = data['_scaling']
        self._sh0 = data['_sh0']
        self._shN = loader('_shN')
        self._albedo = loader('_albedo')
        self._normal = loader('_normal')
        self._roughness = loader('_roughness')
        self._specular = loader('_specular')
        self.sh_degree = data['sh_degree']

        self._weights = data['_weights']

        self.t_joints = loader('t_joints')
        self.all_poses = loader('all_poses')
        self.joint_parents = loader('joint_parents')

        self.nbr_gs = loader('nbr_gs')
        self.nbr_vt = loader('nbr_vt')
        self.nbr_gs_invdist = loader('nbr_gs_invdist')
        self.nbr_gsft = loader('nbr_gsft')
        self.nbr_vtft = loader('nbr_vtft')
        self.nbr_gsft_wght = loader('nbr_gsft_wght')
        self.nbr_vtft_wght = loader('nbr_vtft_wght')

        self.xyz_vt = loader('xyz_vt')
        self.xyz_ft = loader('xyz_ft')

        self.num_vt_basis = loader('num_vt_basis')
        self.num_basis = loader('num_basis')

        self.encoder_feat_params = loader('encoder_feat_params')
        self.encoder_feat_model_meta = loader('encoder_feat_model_meta')

        self.dxyz_bs = loader('dxyz_bs')
        self.sh0_bs = loader('sh0_bs')
        self.shN_bs = loader('shN_bs')
        self.scaling_bs = loader('scaling_bs')
        self.rotation_bs = loader('rotation_bs') 
        self.opacity_bs = loader('opacity_bs')
        self.albedo_bs = loader('albedo_bs')
        self.normal_bs = loader('normal_bs')
        self.roughness_bs = loader('roughness_bs')
        self.specular_bs = loader('specular_bs')

        self.is_dxyz_bs = loader('is_dxyz_bs')
        self.is_gsparam_bs = loader('is_gsparam_bs')
        self.use_deferredgs = loader('use_deferredgs')
        self.deferred_light_sh = loader('deferred_light_sh')
        self.deferred_light_dc = loader('deferred_light_dc')

        if self.use_deferredgs is None:
            self.use_deferredgs = False

        self._ensure_deferred_params()

        self.init()

    def init(self):
        self.init_body() 
        self.reset_pose()   

    def _ensure_deferred_params(self):
        if (not torch.is_tensor(self._xyz)) or self._xyz.numel() == 0:
            return
        device = self._xyz.device
        N = self._xyz.shape[0]
        is_legacy_ckpt = not bool(self.use_deferredgs)

        if is_legacy_ckpt or (not torch.is_tensor(self._albedo)) or self._albedo is None or self._albedo.numel() == 0:
            if torch.is_tensor(self._sh0) and self._sh0.numel() > 0:
                if torch.is_tensor(self._shN) and self._shN is not None and self._shN.numel() > 0 and self.sh_degree > 0:
                    sh = torch.cat([self._sh0, self._shN], dim=1)
                    sample_dirs = torch.tensor([
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0],
                    ], dtype=sh.dtype, device=sh.device)
                    sh_eval = []
                    for d in sample_dirs:
                        dirs = d[None].repeat(N, 1)
                        sh_eval.append(spherical_harmonics(self.sh_degree, dirs, sh) + 0.5)
                    sh_eval = torch.stack(sh_eval, dim=0)
                    init_albedo_rgb = torch.clamp(torch.quantile(sh_eval, 0.75, dim=0), 1e-4, 1.0 - 1e-4)
                else:
                    init_albedo_rgb = torch.clamp(SH2RGB(self._sh0[:, 0]), 1e-4, 1.0 - 1e-4)
            else:
                init_albedo_rgb = torch.full((N, 3), 0.5, device=device)
            init_albedo = self.inverse_color_activation(init_albedo_rgb)
            self._albedo = nn.Parameter(init_albedo.requires_grad_(True))
        if is_legacy_ckpt or (not torch.is_tensor(self._normal)) or self._normal is None or self._normal.numel() == 0:
            normal = GaussianModel._estimate_point_normals_from_xyz(self._xyz)
            self._normal = nn.Parameter(normal.requires_grad_(True))
        if (not torch.is_tensor(self._roughness)) or self._roughness is None or self._roughness.numel() == 0:
            roughness = torch.full((N,), self.inverse_opacity_activation(torch.tensor(1.0, device=device)), device=device)
            self._roughness = nn.Parameter(roughness.requires_grad_(True))
        if (not torch.is_tensor(self._specular)) or self._specular is None or self._specular.numel() == 0:
            specular = torch.full((N,), self.inverse_opacity_activation(torch.tensor(0.0, device=device)), device=device)
            self._specular = nn.Parameter(specular.requires_grad_(True))

        if (not torch.is_tensor(self.albedo_bs)) or self.albedo_bs is None or self.albedo_bs.numel() == 0:
            self.albedo_bs = nn.Parameter(torch.zeros((N, self.num_basis, 1, 3), dtype=torch.float32, device=device).requires_grad_(True))
        if (not torch.is_tensor(self.normal_bs)) or self.normal_bs is None or self.normal_bs.numel() == 0:
            self.normal_bs = nn.Parameter(torch.zeros((N, self.num_basis, 3), dtype=torch.float32, device=device).requires_grad_(True))
        if (not torch.is_tensor(self.roughness_bs)) or self.roughness_bs is None or self.roughness_bs.numel() == 0:
            self.roughness_bs = nn.Parameter(torch.zeros((N, self.num_basis), dtype=torch.float32, device=device).requires_grad_(True))
        if (not torch.is_tensor(self.specular_bs)) or self.specular_bs is None or self.specular_bs.numel() == 0:
            self.specular_bs = nn.Parameter(torch.zeros((N, self.num_basis), dtype=torch.float32, device=device).requires_grad_(True))

        if (not torch.is_tensor(self.deferred_light_sh)) or self.deferred_light_sh is None or self.deferred_light_sh.numel() == 0:
            self.deferred_light_sh = nn.Parameter(torch.zeros((9, 3), dtype=torch.float32, device=device).requires_grad_(True))
        if (not torch.is_tensor(self.deferred_light_dc)) or self.deferred_light_dc is None or self.deferred_light_dc.numel() == 0:
            self.deferred_light_dc = nn.Parameter(torch.full((3,), 0.5, dtype=torch.float32, device=device).requires_grad_(True))

    @property
    def get_cano_scaling(self):
        if 'get_cano_scaling' in self.cache_dict: return self.cache_dict['get_cano_scaling'] 
        if not self.is_gsparam_bs: 
            scaling = self.scaling_activation(self._scaling)
        else:
            features = self.get_encoded_feature_gsparam_weight
            dscaling = torch.einsum('nc,ncl->nl', features, self.scaling_bs)

            scaling = self._scaling + dscaling
            scaling = self.scaling_activation(scaling)
        
        self.cache_dict['get_cano_scaling'] = scaling
        return scaling
    
    @property
    def get_weights(self):
        if self._weights is None:
            xyz = self._xyz
            weights = interpolate_skinningfield(self.weights_grid_info, xyz)
            self._weights = weights
        else:
            weights = self._weights
        return weights

    @property
    def get_rigid_transform(self):
        if 'get_rigid_transform' in self.cache_dict: return self.cache_dict['get_rigid_transform']
        pose = self.smpl_poses.numpy()
        joints = self.t_joints.numpy()
        parent = self.joint_parents.numpy()
        Ac_inv = self.Ac_inv.numpy()

        rots = Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().astype(np.float32)
        A = rigid_transform_numba(rots, joints, parent)
        G = np.matmul(A, Ac_inv)

        data = [torch.as_tensor(d).cuda(non_blocking=True) for d in [rots, G]]
        self.cache_dict['get_rigid_transform'] = data
        return data

    @property
    def get_Gweights(self):
        if 'get_Gweights' in self.cache_dict: return self.cache_dict['get_Gweights']

        # Rots = batch_rodrigues(self.smpl_poses.reshape(-1,3))
        # A = batch_rigid_transform(Rots[None], self.t_joints[None], self.joint_parents)[1][0]
        # G = torch.matmul(A, self.Ac_inv)
        
        G = self.get_rigid_transform[1]
        G_weight = torch.einsum('vp,pij->vij', self.get_weights, G)

        self.cache_dict['get_Gweights'] = G_weight
        return G_weight

    @property
    def get_cano_rotation(self):
        if not self.is_gsparam_bs: 
            rotation = self.rotation_activation(self._rotation)
        else:
            features = self.get_encoded_feature_gsparam_weight
            drotation = torch.einsum('nc,ncl->nl', features, self.rotation_bs)

            rotation = self._rotation + drotation
            rotation = self.rotation_activation(rotation)

        return rotation

    @property
    def get_cano_albedo(self):
        if 'get_cano_albedo' in self.cache_dict: return self.cache_dict['get_cano_albedo']
        albedo = self.color_activation(self._albedo)
        if self.is_gsparam_bs and self.albedo_bs.numel() > 0:
            features = self.get_encoded_feature_gsparam_weight
            dalbedo = torch.einsum('nc,ncxy->nxy', features, self.albedo_bs).squeeze(1)
            albedo = self.color_activation(self._albedo + dalbedo)
        self.cache_dict['get_cano_albedo'] = albedo
        return albedo

    @property
    def get_cano_normal(self):
        if 'get_cano_normal' in self.cache_dict: return self.cache_dict['get_cano_normal']
        normal = self._normal
        if self.is_gsparam_bs and self.normal_bs.numel() > 0:
            features = self.get_encoded_feature_gsparam_weight
            dnormal = torch.einsum('nc,ncl->nl', features, self.normal_bs)
            normal = normal + dnormal
        normal = F.normalize(normal, dim=-1)
        self.cache_dict['get_cano_normal'] = normal
        return normal

    @property
    def get_roughness(self):
        if 'get_roughness' in self.cache_dict: return self.cache_dict['get_roughness']
        roughness = self.opacity_activation(self._roughness)
        if self.is_gsparam_bs and self.roughness_bs.numel() > 0:
            features = self.get_encoded_feature_gsparam_weight
            droughness = torch.einsum('nc,nc->n', features, self.roughness_bs)
            roughness = self.opacity_activation(self._roughness + droughness)
        self.cache_dict['get_roughness'] = roughness
        return roughness

    @property
    def get_specular(self):
        if 'get_specular' in self.cache_dict: return self.cache_dict['get_specular']
        specular = self.opacity_activation(self._specular)
        if self.is_gsparam_bs and self.specular_bs.numel() > 0:
            features = self.get_encoded_feature_gsparam_weight
            dspecular = torch.einsum('nc,nc->n', features, self.specular_bs)
            specular = self.opacity_activation(self._specular + dspecular)
        self.cache_dict['get_specular'] = specular
        return specular

    def get_covariance(self, scaling_modifier=1):
        rots = self.get_Gweights[:,:3,:3].contiguous()
        covs = quat_scale_to_covar_preci(
            quats=self.get_cano_rotation,
            scales=self.get_cano_scaling * scaling_modifier,
            compute_preci=False,
        )[0]

        if self.Rh is not None: rots = self.Rh @ rots
        covs = rots @ covs @ rots.transpose(-1,-2)
        return covs

    @property
    def get_joint_features(self):

        if self.is_test:
            sigma_pca = 2.0
            features = self.smpl_poses_cuda[1*3:22*3][None]
            lowdim_pose_conds = self.pca.transform(features)
            std = self.pca_std
            lowdim_pose_conds = torch.maximum(lowdim_pose_conds, -sigma_pca * std)
            lowdim_pose_conds = torch.minimum(lowdim_pose_conds, sigma_pca * std)
            features = self.pca.inverse_transform(lowdim_pose_conds).reshape(-1)
        else:
            features = self.smpl_poses_cuda[3:3*22]

        return features

    @torch.no_grad()
    def prepare_test(self):
        pose_set = []
        for k, v in self.all_poses.items():
            pose_set.append(v[1*3:22*3].detach())     
        N_pose = len(pose_set)
        pose_set = torch.stack(pose_set, dim=0).reshape(N_pose,21,3).cpu().numpy()
        features = pose_set.reshape(N_pose, -1)

        pca_num = 20

        features = torch.as_tensor(features).cuda()
        from torch_pca import PCA
        self.pca = PCA(n_components=pca_num)
        self.pca.fit(features)
        self.pca_std = torch.sqrt(self.pca.explained_variance_)

        print(f'Use PCA components: {pca_num}')

    @property
    def get_encoded_feature(self):
        if 'get_encoded_feature' in self.cache_dict: return self.cache_dict['get_encoded_feature']
        features = self.get_joint_features
        N_feat = len(self.encoder_feat_params['layers.0.weight'])
        features = features.tile([N_feat, 1])
        features = vmap_mlp(self.encoder_feat_params, features)

        self.cache_dict['get_encoded_feature'] = features
        return features

    @property
    def get_encoded_feature_gsparam_weight(self):
        if 'get_encoded_feature_gsparam_weight' in self.cache_dict: return self.cache_dict['get_encoded_feature_gsparam_weight']
        features = self.get_encoded_feature[...,:self.num_basis]
        features = torch.einsum('nrc,nr->nc', features[self.nbr_gsft], self.nbr_gsft_wght)

        self.cache_dict['get_encoded_feature_gsparam_weight'] = features
        return features

    @property
    def get_dxyz_vt(self):
        if 'get_dxyz_vt' in self.cache_dict: return self.cache_dict['get_dxyz_vt']
        if not self.is_dxyz_bs: return self.dxyz_vt

        features = self.get_encoded_feature[...,self.num_basis:]
        features = torch.einsum('nrc,nr->nc', features[self.nbr_vtft], self.nbr_vtft_wght)

        dxyz_vt = torch.einsum('vc,vcl->vl', features, self.dxyz_bs)

        dxyz_vt = self.dxyz_vt + dxyz_vt
        self.cache_dict['get_dxyz_vt'] = dxyz_vt

        return dxyz_vt

    @property
    def get_dxyz(self):
        if 'get_dxyz' in self.cache_dict: return self.cache_dict['get_dxyz']

        dxyz = torch.sum(self.nbr_gs_invdist[...,None] * self.get_dxyz_vt[self.nbr_gs], dim=1) / torch.sum(self.nbr_gs_invdist, dim=-1)[...,None]
        self.cache_dict['get_dxyz'] = dxyz
        return dxyz
    
    @property
    def get_cano_xyz(self):
        if 'get_cano_xyz' in self.cache_dict: return self.cache_dict['get_cano_xyz']
        xyz = self._xyz + self.get_dxyz + torch.tanh(self.xyz_offset) * 0.008   # A trick to allow Gaussians to move freely within a small range
        self.cache_dict['get_cano_xyz'] = xyz
        return xyz

    @property
    def get_xyz(self):
        if 'get_xyz' in self.cache_dict: return self.cache_dict['get_xyz']
        xyz = self.get_cano_xyz
        xyz = torch.einsum('vij,vj->vi', self.get_Gweights, F.pad(xyz,(0,1),value=1))[:,:3]
        if self.Rh is not None: xyz = torch.einsum('ij,vj->vi', self.Rh, xyz) 
        xyz = xyz + self.Th

        self.cache_dict['get_xyz'] = xyz
        return xyz

    @property
    def get_opacity(self):
        if not self.is_gsparam_bs:
            opacity = self.opacity_activation(self._opacity)     
        else:
            features = self.get_encoded_feature_gsparam_weight
            dopacity = torch.einsum('nc,nc->n', features, self.opacity_bs)

            opacity = self._opacity + dopacity
            opacity = self.opacity_activation(opacity)

        return opacity

    @property
    def get_sh(self):
        if 'get_sh' in self.cache_dict: return self.cache_dict['get_sh']

        if self.sh_degree == 0: 
            sh = self._sh0
        else:
            sh = torch.cat([self._sh0, self._shN], dim=1)

        if self.is_gsparam_bs:

            features = self.get_encoded_feature_gsparam_weight
            dsh0 = torch.einsum('nc,ncxy->nxy', features, self.sh0_bs)
            if self.sh_degree == 0: 
                dsh = dsh0
            else: 
                dshN = torch.einsum('nc,ncxy->nxy', features, self.shN_bs)
                dsh = torch.cat([dsh0, dshN], dim=1)

            sh = sh + dsh

        self.cache_dict['get_sh'] = sh
        return sh

    def get_color(self, cam_pos):
        if 'get_color' in self.cache_dict: return self.cache_dict['get_color']

        if self.sh_degree > 0:
            rots = self.get_Gweights[:,:3,:3]
            # with torch.set_grad_enabled(False):
            #     rots = polar_decomposition_newton_schulz(rots)

            dirs = F.normalize(cam_pos - self.get_xyz, dim=-1)
            invrots = rots.transpose(-1,-2)
            dirs = torch.einsum('nij,nj->ni',invrots, dirs)
        else:
            dirs = torch.ones_like(self._xyz)

        sh = self.get_sh
        color = spherical_harmonics(self.sh_degree, dirs, sh)
        color = torch.clamp_min(color + 0.5, 0)

        self.cache_dict['get_color'] = color

        return color

    @property
    def get_target_normal(self):
        if 'get_target_normal' in self.cache_dict: return self.cache_dict['get_target_normal']
        rots = self.get_Gweights[:,:3,:3]
        normal = torch.einsum('nij,nj->ni', rots, self.get_cano_normal)
        if self.Rh is not None:
            normal = torch.einsum('ij,nj->ni', self.Rh, normal)
        normal = F.normalize(normal, dim=-1)
        self.cache_dict['get_target_normal'] = normal
        return normal

    def _render_feature(self, cam, colors, background=None, covars=None):
        channels = colors.shape[-1]
        if background is None:
            background = torch.zeros(channels, device=colors.device, dtype=colors.dtype)
        image, alpha, _ = rasterization(
            means=self.get_xyz,
            quats=None,
            scales=None,
            opacities=self.get_opacity,
            colors=colors,
            viewmats=cam['w2c'][None],
            Ks=cam['K'][None],
            width=cam['width'],
            height=cam['height'],
            packed=False,
            near_plane=0.1,
            backgrounds=background[None],
            covars=covars,
        )
        return image[0], alpha[0]

    def _eval_sh9(self, normals):
        x, y, z = normals.unbind(dim=-1)
        basis = torch.stack([
            torch.ones_like(x),
            y,
            z,
            x,
            x * y,
            y * z,
            3.0 * z * z - 1.0,
            x * z,
            x * x - y * y,
        ], dim=-1)
        return basis

    def _compute_normal_from_xyz_map(self, xyz_map, alpha):
        h, w = xyz_map.shape[:2]
        n = torch.zeros_like(xyz_map)
        if h < 3 or w < 3:
            return F.normalize(xyz_map, dim=-1)

        dx = xyz_map[1:-1, 2:, :] - xyz_map[1:-1, :-2, :]
        dy = xyz_map[2:, 1:-1, :] - xyz_map[:-2, 1:-1, :]
        n_mid = torch.cross(dx, dy, dim=-1)
        n_mid = F.normalize(n_mid, dim=-1)
        n[1:-1, 1:-1, :] = n_mid
        n[0] = n[1]
        n[-1] = n[-2]
        n[:, 0] = n[:, 1]
        n[:, -1] = n[:, -2]

        n = F.normalize(n, dim=-1)
        n = torch.where(alpha > 1e-3, n, torch.zeros_like(n))
        return n

    def render_deferred(self, cam, background=None, scaling_modifier=1.0):
        covars = self.get_covariance(scaling_modifier)
        zeros3 = torch.zeros(3, device=self.get_xyz.device, dtype=self.get_xyz.dtype)
        zeros1 = torch.zeros(1, device=self.get_xyz.device, dtype=self.get_xyz.dtype)

        albedo, alpha = self._render_feature(cam, self.get_cano_albedo, zeros3, covars)
        normal, _ = self._render_feature(cam, self.get_target_normal, zeros3, covars)
        xyz_map, _ = self._render_feature(cam, self.get_xyz, zeros3, covars)
        roughness, _ = self._render_feature(cam, self.get_roughness[:, None], zeros1, covars)
        specular, _ = self._render_feature(cam, self.get_specular[:, None], zeros1, covars)

        denom = alpha.clamp_min(1e-6)
        normal = F.normalize(normal / denom, dim=-1)
        xyz_map = xyz_map / denom
        normal_geom = self._compute_normal_from_xyz_map(xyz_map, alpha)
        xyz_map_h = torch.cat([xyz_map, torch.ones_like(xyz_map[..., :1])], dim=-1)
        depth = torch.einsum('ij,hwj->hwi', cam['w2c'], xyz_map_h)[..., 2:3]
        albedo = torch.clamp(albedo / denom, 0.0, 1.0)
        roughness = torch.clamp(roughness / denom, 0.0, 1.0)
        specular = torch.clamp(specular / denom, 0.0, 1.0)

        use_direct = (
            bool(getattr(self, 'use_direct_envmap', False))
            and torch.is_tensor(self.deferred_envmap)
            and self.deferred_envmap.numel() > 0
        )
        diffuse_mode = str(getattr(self, 'envmap_diffuse_mode', 'direct')).lower()
        if diffuse_mode not in ('direct', 'sh'):
            diffuse_mode = 'direct'

        if use_direct and diffuse_mode == 'direct':
            diffuse_light = self._sample_envmap(normal)
        else:
            sh_basis = self._eval_sh9(normal)
            diffuse_light = torch.einsum('hwc,ck->hwk', sh_basis, self.deferred_light_sh) + self.deferred_light_dc
            diffuse_light = torch.clamp_min(diffuse_light, 0.0)

        cam_pos = torch.linalg.inv_ex(cam['w2c'])[0][:3, 3]
        view_dir = F.normalize(cam_pos[None, None] - xyz_map, dim=-1)
        half_vec = F.normalize(view_dir + torch.tensor([0.0, 0.0, 1.0], device=view_dir.device), dim=-1)
        spec_pow = 4.0 + (1.0 - roughness) * 60.0
        spec_term = torch.clamp((normal * half_vec).sum(dim=-1, keepdim=True), 0.0, 1.0) ** spec_pow
        shaded = albedo * diffuse_light + specular * spec_term
        if background is not None:
            shaded = shaded * alpha + background[None, None] * (1.0 - alpha)

        info = {
            'albedo': albedo,
            'normal': normal,
            'normal_geom': normal_geom,
            'depth': depth,
            'roughness': roughness,
            'specular': specular,
            'alpha': alpha,
        }
        return torch.clamp(shaded, 0.0, 1.0), alpha, info

    def _sample_envmap(self, normals):
        env = self.deferred_envmap
        if env.dim() != 3:
            return torch.zeros_like(normals)
        n = F.normalize(normals, dim=-1)
        x, y, z = n.unbind(dim=-1)
        theta = torch.arccos(torch.clamp(z, -1.0, 1.0))
        phi = torch.atan2(y, x)
        u = torch.remainder(phi / (2.0 * np.pi), 1.0)
        v = theta / np.pi
        grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1)[None]
        tex = env.permute(2, 0, 1)[None]
        sampled = F.grid_sample(tex, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return sampled[0].permute(1, 2, 0)

    @torch.no_grad()
    def set_deferred_lighting(self, light_sh=None, light_dc=None):
        if light_sh is not None:
            light_sh = torch.as_tensor(light_sh, dtype=self.deferred_light_sh.dtype, device=self.deferred_light_sh.device)
            self.deferred_light_sh.copy_(light_sh.reshape_as(self.deferred_light_sh))
        if light_dc is not None:
            light_dc = torch.as_tensor(light_dc, dtype=self.deferred_light_dc.dtype, device=self.deferred_light_dc.device)
            self.deferred_light_dc.copy_(light_dc.reshape_as(self.deferred_light_dc))
        self.use_direct_envmap = False
        self.envmap_diffuse_mode = 'sh'
        self.deferred_envmap_input = torch.empty(0, device=self.deferred_light_sh.device, dtype=self.deferred_light_sh.dtype)
        self.deferred_envmap = torch.empty(0, device=self.deferred_light_sh.device, dtype=self.deferred_light_sh.dtype)
        self.cache_dict = {}

    @torch.no_grad()
    def load_deferred_lighting(self, json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
        self._ensure_deferred_params()
        self.use_deferredgs = True
        self.set_deferred_lighting(
            light_sh=data.get('light_sh', None),
            light_dc=data.get('light_dc', None),
        )
        self.use_direct_envmap = False
        return data

    @torch.no_grad()
    def load_envmap_lighting(self, envmap_path, intensity=1.0, auto_rescale=True, target_avg=0.5, diffuse_mode='direct'):
        import imageio.v3 as iio

        env_raw = iio.imread(envmap_path)
        if np.issubdtype(env_raw.dtype, np.integer):
            # LDR integer formats (png/jpg/...) are normalized to [0,1].
            maxv = float(np.iinfo(env_raw.dtype).max)
            env = env_raw.astype(np.float32) / max(maxv, 1.0)
        else:
            # HDR float formats (hdr/exr/...) are already linear radiance; keep absolute scale.
            env = env_raw.astype(np.float32)
        env = np.clip(env[..., :3], 0.0, None)
        env_input = env.copy()
        applied_rescale = 1.0
        if auto_rescale:
            lum = 0.2126 * env[..., 0] + 0.7152 * env[..., 1] + 0.0722 * env[..., 2]
            lum_mean = float(lum.mean())
            if lum_mean > 1e-6:
                applied_rescale = float(target_avg) / lum_mean
                env = env * applied_rescale
        env = env * float(intensity)
        H, W = env.shape[:2]

        theta = (np.arange(H, dtype=np.float32) + 0.5) / H * np.pi
        phi = (np.arange(W, dtype=np.float32) + 0.5) / W * (2.0 * np.pi)
        theta, phi = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        basis = np.stack([
            np.ones_like(x), y, z, x, x * y, y * z, 3.0 * z * z - 1.0, x * z, x * x - y * y
        ], axis=-1).reshape(-1, 9)
        rgb = env.reshape(-1, 3)
        weights = np.sin(theta).reshape(-1, 1)

        bw = basis * weights
        lhs = bw.T @ basis + np.eye(9, dtype=np.float32) * 1e-6
        rhs = bw.T @ rgb
        coeff = np.linalg.solve(lhs, rhs).astype(np.float32)

        self._ensure_deferred_params()
        self.use_deferredgs = True
        self.set_deferred_lighting(light_sh=coeff, light_dc=np.zeros(3, dtype=np.float32))
        self.envmap_diffuse_mode = str(diffuse_mode).lower()
        self.use_direct_envmap = self.envmap_diffuse_mode == 'direct'
        self.deferred_envmap_input = torch.as_tensor(env_input, dtype=self.deferred_light_sh.dtype, device=self.deferred_light_sh.device)
        self.deferred_envmap = torch.as_tensor(env, dtype=self.deferred_light_sh.dtype, device=self.deferred_light_sh.device)
        return dict(
            light_sh=coeff.tolist(),
            light_dc=[0.0, 0.0, 0.0],
            envmap_path=envmap_path,
            intensity=float(intensity),
            auto_rescale=bool(auto_rescale),
            rescale_factor=float(applied_rescale),
            target_avg=float(target_avg),
            diffuse_mode=self.envmap_diffuse_mode,
            input_envmap_mean=float(env_input.mean()),
            output_envmap_mean=float(env.mean()),
        )

    @torch.no_grad()
    def export_deferred_envmap(self, output_path, height=256, width=512):
        if bool(getattr(self, 'use_direct_envmap', False)) and torch.is_tensor(self.deferred_envmap) and self.deferred_envmap.numel() > 0:
            env = self.deferred_envmap.detach().cpu().numpy()
        else:
            theta = (np.arange(height, dtype=np.float32) + 0.5) / height * np.pi
            phi = (np.arange(width, dtype=np.float32) + 0.5) / width * (2.0 * np.pi)
            theta, phi = np.meshgrid(theta, phi, indexing='ij')
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            basis = np.stack([
                np.ones_like(x), y, z, x, x * y, y * z, 3.0 * z * z - 1.0, x * z, x * x - y * y
            ], axis=-1)
            light_sh = self.deferred_light_sh.detach().cpu().numpy()
            light_dc = self.deferred_light_dc.detach().cpu().numpy()
            env = np.einsum('hwc,ck->hwk', basis, light_sh) + light_dc[None, None]
            env = np.clip(env, 0.0, None)

        self._write_envmap_preview(output_path, env)
        return env

    @torch.no_grad()
    def export_input_envmap(self, output_path):
        if (not torch.is_tensor(self.deferred_envmap_input)) or self.deferred_envmap_input.numel() == 0:
            return None
        env = self.deferred_envmap_input.detach().cpu().numpy()
        self._write_envmap_preview(output_path, env)
        return env

    def _write_envmap_preview(self, output_path, env):
        env = np.clip(env, 0.0, None)
        # Save PNG as tone-mapped preview (HDR linear values can look almost black in naive 8-bit export).
        env_tm = env / (1.0 + env)
        env_tm = np.power(np.clip(env_tm, 0.0, 1.0), 1.0 / 2.2)
        import imageio.v3 as iio
        iio.imwrite(output_path, np.clip(env_tm * 255.0, 0, 255).astype(np.uint8))

    def create_from_pcd(self, xyz=None, t_joints=None, joint_parents=None, all_poses=None, lbs_weights_grid_info=None, xyz_vt=None, xyz_ft=None):
        xyz = torch.as_tensor(xyz).float().cuda() # [N,3]
        N = xyz.shape[0]
        print("Number of points at initialization : ", N)

        init_opacity = 0.8
        init_color = 0.5

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = knn_points(xyz[None], xyz[None], K=4)[0][0,:,1:].mean(dim=-1, keepdim=True)  
        scale = self.scaling_inverse_activation(torch.sqrt(dist2_avg)).tile([1,3])  # [N,3]
        rotation = torch.zeros((N, 4)).float().cuda()
        rotation[:, 0] = 1  # [N,4]
        opacity = torch.full((N,), self.inverse_opacity_activation(torch.tensor(init_opacity))).float().cuda()  # [N,]
        sh0 = torch.full((N, 1, 3), RGB2SH(init_color)).float().cuda() 
        shN = torch.zeros((N, 3, 3)).float().cuda()
        init_albedo = self.inverse_color_activation(torch.full((N, 3), init_color, device=xyz.device))
        albedo = init_albedo.float().cuda()
        normal = GaussianModel._estimate_point_normals_from_xyz(xyz)
        roughness = torch.full((N,), self.inverse_opacity_activation(torch.tensor(1.0, device=xyz.device))).float().cuda()
        specular = torch.full((N,), self.inverse_opacity_activation(torch.tensor(0.0, device=xyz.device))).float().cuda()
        xyz_offset = torch.zeros_like(xyz)

        self._xyz = xyz
        self.xyz_offset = nn.Parameter(xyz_offset.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scale.requires_grad_(True))
        self._sh0 = nn.Parameter(sh0.requires_grad_(True))
        self._shN = nn.Parameter(shN.requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._specular = nn.Parameter(specular.requires_grad_(True))

        self.t_joints = torch.as_tensor(t_joints).detach().float().cpu()
        self.joint_parents = torch.as_tensor(joint_parents).detach().cpu()

        for key in all_poses: all_poses[key] = torch.as_tensor(all_poses[key]).float().cpu()
        self.all_poses = all_poses

        ginfo = lbs_weights_grid_info
        for key in ['grid', 'bbox_min', 'bbox_max', 'grid_dims']: ginfo[key] = torch.as_tensor(ginfo[key]).detach().cuda()
        self.weights_grid_info = ginfo

        # Pose encoder
        models = [MLP(layers_size_list=[63, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]) for i in range(len(xyz_ft))]
        params, _ = stack_module_state(models)
        self.encoder_feat_model_meta = MLP(layers_size_list=[63, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]).to('meta')
        for k, v in params.items():
            params[k] = nn.Parameter(v.cuda().requires_grad_(True))
        self.encoder_feat_params = params

        # basis
        dxyz_bs = torch.zeros((len(xyz_vt), self.num_vt_basis, 3)).float().cuda()
        sh0_bs = torch.zeros((N, self.num_basis, 1, 3)).float().cuda()
        shN_bs = torch.zeros((N, self.num_basis, 3, 3)).float().cuda()
        scaling_bs = torch.zeros((N, self.num_basis, 3)).float().cuda()
        rotation_bs = torch.zeros((N, self.num_basis, 4)).float().cuda()
        opacity_bs = torch.zeros((N, self.num_basis)).float().cuda()
        albedo_bs = torch.zeros((N, self.num_basis, 1, 3)).float().cuda()
        normal_bs = torch.zeros((N, self.num_basis, 3)).float().cuda()
        roughness_bs = torch.zeros((N, self.num_basis)).float().cuda()
        specular_bs = torch.zeros((N, self.num_basis)).float().cuda()
        for data in [dxyz_bs, sh0_bs, scaling_bs, rotation_bs, opacity_bs, albedo_bs, normal_bs, roughness_bs, specular_bs]:
            nn.init.uniform_(data[0], -0.002, 0.002)
            data[1:] = data[0]
        self.dxyz_bs = nn.Parameter(dxyz_bs.requires_grad_(True))
        self.sh0_bs = nn.Parameter(sh0_bs.requires_grad_(True))
        self.shN_bs = nn.Parameter(shN_bs.requires_grad_(True))
        self.scaling_bs = nn.Parameter(scaling_bs.requires_grad_(True))
        self.rotation_bs = nn.Parameter(rotation_bs.requires_grad_(True))
        self.opacity_bs = nn.Parameter(opacity_bs.requires_grad_(True))
        self.albedo_bs = nn.Parameter(albedo_bs.requires_grad_(True))
        self.normal_bs = nn.Parameter(normal_bs.requires_grad_(True))
        self.roughness_bs = nn.Parameter(roughness_bs.requires_grad_(True))
        self.specular_bs = nn.Parameter(specular_bs.requires_grad_(True))

        self.deferred_light_sh = nn.Parameter(torch.zeros((9, 3), dtype=torch.float32, device=xyz.device).requires_grad_(True))
        self.deferred_light_dc = nn.Parameter(torch.full((3,), 0.5, dtype=torch.float32, device=xyz.device).requires_grad_(True))

        xyz_ft = torch.as_tensor(xyz_ft).float().cuda()
        xyz_vt = torch.as_tensor(xyz_vt).float().cuda()
        self.dxyz_vt = nn.Parameter(torch.zeros_like(xyz_vt).float().cuda().requires_grad_(True))

        self.prepare_interpolating_weights(xyz_ft, xyz_vt)

        self.init()

    def training_setup(self, args: Config, scene_scale):
        eps=1e-15 
        betas = (1 - 1 * (1 - 0.9), 1 - 1 * (1 - 0.999))
        decay = 0.001

        optimizers = {
            'dxyz': Adam([self.dxyz_vt], args.position_lr * scene_scale, betas, eps),
            'scales': Adam([self._scaling], args.scaling_lr, betas, eps),
            'quats': Adam([self._rotation], args.rotation_lr, betas, eps),
            'opacities': Adam([self._opacity], args.opacity_lr, betas, eps),
            'sh0': Adam([self._sh0], args.color_lr, betas, eps),
            'shN': Adam([self._shN], args.color_lr / 20, betas, eps),

            'dxyz_bs': Adam([self.dxyz_bs], args.position_lr * scene_scale / 10, betas, eps),
            'dscales_bs': Adam([self.scaling_bs], args.scaling_lr / 5, betas, eps),
            'dquats_bs': Adam([self.rotation_bs], args.rotation_lr / 5, betas, eps),
            'dopacities_bs': Adam([self.opacity_bs], args.opacity_lr / 5, betas, eps),
            'dsh0_bs': Adam([self.sh0_bs], args.color_lr / 5, betas, eps),
            'dshN_bs': Adam([self.shN_bs], args.color_lr / 200, betas, eps),

            'encoder_feat_params': AdamW(self.encoder_feat_params.values(), args.encoder_lr, betas, eps, decay),

            'xyz_offset': Adam([self.xyz_offset], args.xyz_offset_lr, betas, eps),
        }
        if getattr(args, 'use_deferredgs', False):
            self.use_deferredgs = True
            optimizers.update({
                'albedo': Adam([self._albedo], args.color_lr, betas, eps),
                'normal': Adam([self._normal], args.rotation_lr, betas, eps),
                'roughness': Adam([self._roughness], args.opacity_lr, betas, eps),
                'specular': Adam([self._specular], args.opacity_lr, betas, eps),
                'albedo_bs': Adam([self.albedo_bs], args.color_lr / 5, betas, eps),
                'normal_bs': Adam([self.normal_bs], args.rotation_lr / 5, betas, eps),
                'roughness_bs': Adam([self.roughness_bs], args.opacity_lr / 5, betas, eps),
                'specular_bs': Adam([self.specular_bs], args.opacity_lr / 5, betas, eps),
                'deferred_light': Adam([self.deferred_light_sh, self.deferred_light_dc], getattr(args, 'deferred_light_lr', args.color_lr), betas, eps),
            })

        schedulers = [
            ExponentialLR(optimizers['dxyz'], gamma=0.01 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['scales'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['quats'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['opacities'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['sh0'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['shN'], gamma=0.1 ** (1.0 / args.iterations)),

            ExponentialLR(optimizers['dxyz_bs'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['encoder_feat_params'], gamma=0.1 ** (1.0 / args.iterations)),

            ExponentialLR(optimizers['xyz_offset'], gamma=0.1 ** (1.0 / args.iterations)),
        ]
        if getattr(args, 'use_deferredgs', False):
            schedulers.extend([
                ExponentialLR(optimizers['albedo'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['normal'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['roughness'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['specular'], gamma=0.1 ** (1.0 / args.iterations)),
                ExponentialLR(optimizers['deferred_light'], gamma=0.1 ** (1.0 / args.iterations)),
            ])

        self.optimizers = optimizers
        self.schedulers = schedulers

    def optimizer_step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()
        
        self.cache_dict = {}

    def render(self, cam, override_color=None, scaling_modifier=1.0, background=None):
        sh = self.get_sh      # can be faster
        covars = self.get_covariance(scaling_modifier)
        if override_color is None:
            cam_pos = torch.linalg.inv_ex(cam['w2c'])[0][:3,3]
            override_color = self.get_color(cam_pos)
        
        image, alpha, info = rasterization(
            means=self.get_xyz,
            quats=None,
            scales=None,
            opacities=self.get_opacity,
            colors=override_color,
            viewmats=cam['w2c'][None],  # [1, 4, 4]
            Ks=cam['K'][None],  # [1, 3, 3]
            width=cam['width'],
            height=cam['height'],
            packed=False,
            near_plane=0.1,
            backgrounds=background[None],  # [1, 3]
            covars=covars,
        )
        return image[0], alpha[0], info

    def init_body(self):
        # Rots = batch_rodrigues(smpl.smpl_bigpose.reshape(-1,3)).cuda()
        # Ac = batch_rigid_transform(Rots[None], self.t_joints[None], self.joint_parents)[1][0]
        Ac = rigid_transform_tensor(smpl.smpl_bigpose, self.t_joints, self.joint_parents).cpu()
        self.Ac_inv = torch.linalg.inv(Ac)
        self.reset_pose()

    def reset_pose(self):
        self.Rh = torch.eye(3, dtype=torch.float32, device='cpu')
        self.Th = torch.zeros(3, dtype=torch.float32, device='cpu')
        self.smpl_poses = smpl.smpl_tpose.cpu()

    @property
    def smpl_poses(self):
        return self._smpl_poses
    
    @smpl_poses.setter
    def smpl_poses(self, value):
        self.cache_dict = {}
        self._smpl_poses = value.cpu()
        self.smpl_poses_cuda = value.cuda(non_blocking=True)

    @property
    def Rh(self):
        return self._Rh
    
    @Rh.setter
    def Rh(self, value):
        if np.allclose(value.cpu().numpy(), np.eye(3), atol=1e-5):
            self._Rh = None
        else:
            self._Rh = value.cuda(non_blocking=True)

    @property
    def Th(self):
        return self._Th
    
    @Th.setter
    def Th(self, value):
        self._Th = value.cuda(non_blocking=True)

    def prepare_interpolating_weights(self, xyz_ft, xyz_vt):
        self.xyz_vt = xyz_vt
        self.xyz_ft = xyz_ft

        dists, idxs, _ = knn_points(
            p1=self._xyz[None],
            p2=xyz_vt[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)

        _, idxs, _ = knn_points(
            p1=xyz_vt[None],
            p2=xyz_vt[None],
            K=7,
        )
        nbr_vt = idxs[0]

        self.nbr_gs = nbr_gs
        self.nbr_gs_invdist = nbr_gs_invdist
        self.nbr_vt = nbr_vt

        dists, idxs, _ = knn_points(
            p1=self._xyz[None],
            p2=xyz_ft[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)
        self.nbr_gsft = nbr_gs
        self.nbr_gsft_wght = nbr_gs_wght

        dists, idxs, _ = knn_points(
            p1=self.xyz_vt[None],
            p2=xyz_ft[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)
        self.nbr_vtft = nbr_gs
        self.nbr_vtft_wght = nbr_gs_wght
