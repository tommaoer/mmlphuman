
import torch
import numpy as np
from torch import nn
import os

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
from utils.sh_utils import RGB2SH
from utils.envmap_utils import load_envmap_tensor, sample_latlong, save_envmap_png

class GaussianModel:

    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = F.normalize

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = torch.logit

    def _to_logit(self, x, eps=1e-4):
        x = torch.clamp(x, eps, 1 - eps)
        return torch.logit(x)

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
        self._roughness = torch.empty(0)
        self._specular = torch.empty(0)
        self.sh_degree = 0
        self.use_deferred = False
        self.optimize_envmap = False
        self.envmap = None

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
            '_roughness': self._roughness,
            '_specular': self._specular,
            'sh_degree': self.sh_degree,
            'use_deferred': self.use_deferred,
            'optimize_envmap': self.optimize_envmap,
            'envmap': self.envmap,

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

            'is_dxyz_bs': self.is_dxyz_bs,
            'is_gsparam_bs': self.is_gsparam_bs,
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
        self._roughness = loader('_roughness')
        self._specular = loader('_specular') if '_specular' in data else loader('_metallic')
        self.sh_degree = data['sh_degree']
        self.use_deferred = bool(loader('use_deferred') if 'use_deferred' in data else False)
        self.optimize_envmap = bool(loader('optimize_envmap') if 'optimize_envmap' in data else False)
        self.envmap = loader('envmap')

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

        self.is_dxyz_bs = loader('is_dxyz_bs')
        self.is_gsparam_bs = loader('is_gsparam_bs')

        self.init()

    def init(self):
        self.init_body() 
        self.reset_pose()   

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

    def init_deferred(self, args: Config, mode='train'):
        self.use_deferred = bool(getattr(args, 'use_deferred_rendering', False))
        if not self.use_deferred:
            return
        if self._albedo is None or self._albedo.numel() == 0:
            n = self._xyz.shape[0]
            self._albedo = nn.Parameter(torch.full((n, 3), float(self._to_logit(torch.tensor(0.5))), device='cuda').requires_grad_(True))
            self._roughness = nn.Parameter(torch.full((n, 1), float(self._to_logit(torch.tensor(0.95))), device='cuda').requires_grad_(True))
            self._specular = nn.Parameter(torch.full((n, 1), float(self._to_logit(torch.tensor(0.05))), device='cuda').requires_grad_(True))

        is_train = mode == 'train'
        self.optimize_envmap = True if is_train else False
        envmap_path = getattr(args, 'envmap_path', None)
        env_h = int(getattr(args, 'envmap_height', 32))
        env_w = int(getattr(args, 'envmap_width', 64))

        if is_train:
            env = torch.ones((env_h, env_w, 3), device='cuda')
            self.envmap = nn.Parameter(env.requires_grad_(True))
            return

        if envmap_path is not None and len(envmap_path) > 0:
            env = load_envmap_tensor(envmap_path, device='cuda')
            self.envmap = nn.Parameter(env.requires_grad_(False))
            return

        if self.envmap is not None and self.envmap.numel() > 0:
            self.envmap = nn.Parameter(self.envmap.detach().requires_grad_(False))
        else:
            env = torch.ones((env_h, env_w, 3), device='cuda')
            self.envmap = nn.Parameter(env.requires_grad_(False))

    def _quat_to_rot(self, quat):
        quat = F.normalize(quat, dim=-1)
        w, x, y, z = quat.unbind(dim=-1)
        two = 2.0
        rot = torch.stack([
            1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
        ], dim=-1)
        return rot.reshape(-1, 3, 3)

    def get_deferred_components(self, cam_pos):
        if self.envmap is None:
            color = self.get_color(cam_pos)
            return dict(color=color, albedo=color, diffuse=color, specular=torch.zeros_like(color), normal=torch.zeros_like(color))

        scales = self.get_cano_scaling
        axis_id = torch.argmin(scales, dim=-1)
        local_n = F.one_hot(axis_id, num_classes=3).float()
        rot = self._quat_to_rot(self.get_cano_rotation)
        n_cano = torch.einsum('nij,nj->ni', rot, local_n)

        pose_rot = self.get_Gweights[:, :3, :3]
        n_world = torch.einsum('nij,nj->ni', pose_rot, n_cano)
        if self.Rh is not None:
            n_world = torch.einsum('ij,nj->ni', self.Rh, n_world)
        n_world = F.normalize(n_world, dim=-1)

        view_dir = F.normalize(cam_pos - self.get_xyz, dim=-1)
        reflect_dir = F.normalize(2 * (n_world * view_dir).sum(-1, keepdim=True) * n_world - view_dir, dim=-1)

        diffuse = sample_latlong(self.envmap, n_world)
        spec_env = sample_latlong(self.envmap, reflect_dir)

        albedo = torch.sigmoid(self._albedo)
        roughness = torch.sigmoid(self._roughness)
        spec_power = torch.clamp(1.0 - roughness, min=0.02, max=1.0)
        specular_strength = torch.sigmoid(self._specular)
        specular = spec_env * spec_power * specular_strength
        color = albedo * diffuse + specular
        normal = (n_world + 1.0) * 0.5
        return dict(
            color=torch.clamp(color, 0.0, 10.0),
            albedo=torch.clamp(albedo, 0.0, 1.0),
            diffuse=torch.clamp(diffuse, 0.0, 10.0),
            specular=torch.clamp(specular, 0.0, 10.0),
            normal=torch.clamp(normal, 0.0, 1.0),
        )

    def get_deferred_color(self, cam_pos):
        return self.get_deferred_components(cam_pos)['color']

    @torch.no_grad()
    def save_envmap_visualization(self, save_path):
        if self.envmap is None:
            return
        save_envmap_png(self.envmap, save_path)

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
        xyz_offset = torch.zeros_like(xyz)

        self._xyz = xyz
        self.xyz_offset = nn.Parameter(xyz_offset.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scale.requires_grad_(True))
        self._sh0 = nn.Parameter(sh0.requires_grad_(True))
        self._shN = nn.Parameter(shN.requires_grad_(True))
        self._albedo = nn.Parameter(torch.full((N, 3), float(self._to_logit(torch.tensor(0.5))), device='cuda').requires_grad_(True))
        self._roughness = nn.Parameter(torch.full((N, 1), float(self._to_logit(torch.tensor(0.95))), device='cuda').requires_grad_(True))
        self._specular = nn.Parameter(torch.full((N, 1), float(self._to_logit(torch.tensor(0.05))), device='cuda').requires_grad_(True))

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
        for data in [dxyz_bs, sh0_bs, scaling_bs, rotation_bs, opacity_bs]:
            nn.init.uniform_(data[0], -0.002, 0.002)
            data[1:] = data[0]
        self.dxyz_bs = nn.Parameter(dxyz_bs.requires_grad_(True))
        self.sh0_bs = nn.Parameter(sh0_bs.requires_grad_(True))
        self.shN_bs = nn.Parameter(shN_bs.requires_grad_(True))
        self.scaling_bs = nn.Parameter(scaling_bs.requires_grad_(True))
        self.rotation_bs = nn.Parameter(rotation_bs.requires_grad_(True))
        self.opacity_bs = nn.Parameter(opacity_bs.requires_grad_(True))

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
        if self.use_deferred:
            optimizers['albedo'] = Adam([self._albedo], args.color_lr, betas, eps)
            optimizers['roughness'] = Adam([self._roughness], args.color_lr / 2, betas, eps)
            optimizers['specular'] = Adam([self._specular], args.color_lr / 2, betas, eps)
            if self.optimize_envmap and self.envmap is not None:
                optimizers['envmap'] = Adam([self.envmap], args.envmap_lr, betas, eps)

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
        if self.use_deferred and self.optimize_envmap and self.envmap is not None:
            schedulers.append(ExponentialLR(optimizers['envmap'], gamma=0.2 ** (1.0 / args.iterations)))

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
        covars = self.get_covariance(scaling_modifier)
        if override_color is None:
            cam_pos = torch.linalg.inv_ex(cam['w2c'])[0][:3,3]
            override_color = self.get_deferred_color(cam_pos) if self.use_deferred else self.get_color(cam_pos)
        
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

    def render_deferred_buffers(self, cam, scaling_modifier=1.0, background=None):
        cam_pos = torch.linalg.inv_ex(cam['w2c'])[0][:3,3]
        comp = self.get_deferred_components(cam_pos)
        renders = {}
        for name in ['albedo', 'diffuse', 'specular', 'normal']:
            image, _, _ = self.render(cam, override_color=comp[name], scaling_modifier=scaling_modifier, background=background)
            renders[name] = image
        return renders

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

        def _safe_knn_weights(dists):
            dist_sqrt = torch.sqrt(torch.clamp_min(dists[0], 1e-12))
            invdist = 1.0 / dist_sqrt
            invdist_sum = torch.clamp_min(torch.sum(invdist, dim=-1, keepdim=True), 1e-12)
            weights = invdist / invdist_sum
            return invdist, weights

        dists, idxs, _ = knn_points(
            p1=self._xyz[None],
            p2=xyz_vt[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist, nbr_gs_wght = _safe_knn_weights(dists)

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
        nbr_gs_invdist, nbr_gs_wght = _safe_knn_weights(dists)
        self.nbr_gsft = nbr_gs
        self.nbr_gsft_wght = nbr_gs_wght

        dists, idxs, _ = knn_points(
            p1=self.xyz_vt[None],
            p2=xyz_ft[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist, nbr_gs_wght = _safe_knn_weights(dists)
        self.nbr_vtft = nbr_gs
        self.nbr_vtft_wght = nbr_gs_wght
