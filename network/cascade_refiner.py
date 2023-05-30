import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from dataset.database import NormalizedDatabase, normalize_pose, get_object_center, get_diameter, denormalize_pose
from network.operator import pose_apply_th, normalize_coords
from network.pretrain_models import VGGBNPretrainV3
from utils.base_utils import pose_inverse, project_points, color_map_forward, to_cuda, pose_compose
from utils.database_utils import look_at_crop, select_reference_img_ids_refinement, normalize_reference_views
from utils.pose_utils import let_me_look_at, compose_sim_pose, pose_sim_to_pose_rigid
from utils.imgs_info import imgs_info_to_torch
from network.vis_dino_encoder import VitExtractor
from loguru import logger
from gpu_mem_track import MemTracker
gpu_tracker = MemTracker()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
import time

class RefineFeatureNet(nn.Module):
    def __init__(self, \
                 norm_layer='instance',\
                 use_dino=False,\
                 use_fpn=True):

        super().__init__()
        if norm_layer == 'instance':
            norm=nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1),
            norm(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            norm(64),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1),
            norm(128),
        )
 

        self.use_fpn = use_fpn
        self.use_dino = use_dino

        if self.use_dino:
            self.fuse_conv = nn.Conv2d(in_channels=512, \
                                       out_channels=128, \
                                       kernel_size=1,\
                                       stride=1,\
                                       padding=0, \
                                       bias=True)

            self.fuse_conv1 = nn.Conv2d(in_channels=256+384, \
                            out_channels=256, \
                            kernel_size=1,\
                            stride=1,\
                            padding=0, \
                            bias=True)  
            
            self.fuse_conv2 = nn.Conv2d(in_channels=512+384, \
                                        out_channels=512, \
                                        kernel_size=1,\
                                        stride=1,\
                                        padding=0, \
                                        bias=True) 

            self.fuse_conv3 = nn.Conv2d(in_channels=512+384, \
                            out_channels=512, \
                            kernel_size=1,\
                            stride=1,\
                            padding=0, \
                            bias=True) 
                            
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        
        if self.use_dino:
            self.fea_ext =  VitExtractor(model_name='dino_vits8').eval()
            for para in self.fea_ext.parameters():
                para.requires_grad = False
            self.fea_ext.requires_grad_(False) 

        self.backbone = VGGBNPretrainV3().eval()
        for para in self.backbone.parameters():
            para.requires_grad = False
        self.img_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def forward(self, imgs):
        _,_, h,w = imgs.shape
        imgs = self.img_norm(imgs)
        self.backbone.eval()
        with torch.no_grad():
            x0, x1, x2 = self.backbone(imgs)
            x0 = F.normalize(x0, dim=1)
            x1 = F.normalize(x1, dim=1)
            x2 = F.normalize(x2, dim=1) 
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        return [x1,x0]


class RefineVolumeEncodingNet(nn.Module):
    def __init__(self, fea_ch = 128, norm_layer='no_norm'):
        super().__init__()
        if norm_layer == 'instance':
            norm=nn.InstanceNorm3d
        else:
            raise NotImplementedError
     
        self.mean_embed = nn.Sequential(
            nn.Conv3d(fea_ch*2, fea_ch//2, 3, 1, 1),
            norm(fea_ch//2,),
            nn.ReLU(True),
            nn.Conv3d(fea_ch//2, fea_ch//2, 3, 1, 1)
        )

        self.var_embed = nn.Sequential(
            nn.Conv3d(fea_ch, fea_ch//2, 3, 1, 1),
            norm(fea_ch//2),
            nn.ReLU(True),
            nn.Conv3d(fea_ch//2, fea_ch//2, 3, 1, 1)
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(fea_ch, fea_ch//2, 3, 1, 1), # 32
            norm(fea_ch//2),
            nn.ReLU(True),
        ) # 32

        self.conv1 = nn.Sequential(
            nn.Conv3d(fea_ch//2, fea_ch, 3, 2, 1),
            norm(fea_ch),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(fea_ch, fea_ch, 3, 1, 1),
            norm(fea_ch),
            nn.ReLU(True),
        ) # 16

        self.conv3 = nn.Sequential(
            nn.Conv3d(fea_ch, 2*fea_ch, 3, 2, 1),
            norm(2*fea_ch),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(2*fea_ch, 2*fea_ch, 3, 1, 1),
            norm(2*fea_ch),
            nn.ReLU(True),
        )  #8

        self.conv5 = nn.Sequential(
            nn.Conv3d(2*fea_ch, 4*fea_ch, 3, 2, 1),
            norm(4*fea_ch),
            nn.ReLU(True),
            nn.Conv3d(4*fea_ch, 4*fea_ch, 3, 1, 1)
        )

    def forward(self, mean, var):
        x = torch.cat([self.mean_embed(mean),self.var_embed(var)],1)
        x = self.conv0(x)
        x = self.conv2(self.conv1(x))
        x = self.conv4(self.conv3(x))
        x = self.conv5(x)
        return x

def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)


class RefineRegressor(nn.Module):
    def __init__(self, flatten_len = 4096):
        super().__init__()
        self.fc = nn.Sequential(fc(flatten_len, 512), fc(512, 512))
        self.fcr = nn.Linear(512,4)
        self.fct = nn.Linear(512,2)
        self.fcs = nn.Linear(512,1)

    def forward(self, x):
        x = self.fc(x)
        r = F.normalize(self.fcr(x),dim=1)
        t = self.fct(x)
        s = self.fcs(x)
        return r, t, s


class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layer, nhead=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.Sequential(*[nn.TransformerEncoderLayer(hidden_size, \
            nhead=nhead, dropout=dropout, batch_first=True) for _ in range(num_layer)])
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, state=None):
        x = self.linear1(self.dropout(x))
        output = self.transformer_encoder(x)
        output = self.linear2(output)
        return output, state



class VolumeRefiner(nn.Module):
    default_cfg = {
        "refiner_sample_num": 32,
    }
    def __init__(self, cfg, upsample=False):
        self.cfg={**self.default_cfg, **cfg}
        super().__init__()
        self.use_dino = self.cfg.get("use_dino", False) 
        self.use_fpn = self.cfg.get("use_fpn", False)  
        self.upsample = upsample
        self.feature_net = RefineFeatureNet('instance', self.use_dino, self.use_fpn )
        self.training = self.cfg.get("trainng", False)

        # self.feature_channel = [[128,64], [64,32,16], [64,16,4] ]
        self.feature_channel = [ 128,64 ]
        self.scale_rmin, self.scale_rmax = nn.Parameter(torch.Tensor(1)), nn.Parameter(torch.Tensor(1)) 
        self.translation_rmin, self.translation_rmax = nn.Parameter(torch.Tensor(1)), nn.Parameter(torch.Tensor(1))
        self.quan_rmin , self.quan_rmax =  nn.Parameter(torch.Tensor(1)), nn.Parameter(torch.Tensor(1))

        self.stage = len(self.feature_channel)
        self.volume_net = list()
        for idx,f_c in enumerate(self.feature_channel):
            self.volume_net.append(RefineVolumeEncodingNet(self.feature_channel[idx],'instance').cuda())

        self.regressor = [RefineRegressor(flatten_len=4096).cuda(),\
                          RefineRegressor(flatten_len=16384).cuda(),  ]

        self.ref_database = None
        self.ref_ids = None
        # self.stage = 3
        self.stage = 2
        logger.debug( f"VolumeRefiner use_dino:{self.use_dino} ,  stage:{self.stage}" )

    @staticmethod
    def update_state(scale, tranlation, quan):
        self.scale_rmin = min(self.scale_rmin, scale.min() )
        self.scale_rmax = max(self.scale_max, scale.max() )
        self.translation_rmin = min( tranlation.min(),  self.translation_rmin   )
        self.translation_rmax  = max(tranlation.max(), self.translation_rmax  )
        self.quan_rmin = min( self.quan_rmin, quan.min()  )
        self.quan_rmax = max(self.quan_rmax,  quan.max() )


    @staticmethod
    def interpolate_volume_feats(feats, verts, projs, h_in, w_in):
        """
        @param feats: b,f,h,w 
        @param verts: b,sx,sy,sz,3
        @param projs: b,3,4 : project matric
        @param h_in:  int
        @param w_in:  int
        @return:
        """
        b, sx, sy, sz, _ = verts.shape
        b, f, h, w = feats.shape
        R, t = projs[:,:3,:3], projs[:,:3,3:] # b,3,3  b,3,1
        verts = verts.reshape(b,sx*sy*sz,3)
        verts = verts @ R.permute(0, 2, 1) + t.permute(0, 2, 1) #

        depth = verts[:, :, -1:]
        depth[depth < 1e-4] = 1e-4
        verts = verts[:, :, :2] / depth  # [b,sx*sy*sz,2]
        verts = normalize_coords(verts, h_in, w_in) # b,sx*sy*sz,2]
        verts = verts.reshape([b, sx, sy*sz, 2])
        volume_feats = F.grid_sample(feats, verts, mode='nearest', align_corners=False) # b,f,sx,sy*sz
        return volume_feats.reshape(b, f, sx, sy, sz)


    def construct_feature_volume(self, que_imgs_info, ref_imgs_info, \
                                 poses_in, query_image_feature,\
                                 ref_image_featrues_list, \
                                 sample_num = 32, \
                                 stage_idx = 0):
        """_summary_

        Args:
            que_imgs_info (_type_): _description_
            ref_imgs_info (_type_): _description_
            poses_in (_type_): 输入的姿态
            query_image_feature (_type_): feature of query image
            ref_image_featrues: feature of reference images
            sample_num: feature volume sample number
            stage_idx: the index of the cacaded network stage
        Returns:
            _type_: _description_
        """

        # build a volume on the unit cube
        sn = (sample_num//2) *  (2**stage_idx)
        device = que_imgs_info['imgs'].device
        vol_coords = torch.linspace(-1, 1, sn, dtype=torch.float32, device=device)
        vol_coords = torch.stack(torch.meshgrid(vol_coords,vol_coords,vol_coords),-1) # sn,sn,sn,3
        vol_coords = vol_coords.reshape(1,sn**3,3)
        rotation = poses_in[:,:3,:3] # qn,3,3
        vol_coords = vol_coords @ rotation # qn,sn**3,3
        qn = poses_in.shape[0]
        vol_coords = vol_coords.reshape(qn, sn, sn, sn, 3)
        # project onto every reference view
        ref_poses = ref_imgs_info['poses'] # qn,rfn,3,4
        ref_Ks = ref_imgs_info['Ks'] # qn,rfn,3,3
        ref_proj = ref_Ks @ ref_poses # qn,rfn,3,4
        vol_feats_mean, vol_feats_std = [], []
        h_in, w_in = ref_imgs_info['imgs'].shape[-2:]
        for qi in range(qn):
            ref_feats = ref_image_featrues_list[qi][stage_idx]    # rfn,f,h,w
            rfn = ref_feats.shape[0]
            vol_coords_cur = vol_coords[qi:qi+1].repeat(rfn,1,1,1,1) # rfn,sx,sy,sz,3

            vol_feats = VolumeRefiner.interpolate_volume_feats(ref_feats, vol_coords_cur, ref_proj[qi], h_in, w_in)

            vol_feats_mean.append(torch.mean(vol_feats, 0))
            vol_feats_std.append(torch.std(vol_feats, 0))
        vol_feats_mean = torch.stack(vol_feats_mean, 0)
        vol_feats_std = torch.stack(vol_feats_std, 0)
        # project onto query view
        h_in, w_in = que_imgs_info['imgs'].shape[-2:]
        que_feats = query_image_feature[stage_idx]
        que_proj = que_imgs_info['Ks_in'] @ que_imgs_info['poses_in']
        # qn,f,sx,sy,sz
        vol_feats_in = VolumeRefiner.interpolate_volume_feats(que_feats, vol_coords, que_proj, h_in, w_in) 
        return vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords

    def forward(self, data):
        is_inference = data['inference'] if 'inference' in data else False
        que_imgs_info = data['que_imgs_info'].copy()
        ref_imgs_info = data['ref_imgs_info'].copy()
        if self.upsample:
            refiner_sample_num = int(self.cfg['refiner_sample_num']*1.5) 
        else:
            refiner_sample_num = self.cfg['refiner_sample_num']
        # qn,f,dn,h,w   qn,dn
        query_image_feature = self.feature_net(que_imgs_info['imgs']) # qn,f,h,w
        # rotate volume to align with the input pose, but still in the object coordinate
        poses_in = que_imgs_info['poses_in'] # qn,3,4
        qn = poses_in.shape[0]
        ref_image_featrues_list = list(list())
        # gpu_tracker.track()
        for qi in range(qn):
            ref_feats = self.feature_net(ref_imgs_info['imgs'][qi])
            ref_image_featrues_list.append(ref_feats)
        # gpu_tracker.track()

        all_stage_outputs = list()
        for stage_idx in range(self.stage):
            # gpu_tracker.track()
            vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords = \
            self.construct_feature_volume(
                que_imgs_info,\
                ref_imgs_info, \
                poses_in, \
                query_image_feature, \
                ref_image_featrues_list , \
                sample_num = refiner_sample_num, \
                stage_idx = stage_idx
            ) 
            # gpu_tracker.track()
            vol_feats = torch.cat([vol_feats_mean, vol_feats_in ], 1)
            vol_feats = self.volume_net[stage_idx](vol_feats, vol_feats_std)
            vol_feats = vol_feats.flatten(1) # qn, f* 4**3
            #traing the regression model, different levels of vol feature
            rotation, offset, scale = self.regressor[stage_idx](vol_feats)
            outputs = {'rotation': rotation, 'offset': offset, 'scale': scale}
            if not is_inference:
                qn, sx, sy, sz, _ = vol_coords.shape
                grids = pose_apply_th(que_imgs_info['poses_in'],\
                                      vol_coords.reshape(qn, sx * sy * sz, 3))
                outputs['grids'] = grids
                self.update_state(scale, tranlation, quan)

            all_stage_outputs.append(outputs)
        if not is_inference:
            return all_stage_outputs[-1]
        else:
            return all_stage_outputs[-1]

    def load_ref_imgs(self,ref_database,ref_ids):
        self.ref_database = ref_database
        self.ref_ids = ref_ids


    def refine_que_imgs(self, que_img, que_K, in_pose, size=128, ref_num=6, ref_even=False):
        """
        @param que_img:  [h,w,3]
        @param que_K:    [3,3]
        @param in_pose:  [3,4]
        @param size:     int
        @param ref_num:  int
        @param ref_even: bool
        @return:
        """
        margin = 0.05
        ref_even_num = min(128, len(self.ref_ids))
        # Normlization: normalize database and input pose,
        ref_database = NormalizedDatabase(self.ref_database) 
        # Wrapper: object is in the unit sphere at origin 
        in_pose = normalize_pose(in_pose, ref_database.scale, ref_database.offset)
        object_center = get_object_center(ref_database)
        object_diameter = get_diameter(ref_database)
        # warp the query image to look at the object w.r.t input pose
        _, new_f = let_me_look_at(in_pose, que_K, object_center)
        in_dist = np.linalg.norm(pose_inverse(in_pose)[:,3] - object_center)
        in_f = size * (1 - margin) / object_diameter * in_dist
        scale = in_f / new_f

        position = project_points(object_center[None], in_pose, que_K)[0][0]
        que_img_warp, que_K_warp, in_pose_warp, que_pose_rect, H = look_at_crop(
            que_img, que_K, in_pose, position, 0, scale, size, size)

        que_imgs_info = {
            'imgs': color_map_forward(que_img_warp).transpose([2,0,1]),  # 3,h,w
            'Ks_in': que_K_warp.astype(np.float32), # 3,3
            'poses_in': in_pose_warp.astype(np.float32), # 3,4
        }

        # import pdb
        # pdb.set_trace()
        # print( que_imgs_info['imgs'].shape ,  que_imgs_info['Ks_in'].shape  , que_imgs_info['poses_in'].shape )
        # select reference views for refinement
        ref_ids = select_reference_img_ids_refinement(ref_database, object_center, self.ref_ids, \
                                                      in_pose_warp, ref_num, ref_even, ref_even_num)

        # normalize the reference images and align the in-plane orientation w.r.t input pose.
        ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs = normalize_reference_views(
            ref_database, ref_ids, size, margin, True, in_pose_warp, que_K_warp)
        
        ref_imgs_info = {
            'imgs': color_map_forward(np.stack(ref_imgs, 0)).transpose([0, 3, 1, 2]),  # rfn,3,h,w
            'poses': np.stack(ref_poses, 0).astype(np.float32),
            'Ks': np.stack(ref_Ks, 0).astype(np.float32),
        }

        que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        ref_imgs_info = to_cuda(imgs_info_to_torch(ref_imgs_info))

        for k,v in que_imgs_info.items(): que_imgs_info[k] = v.unsqueeze(0)
        for k,v in ref_imgs_info.items(): ref_imgs_info[k] = v.unsqueeze(0)

        with torch.no_grad():
            outputs = self.forward({'que_imgs_info': que_imgs_info, \
                                    'ref_imgs_info': ref_imgs_info, \
                                    'inference': True})

            quat = outputs['rotation'].detach().cpu().numpy()[0] # 4
            scale = 2**outputs['scale'].detach().cpu().numpy()[0] # 1
            offset = outputs['offset'].detach().cpu().numpy()[0] # 2

        # compose rotation/scale/offset into a similarity transformation matrix
        pose_sim = compose_sim_pose(scale, quat, offset, in_pose_warp, object_center)
        # convert the similarity transformation to the rigid transformation
        pose_pr = pose_sim_to_pose_rigid(pose_sim, in_pose_warp, que_K_warp, que_K_warp, object_center)
        # apply the pose residual
        pose_pr = pose_compose(pose_pr, pose_inverse(que_pose_rect))
        # convert back to original coordinate system (because we use NormalizedDatabase to wrap the input)
        pose_pr = denormalize_pose(pose_pr, ref_database.scale, ref_database.offset)
        return pose_pr
    

if __name__ == "__main__":
    from utils.base_utils import load_cfg
    cfg = "configs/refiner/refiner_pretrain.yaml"
    refiner_cfg = load_cfg(cfg)
    refiner = VolumeRefiner(refiner_cfg)
    refiner_sample_num = 32
    ref_imgs_info = {
        'imgs': torch.randn(6,3,128,128) , # rfn,3,h,w
        'poses': torch.randn(6, 3, 4),
        'Ks': torch.randn(6,3,3),
    }

    que_imgs_info = {
        'imgs': torch.randn(3,128,128),  # 3,h,w
        'Ks_in': torch.randn(3, 3), # 3,3
        'poses_in':  torch.randn(3, 4), # 3,4
    }

    for k,v in que_imgs_info.items(): que_imgs_info[k] = v.unsqueeze(0)
    for k,v in ref_imgs_info.items(): ref_imgs_info[k] = v.unsqueeze(0)

    # unit test1 : refine_que_images
    # pose_pr = refiner.refine_que_imgs(que_img, que_K, pose_pr, size=128, ref_num=6, ref_even=True)

    # vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords = refiner.construct_feature_volume(
    #         que_imgs_info, ref_imgs_info, refiner.feature_net, refiner_sample_num, stage=1)
    # print(vol_feats_mean.shape, vol_feats_std.shape, vol_feats_in.shape, vol_coords.shape)

    data = dict()
    data['que_imgs_info'] = que_imgs_info
    data['ref_imgs_info'] = ref_imgs_info
    refiner(data)