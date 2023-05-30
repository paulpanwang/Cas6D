
import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from network.refiner_ablation import FPN
from dataset.database import NormalizedDatabase, normalize_pose, get_object_center, get_diameter, denormalize_pose
from network.operator import pose_apply_th, normalize_coords
from network.pretrain_models import VGGBNPretrainV3
from utils.base_utils import pose_inverse, project_points, color_map_forward, to_cuda, pose_compose
from utils.database_utils import look_at_crop, select_reference_img_ids_refinement, normalize_reference_views
from utils.pose_utils import let_me_look_at, compose_sim_pose, pose_sim_to_pose_rigid
from utils.imgs_info import imgs_info_to_torch
from network.vis_dino_encoder import VitExtractor


# sin-cose embedding module
class CasEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super(CasEmbedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class CasFeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(CasFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class CasAttention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(CasAttention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class CasTransformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(CasTransformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = CasFeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = CasAttention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


class CasRefineFeatureNet(nn.Module):
    def __init__(self, \
                 norm_layer='instance',\
                 use_dino=False,\
                 upsample=False):

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
            nn.Conv2d(256, 64, 3, 1, 1),
            norm(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
            nn.Conv2d(256, 64, 3, 1, 1),
            norm(64),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(64*3, 128, 3, 1, 1),
            norm(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            norm(128),
        )
        self.upsample = upsample
        self.use_dino = use_dino

        if self.upsample:
            self.down_sample = nn.Conv2d(in_channels=128, \
                                        out_channels=64, \
                                        kernel_size=1,\
                                        stride=1,\
                                        padding=0, \
                                        bias=True)  

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
        self.fpn = FPN([512,512,128],128)  
        self.use_fpn = False     
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
        if self.upsample:
            imgs = F.interpolate(imgs, size=(int(1.5*h), int(1.5*h)))

        if self.use_dino:
            dino_imgs = imgs.clone()
            
        imgs = self.img_norm(imgs)
        self.backbone.eval()
        with torch.no_grad():
            x0, x1, x2 = self.backbone(imgs)
            x0 = F.normalize(x0, dim=1)
            x1 = F.normalize(x1, dim=1)
            x2 = F.normalize(x2, dim=1)
        x0 = self.conv0(x0)
        x1 = F.interpolate(self.conv1(x1),scale_factor=2,mode='bilinear')
        x2 = F.interpolate(self.conv2(x2),scale_factor=4,mode='bilinear')
        x = torch.cat([x0,x1,x2],1)
        if self.use_fpn:
            x = self.fpn([x0,x1,x])
        else:
            x = self.conv_out(x)  
        if self.use_dino:
            # -------------------------------------------------------- 
            dino_imgs = F.interpolate(dino_imgs, size=(256, 256)) 
            dino_ret =  self.fea_ext.get_vit_attn_feat(dino_imgs)
            attn, cls_, feat = dino_ret['attn'], dino_ret['cls_'], dino_ret['feat']
            dino_fea = feat.permute(0,2,1).reshape(-1,384,32,32)    
            fused_fea = torch.cat( (x,dino_fea), dim = 1)
            x = self.fuse_conv(fused_fea)
            # --------------------------------------------------------
        return x

class CasRefineVolumeEncodingNet(nn.Module):
    def __init__(self,norm_layer='no_norm'):
        super().__init__()
        if norm_layer == 'instance':
            norm=nn.InstanceNorm3d
        else:
            raise NotImplementedError

        self.mean_embed = nn.Sequential(
            nn.Conv3d(128 * 2, 64, 3, 1, 1),
            norm(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, 3, 1, 1)
        )
        self.var_embed = nn.Sequential(
            nn.Conv3d(128, 64, 3, 1, 1),
            norm(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, 3, 1, 1)
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(64*2, 64, 3, 1, 1), # 32
            norm(64),
            nn.ReLU(True),
        ) # 32

        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1),
            norm(128),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1),
            norm(128),
            nn.ReLU(True),
        ) # 16

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1),
            norm(256),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, 1),
            norm(256),
            nn.ReLU(True),
        )  #8

        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, 2, 1),
            norm(512),
            nn.ReLU(True),
            nn.Conv3d(512, 512, 3, 1, 1)
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

class CasRefineRegressor(nn.Module):
    def __init__(self, upsample=False):
        super().__init__()
        if upsample:
            self.fc = nn.Sequential(   fc( int((1.5)**3*512 * 4**3) , 512), nn.Dropout(p=0.15), fc(512, 512))
        else:
            self.fc = nn.Sequential(fc(512 * 4**3, 512), fc(512, 512))
        self.fcr = nn.Linear(512,4)
        self.fct = nn.Linear(512,2)
        self.fcs = nn.Linear(512,1)

    def forward(self, x):
   
        x = self.fc(x)
        r = F.normalize(self.fcr(x),dim=1)
        t = self.fct(x)
        s = self.fcs(x)
        return r, t, s


class CasTransformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layer, nhead=8, dropout=0.1):
        super(CasTransformer, self).__init__()
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

from loguru import logger

class CasVolumeRefiner(nn.Module):
    default_cfg = {
        "refiner_sample_num": 32,
    }
    def __init__(self, cfg, upsample=False):
        self.cfg={**self.default_cfg, **cfg}
        super().__init__()
        
        self.use_dino = self.cfg.get("use_dino", False)  
        self.use_transformer = self.cfg.get("use_transformer", False) 
        logger.debug( f"VolumeRefiner use_dino:{self.use_dino}, use_transformer:{self.use_transformer}" )
        self.upsample = upsample
        self.feature_net = CasRefineFeatureNet('instance', self.use_dino, upsample)
        self.volume_net = CasRefineVolumeEncodingNet('instance')
        self.regressor = CasRefineRegressor(upsample)
        
        # used in inference
        self.ref_database = None
        self.ref_ids = None

        if self.use_transformer:
            self.view_trans = CasTransformer(
                input_size=32768, 
                output_size=32768, 
                hidden_size=64,
                num_layer=1,
                nhead=8, 
                dropout=0.1
            )

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
        volume_feats = F.grid_sample(feats, verts, mode='bilinear', align_corners=False) # b,f,sx,sy*sz
        return volume_feats.reshape(b, f, sx, sy, sz)


    def construct_feature_volume(self, que_imgs_info, ref_imgs_info, feature_extractor, sample_num):
        """_summary_

        Args:
            que_imgs_info (_type_): _description_
            ref_imgs_info (_type_): _description_
            feature_extractor (_type_): 特征提取器
            sample_num (_type_): 采样图片的个数

        Returns:
            _type_: _description_
        """
        # build a volume on the unit cube
        sn = sample_num
        device = que_imgs_info['imgs'].device
        vol_coords = torch.linspace(-1, 1, sample_num, dtype=torch.float32, device=device)
        vol_coords = torch.stack(torch.meshgrid(vol_coords,vol_coords,vol_coords),-1) # sn,sn,sn,3
        vol_coords = vol_coords.reshape(1,sn**3,3)

        # rotate volume to align with the input pose, but still in the object coordinate
        poses_in = que_imgs_info['poses_in'] # qn,3,4
    
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
            ref_feats = feature_extractor(ref_imgs_info['imgs'][qi]) # rfn,f,h,w
            rfn = ref_feats.shape[0]
            vol_coords_cur = vol_coords[qi:qi+1].repeat(rfn,1,1,1,1) # rfn,sx,sy,sz,3
            vol_feats = CasVolumeRefiner.interpolate_volume_feats(ref_feats, vol_coords_cur, ref_proj[qi], h_in, w_in)

            if self.use_transformer:
                x = vol_feats.view(rfn,128,sn*sn*sn)
                x  = self.view_trans(x)
                vol_feats = x[0].view(rfn,128,sn,sn,sn)
    
            vol_feats_mean.append(torch.mean(vol_feats, 0))
            vol_feats_std.append(torch.std(vol_feats, 0))

        vol_feats_mean = torch.stack(vol_feats_mean, 0)
        vol_feats_std = torch.stack(vol_feats_std, 0)

        # project onto query view
        h_in, w_in = que_imgs_info['imgs'].shape[-2:]
        que_feats = feature_extractor(que_imgs_info['imgs']) # qn,f,h,w
        que_proj = que_imgs_info['Ks_in'] @ que_imgs_info['poses_in']
        vol_feats_in = CasVolumeRefiner.interpolate_volume_feats(que_feats, vol_coords, que_proj, h_in, w_in) # qn,f,sx,sy,sz

        return vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords

    def forward(self, data):
        is_inference = data['inference'] if 'inference' in data else False
        que_imgs_info = data['que_imgs_info'].copy()
        ref_imgs_info = data['ref_imgs_info'].copy()

        if self.upsample:
            refiner_sample_num = int(self.cfg['refiner_sample_num']*1.5) 
        else:
            refiner_sample_num = self.cfg['refiner_sample_num']

        vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords = self.construct_feature_volume(
            que_imgs_info, ref_imgs_info, self.feature_net, refiner_sample_num) # qn,f,dn,h,w   qn,dn

        vol_feats = torch.cat([vol_feats_mean, vol_feats_in], 1)
        vol_feats = self.volume_net(vol_feats, vol_feats_std)
        vol_feats = vol_feats.flatten(1) # qn, f* 4**3
        rotation, offset, scale = self.regressor(vol_feats)
        outputs={'rotation': rotation, 'offset': offset, 'scale': scale}

        if not is_inference:
            # used in training not inference
            qn, sx, sy, sz, _ = vol_coords.shape
            grids = pose_apply_th(que_imgs_info['poses_in'], vol_coords.reshape(qn, sx * sy * sz, 3))
            outputs['grids'] = grids

        return outputs

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
        ref_even_num = min(128,len(self.ref_ids))

        # normalize database and input pose
        ref_database = NormalizedDatabase(self.ref_database) # wrapper: object is in the unit sphere at origin
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
            outputs = self.forward({'que_imgs_info': que_imgs_info, 'ref_imgs_info': ref_imgs_info, 'inference': True})
            quat = outputs['rotation'].detach().cpu().numpy()[0] # 4
            scale = 2**outputs['scale'].detach().cpu().numpy()[0] # 1
            offset = outputs['offset'].detach().cpu().numpy()[0] # 2

            # print("scale:", scale , "quat:", quat, "offset:", offset )

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
    refiner = CasVolumeRefiner(refiner_cfg)
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

    # pose_pr = refiner.refine_que_imgs(que_img, que_K, pose_pr, size=128, ref_num=6, ref_even=True)
    vol_feats_mean, vol_feats_std, vol_feats_in, vol_coords = refiner.construct_feature_volume(
            que_imgs_info, ref_imgs_info, refiner.feature_net, refiner_sample_num)

    mock_data = torch.randn(6,3,128,128)
    net = CasRefineFeatureNet()
    out =  net(mock_data)
    print(out.shape)