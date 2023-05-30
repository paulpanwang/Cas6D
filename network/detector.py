import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from network.pretrain_models import VGGBNPretrain
from utils.base_utils import color_map_forward, transformation_crop, to_cpu_numpy
from utils.bbox_utils import parse_bbox_from_scale_offset
from network.vis_dino_encoder import VitExtractor
from loguru import logger
import skimage
import cv2
# count = 0

def show_heatmap(feature, output_jpg_name):
    data = feature
    heatmap = data.sum(0)/data.shape[0]
    heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)
    # heatmap = 1.0 - heatmap
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # skimage.io.imsave(output_jpg_name, heatmap)
    cv2.imwrite(output_jpg_name, heatmap)



class BaseDetector(nn.Module):
    def load_impl(self, ref_imgs):
        raise NotImplementedError

    def detect_impl(self, que_imgs):
        raise NotImplementedError

    def load(self, ref_imgs):
        ref_imgs = torch.from_numpy(color_map_forward(ref_imgs)).permute(0, 3, 1, 2).cuda()
        self.load_impl(ref_imgs)

    def detect(self, que_imgs):
        que_imgs = torch.from_numpy(color_map_forward(que_imgs)).permute(0, 3, 1, 2).cuda()
        return self.detect_impl(que_imgs) # 'scores' 'select_pr_offset' 'select_pr_scale'

    @staticmethod
    def parse_detect_results(results):
        """

        @param results: dict
            pool_ratio: int -- pn
            scores: qn,1,h/pn,w/pn
            select_pr_offset: qn,2,h/pn,w/pn
            select_pr_scale:  qn,1,h/pn,w/pn
            select_pr_angle:  qn,2,h/pn,w/pn # optional
        @return: all numpy ndarray
        """
        qn = results['scores'].shape[0]
        pool_ratio = results['pool_ratio']

        # max scores
        _, score_x, score_y = BaseDetector.get_select_index(results['scores']) # qn
        position = torch.stack([score_x, score_y], -1)  # qn,2

        # offset
        offset = results['select_pr_offset'][torch.arange(qn),:,score_y,score_x] # qn,2
        position = position + offset

        # to original coordinate
        position = (position + 0.5) * pool_ratio - 0.5 # qn,2

        # scale
        scale_r2q = results['select_pr_scale'][torch.arange(qn),0,score_y,score_x] # qn
        scale_r2q = 2**scale_r2q
        outputs = {'position': position.detach().cpu().numpy(), 'scale_r2q': scale_r2q.detach().cpu().numpy()}
        # rotation
        if 'select_pr_angle' in results:
            angle_r2q = results['select_pr_angle'][torch.arange(qn),:,score_y,score_x] # qn,2
            angle = torch.atan2(angle_r2q[:,1],angle_r2q[:,0])
            outputs['angle_r2q'] = angle.cpu().numpy() # qn
        return outputs

    @staticmethod
    def detect_results_to_bbox(dets, length):
        pos = dets['position'] # qn,2
        length = dets['scale_r2q'] * length # qn,
        length = length[:,None]
        begin = pos - length/2
        return np.concatenate([begin,length,length],1)

    @staticmethod
    def detect_results_to_image_region(imgs, dets, region_len):
        qn = len(imgs)
        img_regions = []
        for qi in range(qn):
            pos = dets['position'][qi]; scl_r2q = dets['scale_r2q'][qi]
            ang_r2q = dets['angle_r2q'][qi] if 'anlge_r2q' in dets else 0
            img = imgs[qi]
            img_region, _ = transformation_crop(img, pos, 1/scl_r2q, -ang_r2q, region_len)
            img_regions.append(img_region)
        return img_regions

    @staticmethod
    def get_select_index(scores):
        """
        @param scores: qn,rfn or 1,hq,wq
        @return: qn
        """
        qn, rfn, hq, wq = scores.shape
        # 取最大值
        select_id = torch.argmax(scores.flatten(1), 1)
        select_ref_id = select_id // (hq * wq)
        select_h_id = (select_id - select_ref_id * hq * wq) // wq
        select_w_id = select_id - select_ref_id * hq * wq - select_h_id * wq
        return select_ref_id, select_w_id, select_h_id

    @staticmethod
    def parse_detection(scores, scales, offsets, pool_ratio):
        """
        @param scores:    qn,1,h/8,w/8
        @param scales:    qn,1,h/8,w/8
        @param offsets:   qn,2,h/8,w/8
        @param pool_ratio:int
        @return: position in x_cur
        """
        qn, _, _, _ = offsets.shape
        _, score_x, score_y = BaseDetector.get_select_index(scores) # qn
        positions = torch.stack([score_x, score_y], -1)  # qn,2
        offset = offsets[torch.arange(qn),:,score_y,score_x] # qn,2
        positions = positions + offset
        # to original coordinate
        positions = (positions + 0.5) * pool_ratio - 0.5 # qn,2
        # scale
        scales = scales[torch.arange(qn),0,score_y,score_x] # qn
        scales = 2**scales
        return positions, scales # [qn,2] [qn]

def disable_bn_grad(input_module):
    for module in input_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)

def disable_bn_track(input_module):
    for module in input_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

class Detector(BaseDetector):
    default_cfg={
        "vgg_score_stats": [[36.264317,13.151907],[13910.291,5345.965],[829.70807,387.98788]],
        "vgg_score_max": 10,
        "detection_scales": [-1.0,-0.5,0.0,0.5],
        "train_feats": False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__()
        self.backbone = VGGBNPretrain()
        self.use_dino = self.cfg.get("use_dino",False)
        logger.info(f"Detector use_dino: {self.use_dino}")
        if self.use_dino:
            self.fea_ext =  VitExtractor(model_name='dino_vits8').eval()

            for para in self.fea_ext.parameters():
                para.requires_grad = False

            # self.input_zero_conv = nn.Conv2d(in_channels=3, \
            #                             out_channels=3, \
            #                             kernel_size=1,\
            #                             stride=1,\
            #                             padding=0, \
            #                             bias=True)
            # self.input_zero_conv.weight.data.fill_(0)

            # self.zero_conv1 = nn.Conv2d(in_channels=512, \
            #                             out_channels=384, \
            #                             kernel_size=1,\
            #                             stride=1,\
            #                             padding=0, \
            #                             bias=True)

            # self.zero_conv1_re = nn.Conv2d(in_channels=384, \
            #                             out_channels=512, \
            #                             kernel_size=1,\
            #                             stride=1,\
            #                             padding=0, \
            #                             bias=True)

            # self.zero_conv1.weight.data.fill_(0)
            # self.zero_conv1_re.weight.data.fill_(0)

            # self.zero_conv2 = nn.Conv2d(in_channels=512, \
            #                             out_channels=384, \
            #                             kernel_size=1,\
            #                             stride=1,\
            #                             padding=0, \
            #                             bias=True)
            # self.zero_conv2_re = nn.Conv2d(in_channels=384, \
            #                             out_channels=512, \
            #                             kernel_size=1,\
            #                             stride=1,\
            #                             padding=0, \
            #                             bias=True)
            # self.zero_conv2.weight.data.fill_(0)
            # self.zero_conv2_re.weight.data.fill_(0)

            # self.zero_conv3 = nn.Conv2d(in_channels=512, \
            #                             out_channels=384, \
            #                             kernel_size=1,\
            #                             stride=1,\
            #                             padding=0, \
            #                             bias=True)

            # self.zero_conv3_re = nn.Conv2d(in_channels=384, \
            #                             out_channels=512, \
            #                             kernel_size=1,\
            #                             stride=1,\
            #                             padding=0, \
            #                             bias=True)

            # self.zero_conv3.weight.data.fill_(0)
            # self.zero_conv3_re.weight.data.fill_(0)

            self.fuse_conv1 = nn.Conv2d(in_channels=512+384, \
                                        out_channels=512, \
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

        if self.cfg["train_feats"]:
            # disable BN training only
            disable_bn_grad(self.backbone)
        else:
            for para in self.backbone.parameters():
                para.requires_grad = False
        self.pool_ratio = 8
        self.img_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        d = 64
        self.score_conv = nn.Sequential(
            nn.Conv3d(3*len(self.cfg['detection_scales']),d,1,1),
            nn.ReLU(),
            nn.Conv3d(d,d,1,1),
        )
        self.score_predict = nn.Sequential(
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,1,3,1,1),
        )
        self.scale_predict = nn.Sequential(
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,1,3,1,1),
        )
        self.offset_predict = nn.Sequential(
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,2,3,1,1),
        )
        self.ref_center_feats=None
        self.ref_shape=None

    def extract_feats(self, imgs):
        n,c,h,w = imgs.shape
        if self.use_dino:
            with torch.no_grad():
                # dino_ret =  self.fea_ext.get_vit_attn_feat(self.input_zero_conv(F.interpolate(imgs,size=(224,224))))
                dino_ret =  self.fea_ext.get_vit_attn_feat(F.interpolate(imgs,size=(224,224)))
                attn, cls_, feat = dino_ret['attn'], dino_ret['cls_'], dino_ret['feat']
                dino_fea = feat.permute(0,2,1).reshape(-1,384,224//8,224//8)
                dino_fea = F.interpolate(dino_fea,size=(h//8,w//8))
                dino_fea = F.normalize(dino_fea, dim=1)

        imgs = self.img_norm(imgs)
        if self.cfg['train_feats']:
            disable_bn_track(self.backbone)
            x0, x1, x2 = self.backbone(imgs)
        else:
            self.backbone.eval()
            with torch.no_grad():
                x0, x1, x2 = self.backbone(imgs)

        if self.use_dino:
            with torch.no_grad():
                # fuse v1
                fused_fea1 = torch.cat( (x0, dino_fea.clone() ), dim = 1)
                x0 = self.fuse_conv1(fused_fea1)
                dino_fea2 = F.interpolate(dino_fea.clone(), size=(h//16, w//16))
                fused_fea2 = torch.cat( (x1, dino_fea2 ), dim = 1)
                x1 = self.fuse_conv2(fused_fea2)
                dino_fea3 = F.interpolate(dino_fea.clone(), size=(h//32, w//32))
                fused_fea3 = torch.cat( (x2, dino_fea3 ), dim = 1)
                x2 = self.fuse_conv3(fused_fea3)

                # fuse v2
                # dino_fea_0 = self.zero_conv1(dino_fea)
                # x0 = x0 + dino_fea_0
                # dino_fea_1 = F.interpolate(dino_fea.clone(), size=(h//16, w//16))
                # x1 = x1 + dino_fea_1
                # dino_fea_2 = F.interpolate(dino_fea.clone(), size=(h//32, w//32))
                # dino_fea_2 = torch.cat( (x2, dino_fea_2 ), dim = 1)
                # x2 = self.fuse_conv3(dino_fea_2)


        return x0, x1, x2

    def load_impl(self, ref_imgs):
        # resize to 120,120
        ref_imgs = F.interpolate(ref_imgs,size=(120,120))
        # 15, 7, 3
        self.ref_center_feats = self.extract_feats(ref_imgs)
        rfn, _, h, w = ref_imgs.shape
        self.ref_shape = [h, w]

    def normalize_scores(self,scores0,scores1,scores2):
        stats = self.cfg['vgg_score_stats']
        scores0 = (scores0 - stats[0][0])/stats[0][1]
        scores1 = (scores1 - stats[1][0])/stats[1][1]
        scores2 = (scores2 - stats[2][0])/stats[2][1]

        scores0 = torch.clip(scores0,max=self.cfg['vgg_score_max'],min=-self.cfg['vgg_score_max'])
        scores1 = torch.clip(scores1,max=self.cfg['vgg_score_max'],min=-self.cfg['vgg_score_max'])
        scores2 = torch.clip(scores2,max=self.cfg['vgg_score_max'],min=-self.cfg['vgg_score_max'])

        return scores0, scores1, scores2

    def get_scores(self, que_imgs):

        que_x0, que_x1, que_x2 = self.extract_feats(que_imgs)

        ref_x0, ref_x1, ref_x2 = self.ref_center_feats # rfn,f,hr,wr
        scores2 = F.conv2d(que_x2, ref_x2, padding=1)
        scores1 = F.conv2d(que_x1, ref_x1, padding=3)
        scores0 = F.conv2d(que_x0, ref_x0, padding=7)
        scores2 = F.interpolate(scores2, scale_factor=4)
        scores1 = F.interpolate(scores1, scale_factor=2)
        scores0, scores1, scores2 = self.normalize_scores(scores0, scores1, scores2)
        scores = torch.stack([scores0, scores1, scores2],1) # qn,3,rfn,hq/8,wq/8
        return scores

    def detect_impl(self, que_imgs):
        global count
        qn, _, hq, wq = que_imgs.shape
        hs, ws = hq // 8, wq // 8
        scores = []
        for scale in self.cfg['detection_scales']:
            ht, wt = int(np.round(hq*2**scale)), int(np.round(wq*2**scale))
            if ht%32!=0: ht=(ht//32+1)*32
            if wt%32!=0: wt=(wt//32+1)*32
            que_imgs_cur = F.interpolate(que_imgs,size=(ht,wt),mode='bilinear')
            scores_cur = self.get_scores(que_imgs_cur)
            qn, _, rfn, hcs, wcs = scores_cur.shape
            scores.append(F.interpolate(scores_cur.reshape(qn,3*rfn,hcs,wcs),size=(hs,ws),mode='bilinear').reshape(qn,3,rfn,hs,ws))

        scores = torch.cat(scores, 1) # qn,sn*3,rfn,hq/8,wq/8
        scores = self.score_conv(scores)
        scores_feats = torch.max(scores,2)[0] # qn,f,hq/8,wq/8
        scores = self.score_predict(scores_feats) # qn,1,hq/8,wq/8

        # predict offset and bbox
        _, select_w_id, select_h_id = self.get_select_index(scores)
        que_select_id = torch.stack([select_w_id, select_h_id],1) # qn, 2

        select_offset = self.offset_predict(scores_feats)  # qn,1,hq/8,wq/8
        select_scale = self.scale_predict(scores_feats) # qn,1,hq/8,wq/8
        # count +=1

        outputs = {
                   'scores': scores,\
                   'que_select_id': que_select_id, \
                   'pool_ratio': self.pool_ratio, \
                   'select_pr_offset': select_offset,\
                   'select_pr_scale': select_scale
        }

        return outputs

    def forward(self, data):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()

        ref_imgs = ref_imgs_info['imgs']
        self.load_impl(ref_imgs)
        outputs = self.detect_impl(que_imgs_info['imgs'])
        return outputs

    def load_ref_imgs(self, ref_imgs):
        """
        @param ref_imgs: [an,rfn,h,w,3] in numpy
        @return:
        """
        ref_imgs = torch.from_numpy(color_map_forward(ref_imgs)).permute(0,3,1,2).contiguous() # rfn,3,h,w
        ref_imgs = ref_imgs.cuda()
        rfn, _, h, w = ref_imgs.shape
        self.load_impl(ref_imgs)

    def detect_que_imgs(self, que_imgs):
        """
        @param que_imgs: [qn,h,w,3]
        @return:
        """
        que_imgs = torch.from_numpy(color_map_forward(que_imgs)).permute(0,3,1,2).contiguous().cuda()
        qn, _, h, w = que_imgs.shape
        outputs = self.detect_impl(que_imgs)

        positions, scales = self.parse_detection(
                            outputs['scores'].detach(),
                            outputs['select_pr_scale'].detach(),
                            outputs['select_pr_offset'].detach(),
                            self.pool_ratio)

        detection_results = {'positions': positions, 'scales': scales}
        detection_results = to_cpu_numpy(detection_results)
        return detection_results

if __name__ == "__main__":
    # mock_data = torch.randn(6,3,128,128)
    # default_cfg = {
    #     'selector_angle_num': 5,
    # }
    # net = Detector(default_cfg)
    # out =  net.extract_feats(mock_data)
    # print(len(out), out[0].shape, out[1].shape, out[2].shape  )

    from gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()
    dino = VitExtractor(model_name='dino_vits8').eval().cuda()
    data = torch.Tensor(1,3,224,224).cuda()
    import time
    t1 = time.time()
    gpu_tracker.track()
    for i in range(100):
        dino.get_vit_attn_feat(data)
    t2 = time.time()
    gpu_tracker.track()
    print("t2-t1:", (t2-t1)/100 *1000)

