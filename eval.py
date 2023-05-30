# run --type v100-32g --cpu 30 --memory 150 -- python3 eval.py
import argparse
from copy import copy
from pathlib import Path
import numpy as np
from skimage.io import imsave
import cv2
from tqdm import tqdm
from dataset.database import parse_database_name, get_database_split, get_ref_point_cloud, get_diameter, get_object_center
from estimator import name2estimator
from utils.base_utils import load_cfg, save_pickle, read_pickle, project_points, transformation_crop
from utils.database_utils import compute_normalized_view_correlation
from utils.draw_utils import draw_bbox, concat_images_list, draw_bbox_3d, pts_range_to_bbox_pts
from utils.pose_utils import compute_metrics_impl, scale_rotation_difference_from_cameras
from utils.pose_utils import compute_pose_errors
from tabulate import tabulate
from loguru import logger

def get_gt_info(que_pose, que_K, render_poses, render_Ks, object_center):
    gt_corr = compute_normalized_view_correlation(que_pose[None], render_poses, object_center, False)
    gt_ref_idx = np.argmax(gt_corr[0])
    gt_scale_r2q, gt_angle_r2q = scale_rotation_difference_from_cameras(
        render_poses[gt_ref_idx][None], que_pose[None], render_Ks[gt_ref_idx][None], que_K[None], object_center)
    gt_scale_r2q, gt_angle_r2q = gt_scale_r2q[0], gt_angle_r2q[0]
    gt_position = project_points(object_center[None], que_pose, que_K)[0][0]
    size = 128
    gt_bbox = np.concatenate([gt_position - size / 2 * gt_scale_r2q, np.full(2, size) * gt_scale_r2q])

    return gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_corr[0]

def visualize_intermediate_results(object_pts, object_diameter , img, K, inter_results, ref_info, object_bbox_3d, \
                                    object_center=None, pose_gt=None, est_name="",object_name="", que_id=-1 ):
    ref_imgs = ref_info['ref_imgs']  
    if pose_gt is not None:
        gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_scores = \
            get_gt_info(pose_gt, K, ref_info['poses'], ref_info['Ks'], object_center)
    img_h , img_w , _ = img.shape
    output_imgs = []

    pts2d_gt, _ = project_points(object_pts, pose_gt, K)
    x0, y0, w, h = cv2.boundingRect(pts2d_gt.astype(np.int32))
    img_h, img_w, c = img.shape
    max_r = max(w,h)
    x1, y1 = min(x0 + 1.5*max_r,img_w), min(y0 + 1.5*max_r,img_h)
    x0, y0 = max(x0 - 0.5*max_r, 0), max(y0 - 0.5*max_r, 0)

    if 'det_scale_r2q' in inter_results and 'sel_angle_r2q' in inter_results:
        det_scale_r2q = inter_results['det_scale_r2q']
        det_position = inter_results['det_position']
        det_que_img = inter_results['det_que_img']
        size = det_que_img.shape[0]
        pr_bbox = np.concatenate([det_position - size / 2 * det_scale_r2q, np.full(2, size) * det_scale_r2q])
        pr_bbox[0] =  int(pr_bbox[0] )
        pr_bbox[1] =  int(pr_bbox[1] )
        pr_bbox[2] =  int(pr_bbox[2])
        pr_bbox[3] =  int(pr_bbox[3])
        max_r = max(pr_bbox[2], pr_bbox[3] )
        bbox_img = img
        bbox_img = draw_bbox(bbox_img, pr_bbox, color=(0, 0, 255))
        x0,y0,x1,y1 = pr_bbox[0], pr_bbox[1],  pr_bbox[0] + pr_bbox[2],  pr_bbox[1] +  pr_bbox[3]
        x1, y1 = int(min(x0 + 1.5*max_r,img_w)), int(min(y0 + 1.5*max_r,img_h))
        x0, y0 = int(max(x0 - 0.5*max_r, 0)), int(max(y0 - 0.5*max_r, 0))
        if pose_gt is not None: bbox_img = draw_bbox(bbox_img, gt_bbox, color=(0, 255, 0))
        crop_img = bbox_img[y0:y1, x0:x1,:]
        imsave(f'data/vis_final/{est_name}/{object_name}/{que_id}-bbox2d.jpg',bbox_img )
        imsave(f'data/vis_final/{est_name}/{object_name}/{que_id}-bbox2d-crop.jpg', cv2.resize(crop_img, (512, 512)) )   
        output_imgs.append(bbox_img)
        # visualize selection
        sel_angle_r2q = inter_results['sel_angle_r2q']  #
        sel_scores = inter_results['sel_scores']  #
        h, w, _ = det_que_img.shape
        sel_img_rot, _ = transformation_crop(det_que_img, np.asarray([w / 2, h / 2], np.float32), 1.0, -sel_angle_r2q, h)
        an = ref_imgs.shape[0]
        sel_img = concat_images_list(det_que_img, sel_img_rot, *[ref_imgs[an // 2, score_idx] for score_idx in np.argsort(-sel_scores)[:5]], vert=True)
        if pose_gt is not None:
            sel_img_rot_gt, _ = transformation_crop(det_que_img, np.asarray([w/2, h/2], np.float32), 1.0, -gt_angle_r2q, h)
            sel_img_gt = concat_images_list(det_que_img, sel_img_rot_gt, *[ref_imgs[an // 2, score_idx] for score_idx in np.argsort(-gt_scores)[:5]], vert=True)
            sel_img = concat_images_list(sel_img, sel_img_gt)
        output_imgs.append(sel_img)
    # visualize refinements
    refine_poses = inter_results['refine_poses'] if 'refine_poses' in inter_results else []
    refine_imgs = []
    # refine pose 打印出来
    for k in range(1,len(refine_poses)):
        pose_in, pose_out = refine_poses[k-1], refine_poses[k]
        bbox_pts_in, _ = project_points(object_bbox_3d, pose_in, K)
        bbox_pts_out, _ = project_points(object_bbox_3d, pose_out, K)
        prj_err, obj_err, pose_err = compute_pose_errors(object_pts, pose_out, pose_gt, K)
        is_add01 = obj_err>0.1*object_diameter
        img_render = img.copy()
        if is_add01: # bgr
            img_render = draw_bbox_3d(img_render, bbox_pts_out, (0, 0, 255) )
        else:
            img_render = draw_bbox_3d(img_render.copy(), bbox_pts_out, (0, 0, 255))
        if pose_gt is not None:
            bbox_pts_gt, _ = project_points(object_bbox_3d, pose_gt, K)
            img_render = draw_bbox_3d(img_render, bbox_pts_gt, (0, 255, 0))
        crop_img = img_render[y0:y1, x0:x1,:]
        imsave(f'data/vis_final/{est_name}/{object_name}/{que_id}-refiner-{k}-crop.jpg', cv2.resize(crop_img, (512, 512)) )   
        output_imgs.append(bbox_img)
        refine_imgs.append(bbox_img)

    if len(refine_poses)!=0:
        output_imgs.append(concat_images_list(*refine_imgs))
        # cv2.putText(output_imgs[-1], str(is_add01) , (0, 0), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)
    return concat_images_list(*output_imgs)

def visualize_final_poses(object_pts, object_diameter , img, K, object_bbox_3d, pose_pr, pose_gt=None):
    bbox_pts_pr, _ = project_points(object_bbox_3d, pose_pr, K)

    pts2d_pr, _ = project_points(object_pts, pose_pr, K)
    prj_err, obj_err, pose_err = compute_pose_errors(object_pts, pose_pr, pose_gt, K)
    bbox_img = img
    point_size = 1
    thickness = 1
    pts2d_gt, _ = project_points(object_pts, pose_gt, K)
    x0, y0, w, h = cv2.boundingRect(pts2d_gt.astype(np.int))
    img_h, img_w, c = img.shape
    max_r = max(w,h)
    x1, y1 = min(x0 + 1.5*max_r,img_w), min(y0 + 1.5*max_r,img_h)
    x0, y0 = max(x0 - 0.5*max_r, 0), max(y0 - 0.5*max_r, 0)
    
    alpha = 0.5
    if pose_gt is not None:
        bbox_pts_gt, _ = project_points(object_bbox_3d, pose_gt, K)
        bbox_img = draw_bbox_3d(bbox_img, bbox_pts_gt)
    
    if obj_err>0.1*object_diameter:
        bbox_img = draw_bbox_3d(bbox_img, bbox_pts_pr, (0, 0, 255))
        return bbox_img,[x0,y0,x1,y1]
    else:
        bbox_img = draw_bbox_3d(bbox_img, bbox_pts_pr, (0, 0, 255))
        return bbox_img,[x0,y0,x1,y1]
    return None, None
    return bbox_img

import time
def main(args):
    # estimator
    cfg = load_cfg(args.cfg)
    estimator = name2estimator[cfg['type']](cfg)
    
    # object_name_list = [  "linemod/cat", "linemod/duck", "linemod/benchvise", \
    #                       "linemod/cam",  "linemod/driller", \
    #                       "linemod/lamp",  "linemod/eggbox", "linemod/glue"]

    # object_name_list = [ "linemod/cat", "linemod/duck" ]
    object_name_list = ["genmop/chair", "genmop/plug_en", "genmop/piggy",  \
                        "genmop/scissors", "genmop/tformer"]

    t1 = time.time()
    logger.debug(f"use_gt_box2d:{ args.use_gt_box } , use_refiner:{ args.use_refiner }")
    metric_list = list()
    add_list, proj5_list = list() , list()
    for object_name in object_name_list:
        if "eggbox" in object_name or  "glue" in object_name:
            symmetric = True
        else:
            symmetric = False
        
        if object_name.startswith('linemod'):
            ref_database_name = que_database_name = object_name
            que_split = 'linemod_test'
        elif object_name.startswith('genmop'):
            ref_database_name = object_name+'-ref'
            que_database_name = object_name+'-test'
            que_split = 'all'
        else:
            raise NotImplementedError

        ref_database = parse_database_name(ref_database_name)
        ref_split = que_split if args.split_type is None else args.split_type
        estimator.build(ref_database, split_type=ref_split)
        que_database = parse_database_name(que_database_name)
        _, que_ids = get_database_split(que_database, que_split)
        object_pts = get_ref_point_cloud(ref_database)
        object_center = get_object_center(ref_database)
        object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))
        est_name = estimator.cfg["name"] # + f'-{args.render_pose_name}'
        est_name = est_name + args.split_type if args.split_type is not None else est_name
        est_name = "DEBUG"
        Path(f'data/eval/poses/{object_name}').mkdir(exist_ok=True,parents=True)
        Path(f'data/vis_inter/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)
        Path(f'data/vis_final/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)
        # evaluation metrics
        object_diameter = get_diameter(que_database)
        if not args.eval_only:
            pose_pr_list = []
            new_que_ids = []
            print(f"obj number =  {len(que_ids)}")
            for idx, que_id in enumerate(tqdm(que_ids)):
                new_que_ids.append(que_id)
                # estimate pose
                img = que_database.get_image(que_id)
                K = que_database.get_K(que_id)
                pose_gt = que_database.get_pose(que_id)
                if args.use_gt_box:
                    gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_scores = \
                    get_gt_info(pose_gt, K, estimator.ref_info['poses'], estimator.ref_info['Ks'], object_center)
                    
                    pose_pr, inter_results = estimator.predict(img, K, position = gt_position, \
                                                                       scale_r2q = gt_scale_r2q, \
                                                                       need_refiner = args.use_refiner)
                else:
                    pose_pr, inter_results = estimator.predict(img, K, need_refiner = args.use_refiner)        
                    pose_pr_list.append(pose_pr)

                final_img, bbox2d = visualize_final_poses(object_pts , object_diameter, img, K, object_bbox_3d, pose_pr, pose_gt)                
                if final_img is not None and visualize_final_poses is not None:
                    x0, y0, x1, y1 = [int(x) for  x in bbox2d]
                    crop_img = final_img[y0:y1, x0:x1,:]
                    imsave(f'data/vis_final/{est_name}/{object_name}/{idx}-bbox3d.jpg', final_img)
                    imsave(f'data/vis_final/{est_name}/{object_name}/{idx}-bbox3d-crop.jpg', \
                           cv2.resize(crop_img, (512, 512)) )   

        pose_gt_list = [que_database.get_pose(que_id) for que_id in new_que_ids]
        que_Ks = [que_database.get_K(que_id) for que_id  in new_que_ids]
       
        def get_eval_msg(pose_in_list,msg_in,scale=1.0):
            msg_in = copy(msg_in)
            results = compute_metrics_impl(object_pts, object_diameter, pose_gt_list, pose_in_list, \
                                           que_Ks, scale, symmetric = symmetric)
            for k, v in results.items(): msg_in+=f'{k} {v:.4f} '
            return msg_in + '\n', results
        msg_pr = f'{object_name:10} {est_name:20} '
        msg_pr, results = get_eval_msg(pose_pr_list, msg_pr)
        add , prj5 = results['add-0.1d'], results['prj-5']
        if symmetric:
            add = results['add-0.1d-sym']
        add_list.append(add), proj5_list.append(prj5)
        print(object_name +  ": ,  add0.1:", add , " ,proj5:", prj5 )
        metric_list.append(  (object_name, add , prj5 )  )
        with open('data/performance.log','a') as f: f.write(msg_pr)

    print("avg add0.1:", sum(add_list)/len(add_list) , " , avg proj5:", sum(proj5_list)/len(proj5_list))
    metric_list.append(  ("avg", sum(add_list)/len(add_list) , sum(proj5_list)/len(proj5_list) )  )
    print(tabulate(metric_list, headers=['objname', 'add0.1', 'proj5'],tablefmt='fancy_grid'))
    t2 = time.time()
    print(f"[debug]: the exp costs {t2-t1} seconds")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="configs/cas6d_train.yaml")
    parser.add_argument('--object_name', type=str, default='warrior')
    parser.add_argument('--eval_only', action='store_true', dest='eval_only', default=False)
    parser.add_argument('--symmetric', action='store_true', dest='symmetric', default=False)
    parser.add_argument('--split_type', type=str, default=None)
    parser.add_argument('--use_gt_box',action='store_true', dest='use_gt_box', default=False)
    parser.add_argument('--use_refiner',action='store_true', dest='use_refiner', default=True)
    args = parser.parse_args()
    main(args)


