# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# debug
import ptvsd
ptvsd.enable_attach(address=("172.17.0.5",  9024))
ptvsd.wait_for_attach()

import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import os.path as osp

import time
import yaml
import torch

import smplx

from utils import JointMapper
from cmd_parser import parse_config
from data_parser_nomomesh import create_dataset
from fit_single_frame_monomesh import fit_single_frame

# from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False



def main(**args):
    output_folder = args.pop('output_folder') # pop用于获得指定的key对应的value，并删除key-value
    output_folder = osp.expandvars(output_folder) # 用于扩展路径
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file) # python object生成yaml文档

    result_folder = args.pop('result_folder', 'results') # 
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes') # 若存在，则返回args['mesh_folder'],否则返回默认值meshes
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32 
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True) # 字典用法：返回args['use_cuda'],若不存在args['use_cuda']，返回default值
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.get('img_folder', 'images')
    dataset_obj = create_dataset(**args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    # Create the camera object
    # focal_length = args.get('focal_length')
    # camera = create_camera(focal_length_x=focal_length,
    #                        focal_length_y=focal_length,
    #                        dtype=dtype,
    #                        **args)
    # hasattr(object, name):object中是否有name属性，有，返回true，无，返回false
    # if hasattr(camera, 'rotation'):
    #     camera.rotation.requires_grad = False

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)
    use_3dpose = args.get('use_3dpose', True)
    use_mesh = args.get('use_mesh', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    sli_prior = create_prior(
        prior_type= 'l2',
        dtype = dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        # camera = camera.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        sli_prior = sli_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)
    for idx, data in enumerate(dataset_obj):
        img = data['img']
        fn = data['fn']
        keypoints = data['keypoints']
        keypoints3d = data['keypoints3d']
        mesh = data['mesh']
        print('Processing: {}'.format(data['img_path']))

        curr_result_folder = osp.join(result_folder, fn)
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        
        # curr_mesh_folder = osp.join(mesh_folder, fn)
        # if not osp.exists(curr_mesh_folder):
        #     os.makedirs(curr_mesh_folder)

        curr_mesh_folder = mesh_folder

        for person_id in range(keypoints.shape[0]):
            if person_id >= max_persons and max_persons > 0:
                continue

            curr_result_fn = osp.join(curr_result_folder,
                                      '{:03d}.pkl'.format(person_id))
            # curr_mesh_fn = osp.join(curr_mesh_folder,
            #                         '{:03d}.obj'.format(person_id))

            curr_mesh_fn = osp.join(curr_mesh_folder, fn + '.obj')

            curr_img_folder = osp.join(output_folder, 'images', fn,
                                       '{:03d}'.format(person_id))
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)

            if gender_lbl_type != 'none':
                if gender_lbl_type == 'pd' and 'gender_pd' in data:
                    gender = data['gender_pd'][person_id]
                if gender_lbl_type == 'gt' and 'gender_gt' in data:
                    gender = data['gender_gt'][person_id]
            else:
                gender = input_gender

            if gender == 'neutral':
                body_model = neutral_model
            elif gender == 'female':
                body_model = female_model
            elif gender == 'male':
                body_model = male_model

            out_img_fn = osp.join(curr_img_folder, 'output.png')

            args['fn'] = fn
            
            fit_single_frame(img, keypoints[[person_id]],
                             keypoints3d=keypoints3d[[person_id]],
                             mesh = mesh,
                             body_model=body_model,
                             camera=data['cam'],
                             joint_weights=joint_weights,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             out_img_fn=out_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             sli_prior=sli_prior,
                             expr_prior=expr_prior,
                             body_pose_prior=body_pose_prior,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             angle_prior=angle_prior,
                             **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    args['img_folder'] = 'images'
    args['keyp_folder'] = 'keypoints_3d'
    args['focal_length'] = 1820.0 
    args['data_folder'] = 'data_nomomesh/20230221/1348/1_0_1348'
    args['output_folder'] = 'out_nomomesh/20230221/1348_1'
    # args['use_joints_conf'] = False
    # new
    args['use_mesh'] = True
    args['use_3dpose'] = False
    args['mesh_folder'] = 'mesh'
    args['calibration_file'] = 'data_nomomesh/20230221/calibration_20230221.json' 

    args['mesh_loss_weights'] = [0.0, 0.0, 0.0, 0.0, 0.0]
    args['sil_loss_weights'] = [0.0, 0.0, 0.0, 0.0, 0.0]


    main(**args)
    
