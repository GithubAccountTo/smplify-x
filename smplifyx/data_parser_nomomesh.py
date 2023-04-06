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

import sys
import os
import os.path as osp
import glob
import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose
from data_parser_nomomesh_utils import humanpose2smplpose, get_mesh_vertices
from camera import create_camera

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'keypoints3d', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    cam_id = data['cam_id']
    keypoints = []
    keypoints3d = []
    gender_pd = []
    gender_gt = []
   

    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'], dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        body_keypoints3d = np.array(person_data['pose_keypoints_3d'], dtype=np.float32)
        # body_keypoints3d = body_keypoints3d[:,2] - 0.4
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)
        keypoints3d.append(body_keypoints3d)

    return Keypoints(keypoints=keypoints, keypoints3d=keypoints3d,
                    gender_pd=gender_pd, gender_gt=gender_gt), cam_id


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 mesh_folder='mesh',
                 use_hands=False,
                 use_face=False,
                 use_3dpose=False,
                 use_mesh=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.use_3dpose = use_3dpose
        self.use_mesh = use_mesh
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)
        self.data_folder = data_folder
        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)
        self.mesh_folder = osp.join(data_folder.rsplit("/",1)[0], mesh_folder)
        self.mesh_paths = glob.glob(self.mesh_folder+'/*.obj')

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') or
                          img_fn.endswith('.jpeg') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)
        self.mesh_paths = sorted(self.mesh_paths)
        self.cams_data = json.load(open(kwargs['calibration_file']))
        self.cam_obj = create_camera(focal_length_x=torch.tensor(kwargs['focal_length']),
                                    focal_length_y=torch.tensor(kwargs['focal_length']),**kwargs)
        self.cam_obj.rotation.requires_grad = False
        
        self.cnt = 0
        self.use_cuda = kwargs['use_cuda']

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx): # 假定实例为p,则使用p[key]取值时，自动调用该函数
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, mesh_path):
        # mesh_path: data_monomesh/1114/mesh/1672715666764.obj
        # img_path: data_monomesh/1114/1_0_1114/images/1_0_1114_1672715666764.jpeg
        img_path = osp.join(self.img_folder, self.data_folder.rsplit("/",1)[-1] + '_' +
        mesh_path.rsplit("/",1)[-1].replace('obj', 'jpeg')) 
        
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])
        # keypoint_fn: data_monomesh/1114/1_0_1114/keypoints_3d/1_0_1114_1672715666764_openpose.json
        keypoint_fn = osp.join(self.keyp_folder,
                               img_fn + '_openpose.json')
        keyp_tuple, cam_id = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1 or len(keyp_tuple.keypoints3d) < 1:
            return {}
        
        # pose转换，humanpose to smplpose
        keypoints, keypoints3d = humanpose2smplpose(keyp_tuple, use_hands=self.use_hands) # keypoints: array(1,25,3)
                                                                # keypoints3d:  array(1,25,4)
        # mesh gt 引入
        mesh = get_mesh_vertices(mesh_path) # mesh:array(6890, 3)
        # 相机参数读取
        # self.cam_obj.focal_length_x = torch.tensor(self.cams_data[cam_id]['K'][0])
        # self.cam_obj.focal_length_y = torch.tensor(self.cams_data[cam_id]['K'][4])
        # self.cam_obj.center = torch.tensor([self.cams_data[cam_id]['K'][2],self.cams_data[cam_id]['K'][5]]).unsqueeze(0).to(self.dtype)
        # self.cam_obj.rotation.data = torch.from_numpy(np.array(self.cams_data[cam_id]['RT']).reshape(3,-1)[:,:3]).unsqueeze(0).to(self.dtype)
        # self.cam_obj.translation.data = torch.from_numpy(np.array(self.cams_data[cam_id]['RT']).reshape(3,-1)[:,3]).unsqueeze(0).to(self.dtype)
        self.cam_obj.rotation.requires_grad = False
        # self.cam_obj.translation.requires_grad = False
        if self.use_cuda:
            self.cam_obj=self.cam_obj.to(torch.device('cuda'))


        # 返回的数据
        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints, 
                       'keypoints3d': keypoints3d,
                       'mesh': mesh,
                       'cam':self.cam_obj,
                       'img': img}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.mesh_paths):
            raise StopIteration

        mesh_path = self.mesh_paths[self.cnt]
        self.cnt += 1

        return self.read_item(mesh_path)
