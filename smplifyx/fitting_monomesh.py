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

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils

from monomesh.lossTools import compute_similarity_transform_batch

@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               gt_joints3d=None,
                               joints3d_conf=None,
                               gt_mesh=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              gt_joints3d=gt_joints3d,
                              joints3d_conf=joints3d_conf,
                              gt_mesh=gt_mesh,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph= create_graph)  # create_graph

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


def perspectiveprojectionnp(fovy, ratio=1.0, near=0.3, far=750.0):
    tanfov = np.tan(fovy / 2.0)
    return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]], dtype=np.float32)

from math import radians
def render_silhouette(renderer, vertices,faces,PdroneToCamera,H,W, b=1):
    """Set up renderer for visualising drone based on the predicted pose
        PdroneToCamera: relative pose predicted by network or GT
    """
    vertices = vertices.expand(b,vertices.size(1),vertices.size(2))
    device = vertices.get_device()
    # vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
    # vert_max = torch.max(vertices, dim=1, keepdims=True)[0]

    #setup color
    colors = torch.zeros(vertices.size()).to(device)
    colors[:,:,:3] = 1
    
    #get homogeneous coordinates for vertices
    vertices = torch.torch.nn.functional.pad(vertices[:,:,],[0,1],"constant",1.0)
    vertices = vertices.transpose(2,1)

    vertices = torch.matmul(PdroneToCamera,vertices)
    vertices = vertices.transpose(2,1)
    # pro_vertices = torch.div(vertices[:, :, :3],
    #                            vertices[:, :, 2].unsqueeze(dim=-1))
    # pro_vertices[:,:,2] = vertices[:, :, 2]
    b , _ ,_ = PdroneToCamera.size()

    #set camera parameters
    cameras = []
    camera_rot_bx3x3 = torch.zeros((b,3,3),dtype=torch.float32).to(device)
    camera_rot_bx3x3[:,0,1] = -1
    camera_rot_bx3x3[:,1,0] = 1
    camera_rot_bx3x3[:,2,2] = 1
    
    cameras.append(camera_rot_bx3x3)
    camera_pos_bx3 = torch.zeros((b,3),dtype=torch.float32).to(device)
    # camera_pos_bx3 =  PdroneToCamera[:,:3,3]
    cameras.append(camera_pos_bx3)

    camera_proj_3x1 = torch.zeros((3,1),dtype=torch.float32).to(device)
    fov_y = 2 * np.arctan2(float(H), 2 * 5000)
    camera_proj_3x1[:,:] = torch.from_numpy(perspectiveprojectionnp(fov_y))  # ratio= float(W)/float(W), near=0.05, far=100.0
    cameras.append(camera_proj_3x1)

    # print(cameras)

    renderer.set_camera_parameters(cameras)

    #convert points from homogeneous
    # z_vec =  vertices[..., -1:]
    # scale = torch.tensor(1.) / torch.clamp(z_vec, 0.000000001)

    vertices =  vertices[..., :-1]

    #forward pass
    predictions , mask , _ = renderer(points=[vertices,faces.long()],colors_bxpx3=colors)

    return mask #, mask

def gene_mask(pro_vertices, faces, height, width):
    device = pro_vertices.device
    mask = torch.zeros(height, width, dtype= torch.float32, requires_grad= True)
    mask = mask.to(device= device)
    # print(mask.device)
    x_f0 = pro_vertices[:, faces[:, 0], 0].unsqueeze(2)
    y_f0 = pro_vertices[:, faces[:, 0], 1].unsqueeze(2)
    x_f1 = pro_vertices[:, faces[:, 1], 0].unsqueeze(2)
    y_f1 = pro_vertices[:, faces[:, 1], 1].unsqueeze(2)
    x_f2 = pro_vertices[:, faces[:, 2], 0].unsqueeze(2)
    y_f2 = pro_vertices[:, faces[:, 2], 1].unsqueeze(2)
    x_f012 = torch.cat((x_f0, x_f1, x_f2), dim=2)
    y_f012 = torch.cat((y_f0, y_f1, y_f2), dim=2)
    x_min,_ = torch.min(x_f012, 2)
    xmin,_ = torch.min(x_min, 1)
    # x_min = x_min.unsqueeze(2)
    x_max,_ = torch.max(x_f012, 2)
    xmax,_ = torch.max(x_max,1)
    # x_max = x_max.unsqueeze(2)
    y_min,_ = torch.min(y_f012, 2)
    ymin,_ = torch.min(y_min,1)
    # y_min = y_min.unsqueeze(2)
    y_max,_ = torch.max(y_f012, 2)
    ymax = torch.max(y_max,1)
    # y_max = y_max.unsqueeze(2)

    xy_f0 = torch.cat((x_f0,y_f0), 2)
    xy_f1 = torch.cat((x_f1,y_f1), 2)
    xy_f2 = torch.cat((x_f2,y_f2), 2)
    
    for idx in range(x_min.shape[1]):
        x = torch.range(start=x_min[0,idx], end=x_max[0, idx]).unsqueeze(1).to(device=device)
        y = torch.range(start= y_min[0, idx], end= y_max[0, idx]).unsqueeze(1).to(device=device)
        x_len = x.shape[0]
        y_len = y.shape[0]
        if x_len*y_len==0:
            continue

        x = x.repeat(y_len,1 )
        y = y.repeat(x_len, 1).reshape(x_len, -1).t().reshape(-1,1)

        # 判断点是否在三角形内部
        v01 = xy_f1[0,idx,:] - xy_f0[0,idx,:]
        v01 = v01.repeat(x_len*y_len, 1)
        v02 = xy_f2[0,idx,:] - xy_f0[0,idx,:]
        v02 = v02.repeat(x_len*y_len, 1)
        v03 = xy - xy_f0[0,idx, :].repeat(x_len*y_len, 1)
        judge1 = (v01[:,0]*v02[:,1] - v01[:,1]*v02[:,0]) * (v01[:,0]*v03[:,1] - v01[:,1]*v03[:,0]) #18

        xy = torch.cat((x,y),1) #18*2
        v20 = xy_f0[0,idx,:] - xy_f2[0,idx,:]
        v20 = v20.repeat(x_len*y_len, 1)
        v21 = xy_f1[0,idx,:] - xy_f2[0,idx,:]
        v21 = v21.repeat(x_len*y_len, 1)
        v23 = xy - xy_f2[0,idx, :].repeat(x_len*y_len, 1)
        judge2 = (v21[:,0]*v20[:,1] - v21[:,1]*v20[:,0]) * (v21[:,0]*v23[:,1] - v21[:,1]*v23[:,0]) #18
        
        # 点在三角形内部，则将mask对应位置置1
        judge = judge1 * judge2
        for i in range(judge.shape[0]):
            if (judge[i]>0.0 and xy[i,1].item()>=0 and xy[i,1].item()<height and xy[i,0].item()>=0 and xy[i,0].item()<width) :
                mask[xy[i, 1].type(torch.long), xy[i, 0].type(torch.long)] = 1.0 


    return mask


import torch
import math
import torch.nn as nn

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter





class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 sli_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 sil_loss_weight=0.0,
                 mesh_loss_weight=0.0,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_3dpose = kwargs.get('use_3dpose')
        self.use_mesh = kwargs.get('use_mesh')
       

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior
       
        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior
        self.sli_prior = sli_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))
        self.register_buffer('sil_loss_weight',
                                 torch.tensor(sil_loss_weight, dtype=dtype))
        self.register_buffer('mesh_loss_weight',
                                 torch.tensor(sil_loss_weight, dtype=dtype))
        self.iter = 0

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    
    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                gt_joints3d, joints3d_conf,gt_mesh,
                use_vposer=False, pose_embedding=None,
                **kwargs):
        device = joint_weights.device
        projected_joints = camera(body_model_output.joints)

        # 1 joint_loss
        joint_loss = 0.0
        if self.data_weight.item() > 0:
            # Calculate the weights for each joints
            weights = (joint_weights * joints_conf
                    if self.use_joints_conf else
                    joint_weights).unsqueeze(dim=-1)
            # Calculate the distance of the projected joints from
            # the ground truth 2D detections
            joint_diff = self.robustifier(gt_joints - projected_joints)
            joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                        self.data_weight ** 2) # data_weight = 0.833

        # 2 Calculate the loss from the Pose prior
        pprior_loss = 0.0
        if use_vposer and self.body_pose_weight.item() > 0:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2) # body_pose_weight = 404
        elif self.body_pose_weight.item() > 0:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        # 3 shape_loss
        shape_loss = 0.0
        if self.shape_weight.item() > 0:
            shape_loss = torch.sum(self.shape_prior(
                body_model_output.betas)) * self.shape_weight ** 2
        # 4 Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        angle_prior_loss = 0.0
        if self.bending_prior_weight.item() > 0:
            body_pose = body_model_output.full_pose[:, 3:66]
            angle_prior_loss = torch.sum(
                self.angle_prior(body_pose)) * self.bending_prior_weight # bending_prior_weight = 1280.68

        # 5 Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None and self.hand_prior_weight.item() > 0:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None and self.hand_prior_weight.item() > 0:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2  # hand_prior_weight = 404

       # 6 expression_loss and jaw_loss
        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face and self.expr_prior_weight.item() > 0:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2  # expr_prior_weight = 100

            if hasattr(self, 'jaw_prior') and self.jaw_prior_weight.item() > 0:
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight))) # jaw_prior_weight = [4040, 40400, 40400]

        # 7 Calculate the loss due to interpenetration
        pen_loss = 0.0
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *  # coll_loss_weight = 0
                    self.pen_distance(triangles, collision_idxs))

        ##################OPENCV 输出mask######################################
        # import cv2
        # import time
        # maskGTPath = '/mnt/dy_data/smplify-x-master/data/segment/' + '1461_seg.jpg'
        # maskGT = cv2.imread(maskGTPath)
        # mask = np.zeros(maskGT.shape, np.uint8)
        # points = camera(body_model_output.vertices)
        
        # start = time.clock()
        # for i in range(int(body_model_faces.shape[0]/3)):
        #     triangle = np.zeros((3, 2), dtype= int)
        #     j = body_model_faces[i*3 + 0].item()
        #     triangle[0][:] = round(points[0,j,0].item()), round(points[0,j,1].item())
        #     j = body_model_faces[i*3 + 1].item()
        #     triangle[1][:] = round(points[0,j,0].item()), round(points[0,j,1].item())
        #     j = body_model_faces[i*3 + 2].item()
        #     triangle[2][:] = round(points[0,j,0].item()), round(points[0,j,1].item())
        #     cv2.fillPoly(mask, [triangle], 255)
        # end = time.clock()
        # print('running time:',end-start)
        # cv2.imwrite('/mnt/dy_data/smplify-x-master/output/out.jpg', mask)
         ##################OPENCV 输出mask#####################################

        # 8 sil_loss 
        sil_loss = 0.0
        if self.sil_loss_weight.item() > 0:
            import cv2
            maskGTPath = '/mnt/dy_data/smplify-x-master/data/segment/' + '1461_seg.jpg'
            maskGT = cv2.imread(maskGTPath, cv2.IMREAD_UNCHANGED)
            # saveGT = np.where(maskGT > 0.8, 255,0)
            # cv2.imwrite('/mnt/dy_data/smplify-x-master/output/GT.jpg', saveGT)

            from kaolin.graphics import DIBRenderer as Renderer
            H,W = maskGT.shape
            render_res= max(H,W)
            renderer = Renderer(render_res, render_res)
            Pdw = torch.zeros((1,4,4)).to(device)
            for key, val in camera.named_parameters():
                if key == 'rotation':
                    Pdw[:,:3,:3] = val
                if key == 'translation':
                    Pdw[:,:3,3] = val

            Pdw[:,3,3] = 1
            faces = body_model_faces.view(-1,3)
            vertices = body_model_output.vertices
            
            # 存储vertices
            # import json
            # save_points = vertices.squeeze(0).detach().cpu().numpy().tolist()
            # save_dic = {"points": save_points}
            # json_str = json.dumps(save_dic)
            # with open('output/out.json', 'w') as json_file:
            #     json_file.write(json_str)
            


            #############线性插值 输出mask#####################################
            # import time
            # projected_points = camera(vertices)
            # start = time.clock()
            # mask = gene_mask(projected_points, faces,H,W)
            # end = time.clock()
            # print('running time: ',(end - start))
            # mm = mask.detach().cpu().numpy()
            # mn = np.where(mm==1, 255,0)
            # cv2.imwrite('/mnt/dy_data/smplify-x-master/output/out.jpg', mn)
            ####################################################################

            ##################### kaolin #######################################
            mask_ori = render_silhouette(renderer,vertices,faces,Pdw, render_res, render_res)
            mask_ori = mask_ori.squeeze()
            mask_ori = mask_ori.transpose(0,1)
            # mask_max = torch.max(mask_ori)
            # mask = torch.where(mask_ori == mask_max, 0.0*mask_ori, mask_ori)
            # mask_min = torch.min(mask_ori)
            # mask = torch.where(mask_ori == mask_min, mask_ori, 0.0*mask_ori)
            mask = mask_ori
            # mask = mask.detach().cpu().numpy()
            # mask = mask * 255
            # cv2.imwrite('/mnt/dy_data/smplify-x-master/output/out.jpg', mask)
        

            # # mask剪切
            # if render_res > H:
            #     top_clip = round((render_res - H) / 2)
            #     bottom_clip = H + top_clip
            #     maskClip = mask[top_clip:bottom_clip, :]
            # elif render_res > W:
            #     left_clip = round((render_res - W) / 2)
            #     right_clip = W + left_clip
            #     maskClip = mask[:, left_clip:right_clip]
            # # mask存储    
            # maskClip = maskClip.detach().cpu().numpy()
            # maskClip = maskClip * 255
            # maskClip = np.where(maskClip == 1.0, 0, maskClip)
            # maskClip = np.where(maskClip > 0.0, 255, 0)
            # cv2.imwrite('/mnt/dy_data/smplify-x-master/output/out.jpg', maskClip)

            # maskGT扩展
            if render_res > H:
                top_H = round((render_res - H) / 2)
                bottom_H = int(render_res - H - top_H)
                top_expand = np.zeros( (top_H, W), dtype= np.uint8 )
                bottom_expand = np.zeros( (bottom_H, W), dtype= np.uint8 )
                maskGT = np.concatenate((top_expand, maskGT, bottom_expand),axis=0)    
            elif render_res > W: 
                left_W =  round((render_res - W) / 2)
                right_W = int(render_res - W - left_W)
                left_expand = np.zeros( (H, left_W) ,dtype=np.uint8 )
                right_expand = np.zeros( (H, right_W)  ,dtype= np.uint8)
                maskGT = np.concatenate((left_expand, maskGT, right_expand),axis=1)  
              
    
            maskGT = maskGT.astype(np.float32)
            maskGT = torch.from_numpy(maskGT)
            maskGT = maskGT.to(device)
            # maskGT = maskGT.unsqueeze(dim=0)
            # maskGT = maskGT.unsqueeze(dim=0)

            # blur_layer = get_gaussian_kernel(kernel_size=45, channels=1).to(device)
            # blured_img = blur_layer(maskGT)
            # blured_img = blured_img.squeeze()
            # maskGT = blured_img
            # if hasattr(torch.cuda, 'empty_cache'):
	        #     torch.cuda.empty_cache()

            # blured_max = torch.max(blured_img)
            # blured_img = torch.where(blured_img == blured_max, 0.0*blured_img, blured_img*125)
           
            # blured_img = blured_img.detach().cpu().numpy()
            # cv2.imwrite('/mnt/dy_data/smplify-x-master/output/out.jpg', blured_img)
            
            # loss1  IOU
            # maskGT =  torch.where(maskGT == 0, 0.0*maskGT + 1e-6, maskGT)
            # mul = (maskGT * mask).reshape(1,-1).sum(1)
            # add = (maskGT + mask).reshape(1,-1).sum(1)
            # iou_ = mul / (add - mul + 1e-7)
            # sil_loss = torch.sum(torch.sum(1 - torch.mean(iou_)).pow(2)) * self.sil_loss_weight
            ## sil_loss = torch.sum(self.sli_prior(1 - torch.mean(iou_))) * self.sil_loss_weight

            # loss2
            # sil_loss = torch.sum(torch.sum((mask - maskGT).pow(2))) * self.sil_loss_weight

            # loss3
            sil_diff = mask * (1- maskGT) + maskGT * ( 1 - mask)
            sil_loss = torch.sum(torch.sum((sil_diff).pow(2))) * self.sil_loss_weight
            ####################################################################

        
        # 9、3d joint loss ##################################
        pred_joints3d = body_model_output.joints
        joint3d_loss = 0.0
        if self.use_3dpose:
            # Calculate the weights for each joints
            weights = (joint_weights * joints3d_conf).unsqueeze(dim=-1)
            # Calculate the distance of the projected joints from
            # the ground truth 2D detections
            joint3d_diff = self.robustifier(gt_joints3d - pred_joints3d)
            joint3d_loss = (torch.sum(weights ** 2 * joint3d_diff) *
                        self.data_weight ** 2) # data_weight = 0.833

        # 10、 mesh loss ####################################
        pred_mesh = body_model_output.vertices
        mesh_loss = 0.0
        if self.use_mesh:
            mesh_hat = compute_similarity_transform_batch(pred_mesh, gt_mesh)
            mesh_diff = self.robustifier(mesh_hat - gt_mesh)
            mesh_loss = (torch.sum(mesh_diff) *
                        self.mesh_loss_weight ** 2) # data_weight = 0.833

        # mesh显示
        import trimesh
        self.iter += 1
        if self.iter % 10 == 0:
            vertices = gt_mesh.clone()
            vertices = vertices.detach().cpu().numpy().squeeze()
            face_save = body_model_faces.clone()
            face_save = face_save.cpu().numpy().reshape(-1,3)
            out_mesh = trimesh.Trimesh(vertices, face_save, process=False)
            out_mesh.export("out_nomomesh/temp/gt.obj")

            vertices = mesh_hat.clone()
            vertices = vertices.detach().cpu().numpy().squeeze()
            face_save = body_model_faces.clone()
            face_save = face_save.cpu().numpy().reshape(-1,3)
            out_mesh = trimesh.Trimesh(vertices, face_save, process=False)
            out_mesh.export("out_nomomesh/temp/" + str(self.iter) + "_hat_" + str(mesh_loss.item()) + ".obj")

            vertices_pred = pred_mesh.clone()
            vertices_pred = vertices_pred.detach().cpu().numpy().squeeze()
            out_mesh = trimesh.Trimesh(vertices_pred, face_save, process=False)
            out_mesh.export("out_nomomesh/temp/" + str(self.iter) + "_pred_" + str(mesh_loss.item()) + ".obj")


        # total loss
        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss 
                      + sil_loss + joint3d_loss + mesh_loss)

        # total_loss = mesh_loss
        

        print("joint_loss: ", joint_loss)
        print("pprior_loss: ", pprior_loss)
        print("shape_loss: ", shape_loss)
        print("angle_prior_loss: ", angle_prior_loss)
        print("pen_loss: ", pen_loss)
        # print("joints3d_loss", joint3d_loss)
        print("mesh_loss", mesh_loss)
        print('total loss: ', total_loss)

        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss
