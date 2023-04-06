# from turtle import color
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_whole_skeleton():
    # 存在neck和pelvis
    return np.array([
                    [0,1], # mid
                    [1,8], # 1

                    
                    [1,2],  #right
                    [2,3],
                    [3,4],
                    [8,9],
                    [9,10],
                    [10,11],
                    [11,24],
                    [22,24],
                    [0,15],
                    [15,17], #11

                    [0,16], #left
                    [16,18],
                    [1,5],
                    [5,6],
                    [6,7],
                    [8,12],
                    [12,13],
                    [13,14],
                    [14,21],
                    [19,21]  #21
                ])

def get_part_skeleton():
    # 不存在neck和pelvis
    return np.array([
                    [2,3],
                    [3,4],
                    [2,5],
                    [5,6],
                    [6,7],
                    [5,12],
                    [2,9],
                    [9,10],
                    [10,11],
                    [11,24],
                    [22,24],
                    [12,13],
                    [13,14],
                    [14,21],
                    [19,21],
                    [0,16],
                    [0,15],
                    [16,18],
                    [15,17]
                ])

def drawSkeleton(joints,  p_normal, save_path, m_normal=None, g_coeff=None):
    skeletonMap = get_whole_skeleton() # 不存在neck和pelvis
    # 创建画布
    fig = plt.figure(figsize=(15, 15), facecolor='lightyellow')
    # 创建 3D 坐标系
    ax = fig.gca(fc='whitesmoke', projection='3d')
    # 二元函数定义域
    x = np.linspace(-1, 1, 10)
    y = np.linspace(1, 2, 10)
    X, Y = np.meshgrid(x, y)
    # 绘制地面
    # ax.plot_surface(X,
    #             Y,
    #             Z=X*g_coeff[0]+Y*g_coeff[1]+g_coeff[2],
    #             color='g'
    #            )
    # # 绘制地面向量  绿色
    if g_coeff is not None:
        g_normal = [g_coeff[0], g_coeff[1], -1]
        g_points = [joints[8,0],joints[8,1],joints[8,2],
                    joints[8,0] + 0.025, joints[8,1]+g_normal[1]/g_normal[0]*0.025, 
                    joints[8,2]+g_normal[2]/g_normal[0]*0.025]

        ax.quiver(g_points[0],g_points[1],g_points[2],
                g_points[3],g_points[4],g_points[5], color='g')
    # 绘制人体向量 蓝色
    x_len = 0.025
    p_points = [joints[8,0],joints[8,1],joints[8,2],
                joints[8,0] + x_len, joints[8,1]+p_normal[1]/p_normal[0]*x_len, 
                joints[8,2]+p_normal[2]/p_normal[0]*x_len]

    ax.quiver(p_points[0],p_points[1],p_points[2],
              p_points[3],p_points[4],p_points[5], color='b')

    # 绘制人体平均向量
    if m_normal is not None:
        m_points = [joints[8,0],joints[8,1],joints[8,2],
                joints[8,0] + x_len, joints[8,1]+m_normal[1]/m_normal[0]*x_len, 
                joints[8,2]+m_normal[2]/m_normal[0]*x_len]

        ax.quiver(m_points[0],m_points[1],m_points[2],
                m_points[3],m_points[4],m_points[5], color='g')

    # 绘制骨骼点
    # ax.scatter(joints[:,0], joints[:,1], joints[:,2], c='b')
    # 绘制骨骼
    for _, idx in enumerate(skeletonMap):
       s_x = [joints[idx[0],0], joints[idx[1],0]] 
       s_y = [joints[idx[0],1], joints[idx[1],1]] 
       s_z = [joints[idx[0],2], joints[idx[1],2]] 
       ax.plot(s_x, s_y, s_z, c='r')

    # 设置坐标轴标题和刻度
    # ax.set(xlabel='X',
    #     ylabel='Y',
    #     zlabel='Z',
    #     xlim=(-2, 2),
    #     ylim=(-2, 2),
    #     zlim=(-1, 1),
    #     xticks=np.arange(-2, 2, 0.5),
    #     yticks=np.arange(-2, 2, 0.5),
    #     zticks=np.arange(-1, 1, 0.5)
    #     )

    ax.set(xlabel='X',
        ylabel='Y',
        zlabel='Z',
        xlim=(-2, 2),
        ylim=(-2, 2),
        zlim=(3,7),
        xticks=np.arange(-2, 2, 0.5),
        yticks=np.arange(-2, 2, 0.5),
        zticks=np.arange(3, 7, 0.5)
        )

    
    plt.savefig(save_path) 




def render_to_oriImg(data, fn, save_path):
    # oriImg_path = os.path.join('data/images', fn + '.jpg')
    oriImg_path = os.path.join(fn + '.jpg')
    ori_img = cv2.imread(oriImg_path)
    focal_length = data['focal']
    camera_mat = np.zeros((2,2))
    camera_mat[0, 0] = focal_length
    camera_mat[1, 1] = focal_length
    transl = data['transl']
    wor_joints = data['joints']
    center = data['center']
    # cam_joints = wor_joints + transl
    cam_joints = wor_joints
    div_mat = np.expand_dims(cam_joints[:,2], 1)
    cam_joints = cam_joints[:,:-1] / div_mat
    img_joints = np.matmul(cam_joints, camera_mat)  + center
    
    for i in range(img_joints.shape[0]):
        x = round(img_joints[i,0])
        y = round(img_joints[i,1])
        cv2.circle(ori_img, (x,y),5,(0,0,255),-1) 

    # 绘制骨骼
    skeletonMap = get_whole_skeleton()
    for i, idx in enumerate(skeletonMap):
       j1 = tuple([round(img_joints[idx[0],0]), round(img_joints[idx[0],1])])
       j2 = tuple([round(img_joints[idx[1],0]), round(img_joints[idx[1],1])])
       if i<=1: #mid 天蓝色
        cv2.line(ori_img, j1,j2,(255,255,0), 2)
       elif i<=11: #right 绿色
        cv2.line(ori_img, j1,j2,(0,255,0), 2)
       else: #left 蓝色
        cv2.line(ori_img, j1,j2,(255,0,0), 2) #BGR
       
    cv2.imwrite(save_path, ori_img)
    
