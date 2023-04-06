import numpy as np
import os
# from renderSkeleton import drawSkeleton, render_to_oriImg
import configargparse
import glob
import time
import datetime
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # debug
import ptvsd
ptvsd.enable_attach(address=("172.17.0.5",  9024))
ptvsd.wait_for_attach()


def get_plane(points):
    # print(np.linalg.det(A)) 计算行列式
    length = points.shape[0]
    val_sum = np.sum(points, axis=0) # 按列求和[x_sum,y_sum,z_sum]
    xx = points[:,0] * points[:,0]
    xx_sum = np.sum(xx)
    yy = points[:,1] * points[:,1]
    yy_sum = np.sum(yy)
    zz = points[:,2] * points[:,2]
    zz_sum = np.sum(zz)
    xy = points[:,0] * points[:,1]
    xy_sum = np.sum(xy)
    xz = points[:,0] * points[:,2]
    xz_sum = np.sum(xz)
    yz = points[:,1] * points[:,2]
    yz_sum = np.sum(yz)

    D = np.array([[xx_sum, xy_sum, val_sum[0]],
                [xy_sum, yy_sum,val_sum[1]],
                [val_sum[0], val_sum[1], length]])
    T = np.array([xz_sum,
                yz_sum, 
                val_sum[2]])
    D0 = D.copy()
    D0[:,0] = T
    D1 = D.copy()
    D1[:,1] = T
    D2 = D.copy()
    D2[:,2] = T

    d = np.linalg.det(D)
    d0 = np.linalg.det(D0)
    d1 = np.linalg.det(D1)
    d2 = np.linalg.det(D2)
    
    # 平面方程z=a0*x+a1*y+a2
    a0 = d0/d
    a1 = d1/d
    a2 = d2/d

    return np.array([a0,a1,-1]),a2

def get_cos(p1,p2):
    tmp1 = np.sqrt(np.sum(p1**2)) * np.sqrt(np.sum(p2**2))
    tmp2 = np.sum(p1*p2)
    cos_theta = tmp2/tmp1
    # print(p1)
    # print(p2)
    # print(cos_theta)
    return abs(cos_theta)



#######################################################
####
#### person normal的变化来判断是否倒地
####
#######################################################
if __name__ == '__main__':
    parser = configargparse.ArgParser(description= 'falling detection')
    parser.add_argument('--data_path',  default='smpl_debug/1',
                        help='data path')
    parser.add_argument('--vis_skeleton',
                        default=True,
                        help='visualize result')
    parser.add_argument('--angle_thred',
                        default=70)
    args = parser.parse_args()
    

    data_path = args.data_path
    vis_skeleton = args.vis_skeleton
    log_path = os.path.join(data_path, 'log.txt')

    ###################### 倒地视频 测试 ##################################
    fall_files = os.listdir(os.path.join(data_path, 'fall'))
    false_fall = []
    for file in fall_files:
        # file = '1'
        # print(file)
        npz_dir = os.path.join(data_path,'fall',file,'meshes')
        npz_paths = glob.glob(npz_dir+'/*.npz')
        person_num = 0
        mean_normal = np.zeros(3)
        isfall = False
        for npz_path in npz_paths:
            # npz_path = 'smpl_debug/meshes/30_2.npz'
            print(npz_path)
            # mtime = (os.path.getmtime(npz_path))
            # if mtime < 1673532748.8715677:
            #     continue


            npz_data = np.load(npz_path)
            joints= npz_data['joints']
            tmp_normal = joints[1] - (joints[21] + joints[24]) * 0.5
            # skeleton & ground可视化
            if vis_skeleton:
                # drawSkeleton(joints=joints, m_normal= mean_normal, p_normal=tmp_normal, 
                #             save_path=npz_path[:-4]+'.jpg')
                t = npz_path.split('/')[-1]
                # fn = os.path.join('data/fall', file,'data',t[:-4])
                # fn = os.path.join('data/1/data/', t[:-4]) 
                # render_to_oriImg(npz_data, fn=fn, save_path=npz_path[:-4]+'_ori.jpg')

            if person_num > 0:
                cos1 = get_cos(tmp_normal, mean_normal)
                angle1 = math.acos(cos1)*180/math.pi
                print(angle1)
                if angle1 > args.angle_thred:
                    isfall = True
                    break
                else:
                    mean_normal = (mean_normal * person_num + tmp_normal)/(person_num + 1)
            else:
                mean_normal = tmp_normal

            person_num += 1
        
        # exit(0)
        
        if not isfall:
            false_fall.append(file)
        with open(log_path, 'a+') as f:
            line = file + ': isfall GT: True, isfall pred: '+ str(isfall)
            f.write(line +'\n')
    
    with open(log_path, 'a+') as f:
        line = 'total files: ' + str(len(fall_files)) + ', false detec: '+ str(len(false_fall)) + ", they are: "
        f.write(line +'\n')
        f.write(str(false_fall) +'\n\n\n\n\n')
    

    exit(0)
    ############################## 未倒地视频测试 #####################################
    fall_no_files = os.listdir(os.path.join(data_path,'fall_no'))
    false_fall_no = []
    for file in fall_no_files:
        # file = '16'
        print(file)
        npz_dir = os.path.join(data_path,'fall_no',file,'meshes')
        npz_paths = glob.glob(npz_dir+'/*.npz')
        person_num = 0
        mean_normal = np.zeros(3)
        isfall = False
        for npz_path in npz_paths:
            # mtime = (os.path.getmtime(npz_path))
            # if mtime < 1673532748.8715677:
            #     continue
            print(npz_path)
            npz_data = np.load(npz_path)
            joints= npz_data['joints']
            tmp_normal = joints[1] - (joints[21] + joints[24]) * 0.5 # 1 neck, 21 left heel, 24 right heel
            # skeleton & ground可视化
            if vis_skeleton:
                # drawSkeleton(joints=joints, m_normal= mean_normal, p_normal=tmp_normal, 
                #             save_path=npz_path[:-4]+'.jpg')
                t = npz_path.split('/')[-1]
                fn = os.path.join('data/crouch/images_from48', file,'data',t[:-4]) #原图路径
                render_to_oriImg(npz_data, fn=fn, save_path=npz_path[:-4]+'_ori.jpg')

            if person_num > 0:
                cos1 = get_cos(tmp_normal, mean_normal)
                angle1 = math.acos(cos1)*180/math.pi
                print(angle1)
                if angle1 > args.angle_thred:
                    isfall = True
                    break
                else:
                    mean_normal = (mean_normal * person_num + tmp_normal)/(person_num + 1)
            else:
                mean_normal = tmp_normal

            person_num += 1
       
        
        
        if  isfall:
            false_fall_no.append(file)
        with open(log_path, 'a+') as f:
            line = file + ': isfall GT: False, isfall pred: '+ str(isfall)
            f.write(line +'\n')
    
    with open(log_path, 'a+') as f:
        line = 'total files: ' + str(len(fall_no_files)) + ', false detec: '+ str(len(false_fall_no)) + ", they are: "
        f.write(line +'\n')
        f.write(str(false_fall_no) +'\n')







        






#######################################################
####
#### groung normal 与 person normal的夹角
####
#######################################################
# if __name__ == '__main__':
#     parser = configargparse.ArgParser(description= 'falling detection')
#     parser.add_argument('--data_path',  default='smpl_debug/meshes',
#                         help='data path')
#     parser.add_argument('--vis_skeleton',
#                         default=True,
#                         help='visualize result')
#     parser.add_argument('--log_path',
#                         default='smpl_debug/meshes/log.txt',
#                         help='save result')
#     args = parser.parse_args()
    

#     data_path = args.data_path
#     vis_skeleton = args.vis_skeleton
#     # files = os.listdir(data_path)
#     files = glob.glob(data_path + '*.npz')
#     foot_joints=[]
#     person_num = 0
#     for file in files:
#         print(file)
#         # npzdir=os.path.join(data_path,file,'000.npz')
#         person_num += 1
#         if person_num > 21:
#             break
        
#         npzdir = file
#         # print(npzdir)
#         data = np.load(npzdir)
#         joints= data['joints']
#         # trans = data['transl']
#         tmp_joints = joints[19:,:] #+ trans
#         foot_joints.append(tmp_joints)

#     foot_joints = np.array(foot_joints).reshape(-1,3)
#     ground,a2 = get_plane(foot_joints)
#     print("ground normal: ", ground)
#     print("a2: ",a2)
    
#     # person_dir=os.path.join(data_path,files[0],'000.npz')
#     # person_data = np.load(person_dir)
#     # person_joints = person_data['joints']
#     # person_part = []
#     # person_part.append(person_joints[0])  # nose
#     # person_part.append(person_joints[21])  # left_heel
#     # person_part.append(person_joints[24])  # right_heel
#     # person_part = np.array(person_part)
#     # person_plane,_ = get_plane(person_part)
#     # print("person normal: ", person_plane)

#     # # 计算两向量的夹角
#     # tmp1 = np.sum(ground**2) * np.sum(person_plane**2)
#     # tmp2 = np.sum(ground*person_plane)
#     # cos_theta = tmp2/tmp1
#     # print('cos: ', cos_theta)
#     # if abs(cos_theta) > 0.95:
#     #     print("人躺在地上！")
#     # else:
#     #     print("人没有躺在地上！")

#     for file in files:
#         print(file)
#         npzdir = file
#         # npzdir=os.path.join(data_path,file,'000.npz')
#         data = np.load(npzdir)
#         joints= data['joints']
        
#         person_part = []
#         person_part.append(joints[1])  # neck
#         person_part.append(joints[21])  # left_heel
#         person_part.append(joints[24])  # right_heel
#         person_part = np.array(person_part)
#         person_plane,a2 = get_plane(person_part)
#         print("person normal: ", person_plane)

#         # 计算两向量的夹角
#         tmp1 = np.sqrt(np.sum(ground**2)) * np.sqrt(np.sum(person_plane**2))
#         tmp2 = np.sum(ground*person_plane)
#         cos_theta = tmp2/tmp1
        
#         theta = math.acos(abs(cos_theta))*180/math.pi
#         print('theta: ', theta)
#         if theta < 25:
#             render_to_oriImg(data, fn=file, save_path=npzdir[:-4]+'_ori.jpg')
#             with open(args['log_path'], 'a+') as f:
#                 f.write(npzdir+'\n')
#             # print("人躺在地上！")
#         # else:
#         #     print("人没有躺在地上！")

#         # skeleton & ground可视化
#         # if vis_skeleton:
#             # drawSkeleton(joints=joints, g_coeff=np.array([ground[0],ground[1],a2]), p_normal=person_plane, 
#             #             save_path=npzdir[:-4]+'.jpg')
#             # render_to_oriImg(data, fn=file, save_path=npzdir[:-4]+'_ori.jpg')




    
