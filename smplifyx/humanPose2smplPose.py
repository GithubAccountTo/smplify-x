import numpy as np
import json
import os
import configargparse
import cv2

# # debug
# import ptvsd
# ptvsd.enable_attach(address=("172.17.0.5",  9024))
# ptvsd.wait_for_attach()

def get_humanpose_joint_names():
    return[
        'nose', # 0
        'left_eye', #1
        'right_eye', #2 
        'left_ear', #3
        'right_ear', #4
        'left_shoulder', #5
        'right_shoulder', #6
        'left_elbow', #7
        'right_elbow', #8
        'left_wrist', #9
        'right_wrist', #10
        'left_hip', #11
        'right_hip', #12
        'left_knee', #13
        'right_knee', #14
        'left_ankle', #15
        'right_ankle', #16
        'left_tiptoe', #17
        'right_tiptoe', #18
        'left_heel', #19
        'right_heel', #20
        'head_top', #21
        'left_hand', #22
        'right_hand', #23
        'chin' #24
    ]


def get_smpl_joint_names():
    return[
        'nose', #0
        'neck', #1
        'right_shoulder', #2
        'right_elbow', #3
        'right_wrist', #4
        'left_shoulder', #5
        'left_elbow', #6
        'left_wrist', #7
        'pelvis', # 8
        'right_hip', #9
        'right_knee', #10
        'right_ankle', #11
        'left_hip', #12
        'left_knee', #13
        'left_ankle', #14
        'right_eye', #15
        'left_eye', #16
        'right_ear', #17
        'left_ear', #18
        'left_tiptoe', #19
        'left_littletoe', #20
        'left_heel', #21
        'right_tiptoe', #22
        'right_littletoe', #23
        'right_heel' #24
    ]


def undistortJoints(humanpose):
    # test2.avi
    # K = np.array([1066.01247805740,	0,	0,
    #             0,	1065.87722509040,	0,
    #             986.088622251863,	533.116216649322,	1])

    K = np.array([1074.76797377065,	0,	0,
                  0,	1076.85023447758,	0,
                  972.237565879129,	539.251576999699, 1])

    K = K.reshape((3,3)).T
    # test2.avi
    # distCoeffs = np.array([-0.417756976880724,	0.228724087867131,	
    #                        0.000585400126622163,	0.000334838068567396, -0.0732200442003678])
    distCoeffs = np.array([-0.383349390592039,	0.160516442728525, -0.000305313228923375,	
                            0.00133552573651522, -0.0327655001038719])
    pts = humanpose[:,:-1].astype(np.float32)
    # pts为n*3的数组，其中n为点的个数
    undisJoints = cv2.undistortPoints(pts, K, distCoeffs, P=K)
    undisJoints = undisJoints.squeeze()

    return undisJoints



# ## 图像去畸变 #######
# import glob
# if __name__ == "__main__":
#     data_path = "data_undistort/test/bank1/*.jpg"
#     file_list = glob.glob(data_path)

#     # test2.avi
#     K = np.array([1066.01247805740,	0,	0,
#                 0,	1065.87722509040,	0,
#                 986.088622251863,	533.116216649322,	1])

#     # test1.avi
#     # K = np.array([1074.76797377065,	0,	0,
#     #               0,	1076.85023447758,	0,
#     #               972.237565879129,	539.251576999699, 1])

#     K = K.reshape((3,3)).T
#     # test2.avi
#     distCoeffs = np.array([-0.417756976880724,	0.228724087867131,	
#                            0.000585400126622163,	0.000334838068567396, -0.0732200442003678])
   
#     # test1.avi
#     # distCoeffs = np.array([-0.383349390592039,	0.160516442728525, -0.000305313228923375,	
#     #                         0.00133552573651522, -0.0327655001038719])
    
   
#     for file in file_list:
#          src = cv2.imread(file)
#          dst = cv2.undistort(src, K, distCoeffs)
#          cv2.imwrite(file.replace(".jpg", "_cof2.png"), dst)
         






### 多个视频测试 ############
# if __name__ == '__main__':
#     parser = configargparse.ArgParser(description= 'humanpose to smpl pose')
#     parser.add_argument('--d_path',  default='./data/crouch/images_from48',
#                         help='data path')
#     args = parser.parse_args()
    
#     #############################################
#     data_path = args.d_path
#     files = os.listdir(data_path)
#     # files.remove('fileRename.txt')
#     for file in files:
#         human_path = os.path.join(data_path, file,'humanpose.json')
#         with open(human_path, 'r') as f:
#             dic = json.load(f)
#         for key,val in dic.items():
#             humanpose = np.array(val)
#             right_num = np.sum(humanpose[:,2] >= 0.3)
#             if right_num < 22:
#                 continue
#             smplpose = np.zeros([25, 3])
#             humanpose_names = get_humanpose_joint_names()
#             smpl_names = get_smpl_joint_names()
#             idx = 0
#             for h in smpl_names:
#                 if h in humanpose_names:
#                     smplpose[idx] = humanpose[humanpose_names.index(h)]
#                     idx += 1
#                 else:
#                     idx += 1

#             smplpose = smplpose.tolist()
#             people = [{'pose_keypoints_2d':smplpose, 'face_keypoints_2d':[], 'hand_left_keypoints_2d':[],
#             'hand_right_keypoints_2d':[], 'pose_keypoints_3d':[], 'face_keypoints_3d':[],
#             'hand_left_keypoints_3d':[], 'hand_right_keypoints_3d':[]}]
#             save_dic = {
#                 'version':1.2, 
#                 'people':people
#                 }

#             #######################
#             # 原图存储
#             save_img_path = os.path.join(data_path, file,'imgs') #筛选出的图，存储路径
#             if not os.path.exists(save_img_path):
#                 os.makedirs(save_img_path)
#             save_img_path = os.path.join(save_img_path, key)
#             src_img_path = os.path.join(data_path, file,'data',key) # 所有的原图路径
#             image = cv2.imread(src_img_path)
#             cv2.imwrite(save_img_path, image)
#             #######################

            
#             # save_dir = os.path.join(data_path, file,'keypoints')
#             save_dir = os.path.join(data_path, file,'kps') #筛选出的图，关节点存储路径
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             save_path = os.path.join(save_dir, key[:-4]+'_keypoints.json')
#             print('save path: ', save_path)
#             json_str = json.dumps(save_dic, indent=4)
#             with open(save_path, 'w') as json_file:
#                 json_file.write(json_str)



    
    
## 单个视频 测试 ####
if __name__ == '__main__':
    ########## 超参 ############
    parser = configargparse.ArgParser(description= 'humanpose to smpl pose')
    parser.add_argument('--d_path',  default='./data/1/keypoints',
                        help='data path')
    parser.add_argument('--h_path',
                        default='./data/1/humanpose.json',
                        help='the path of humanpose')
    args = parser.parse_args()
    ########## 超参 ############
    
    data_path = args.d_path
    human_path = args.h_path

    with open(human_path, 'r') as f:
        dic = json.load(f)
    for key,val in dic.items():
        humanpose = np.array(val)

        
        # #### 关节点去畸变
        # undis_human = undistortJoints(humanpose)
        # humanpose[:,:-1] = undis_human


        smplpose = np.zeros([25, 3])
        humanpose_names = get_humanpose_joint_names()
        smpl_names = get_smpl_joint_names()
        idx = 0
        for h in smpl_names:
            if h in humanpose_names:
                smplpose[idx] = humanpose[humanpose_names.index(h)]
                idx += 1
            else:
                idx += 1

        smplpose = smplpose.tolist()
        people = [{'pose_keypoints_2d':smplpose, 'face_keypoints_2d':[], 'hand_left_keypoints_2d':[],
          'hand_right_keypoints_2d':[], 'pose_keypoints_3d':[], 'face_keypoints_3d':[],
          'hand_left_keypoints_3d':[], 'hand_right_keypoints_3d':[]}]
        save_dic = {
            'version':1.2, 
            'people':people
            }
        save_path = os.path.join(data_path, key[:-4]+'_keypoints.json')
        print('save path: ', save_path)
        json_str = json.dumps(save_dic, indent=4)
        with open(save_path, 'w') as json_file:
            json_file.write(json_str)
        


