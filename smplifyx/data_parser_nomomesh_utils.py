import numpy as np




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


def humanpose2smplpose(keyp_tuple, use_hands):
    pose2d = np.array(keyp_tuple.keypoints[0][:25, :])
    pose3d = np.array(keyp_tuple.keypoints3d[0])

    # humanpose = np.array(val)
    
    smpl2d = np.zeros([25, 3])
    smpl3d = np.zeros([25, 4])
    humanpose_names = get_humanpose_joint_names()
    smpl_names = get_smpl_joint_names()
    idx = 0
    for h in smpl_names:
        if h in humanpose_names:
            smpl2d[idx] = pose2d[humanpose_names.index(h)]
            smpl3d[idx] = pose3d[humanpose_names.index(h)]
            idx += 1
        else:
            idx += 1
    smpl3d = np.expand_dims(smpl3d, axis=0)
    if use_hands:
        smpl2d = np.vstack((smpl2d, keyp_tuple.keypoints[0][25:, :]))

    smpl2d = np.expand_dims(smpl2d, axis=0)
    return smpl2d, smpl3d # 2d(1,25,3)  3d(1,24,4)




def get_mesh_vertices(mesh_path):
    with open(mesh_path) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == 'v':
                points.append([float(strs[1]), float(strs[2]), float(strs[3])])
            if strs[0] == 'f':
                break

    points = np.array(points)
    points = np.expand_dims(points, axis=0)
    return points


