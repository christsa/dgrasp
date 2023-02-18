import torch
import numpy as np
from manopth.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as R

MANO_TO_CONTACT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 6,
    9: 7,
    10:8,
    11:9,
    12:9,
    13:10,
    14:11,
    15:12,
    16:12,
    17:13,
    18:14,
    19:15,
    20:15,
}

def mano_to_dgrasp(param):
    offset = np.array([0.09566993, 0.00638343, 0.00618631])
    axisang = param[3:48].reshape(-1,3).copy()

    # exchange ring finger and little finger's sequence
    temp = axisang[6:9].copy()
    axisang[6:9]=axisang[9:12]
    axisang[9:12]=temp

    # change axis angle to euler angle
    joint_rot = R.from_rotvec(axisang)
    joint_euler = joint_rot.as_euler('XYZ').reshape(-1)
    global_rot = R.from_rotvec(param[:3])
    global_euler = global_rot.as_euler('XYZ')

    # add an offset so that the wrist is the center
    hand_pos = param[48:]+offset
    dgrasp_qpos = np.concatenate([hand_pos,global_euler,joint_euler])

    return dgrasp_qpos

# hand_param: tensor (1,51) mano parameters in PCA
# obj_pcd: array (3000,3) obj point cloud
# dgrasp_label dict:{'final_qpos': (18,51),...} dexycb_train_label of ONE object
# k: the K'th grasp pose fot that object
def to_dgrasp(hand_param,obj_pcd,dgrasp_label,k):
    # final qpos
    #dgrasp_label = {}
    hand_param = hand_param.detach()

    mano_layer = ManoLayer(mano_root='networks', use_pca=False, ncomps=45)
    hand_axis_angle = torch.einsum('bi,ij->bj', [hand_param[:, 3:48], mano_layer.th_comps])
    hand_param[:, 3:48] = hand_axis_angle

    verts, joints = mano_layer(th_pose_coeffs=hand_param[:,:48],th_trans=hand_param[:,48:])
   
    dgrasp_qpos = mano_to_dgrasp(hand_param.detach().numpy()[0])
    
    full_joints = torch.zeros([21,3])

    joints=joints[0]
    full_joints[0] = joints[0]
    full_joints[1:-4] = joints[5:]
    full_joints[-4:] = joints[1:5]

    dgrasp_label['final_qpos'][k] = dgrasp_qpos

    final_ee = full_joints.reshape(-1).detach().numpy()
    #a = dgrasp_label['final_ee'][k]-final_ee
    #a = a.reshape(-1,3)
    dgrasp_label['final_ee'][k] = final_ee

    # Hand goal pose (48DoF), 3DoF global euler rotation + 45DoF local joint angles
    dgrasp_label['final_pose'][k] = dgrasp_qpos[3:]

    # Object goal state in global frame (7), 3 translation + 4 quaternion rotation，used to get a label
    #dgrasp_label['final_obj_pos'][k] = dgrasp_label['obj_pose_reset'][k]

    contact_threshold = 0.015

    ftip_pos = final_ee.reshape(1,-1,3)#[:,:1]

    final_relftip_pos = np.tile(ftip_pos,(obj_pcd.shape[0],1,1))

    verts = obj_pcd[:,np.newaxis]
    diff_vert_fpos = np.linalg.norm(final_relftip_pos-verts,axis=-1)

    min_vert_dist = np.min(diff_vert_fpos,axis=0)

    idx_below_thresh = np.where(min_vert_dist<contact_threshold)[0]
    target_idxs = [MANO_TO_CONTACT[idx] for idx in idx_below_thresh]

    target_contacts = np.zeros(16)
    target_contacts[target_idxs]=1
    target_contacts[-1]=1
    dgrasp_label['final_contacts'][k] = target_contacts
    
    return dgrasp_label
