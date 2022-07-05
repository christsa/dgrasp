import numpy as np
from raisimGymTorch.helper import rotations

def euler_noise_to_quat(quats, palm_pose, noise):
    eulers_palm_mats = np.array([rotations.euler2mat(pose) for pose in palm_pose]).copy()
    eulers_mats =  np.array([rotations.quat2mat(quat) for quat in quats])

    rotmats_list = np.array([rotations.euler2mat(noise) for noise in noise])

    eulers_new = np.matmul(rotmats_list,eulers_mats)
    eulers_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_new])

    eulers_palm_new = np.matmul(rotmats_list,eulers_palm_mats)
    eulers_palm_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_palm_new])

    quat_list = [rotations.euler2quat(noise) for noise in eulers_rotmated]

    return np.array(quat_list), eulers_new, eulers_palm_rotmated