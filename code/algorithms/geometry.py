import math

import numpy as np
import quaternion


def adjust_angle_arr(yaw_angle):
    new_angle = np.copy(yaw_angle)
    yaw_diff = yaw_angle[1:] - yaw_angle[:-1]

    yaw_diff_cand = yaw_diff[:, None] - np.array([-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4])
    min_id = np.argmin(np.abs(yaw_diff_cand), axis=1)

    yaw_diff = np.choose(min_id, yaw_diff_cand.T)
    new_angle[1:] = np.cumsum(yaw_diff) + new_angle[0]
    return new_angle


def orientation_to_angles(ori):
    if ori.dtype != quaternion.quaternion:
        ori = quaternion.from_float_array(ori)

    rm = quaternion.as_rotation_matrix(ori)
    angles = np.zeros([ori.shape[0], 3])
    angles[:, 0] = adjust_angle_arr(np.arctan2(rm[:, 0, 1], rm[:, 1, 1]))
    angles[:, 1] = adjust_angle_arr(np.arcsin(-rm[:, 2, 1]))
    angles[:, 2] = adjust_angle_arr(np.arctan2(-rm[:, 2, 0], rm[:, 2, 2]))

    return angles
