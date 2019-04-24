import math

import numpy as np
import quaternion

def adjust_angle_array(angles):
    """
    Resolve ambiguities within a array of angles. It assumes neighboring angles should be close.
    Args:
        angles: an array of angles.

    Return:
        Adjusted angle array.
    """
    new_angle = np.copy(angles)
    angle_diff = angles[1:] - angles[:-1]

    diff_cand = angle_diff[:, None] - np.array([-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4])
    min_id = np.argmin(np.abs(diff_cand), axis=1)

    diffs = np.choose(min_id, diff_cand.T)
    new_angle[1:] = np.cumsum(diffs) + new_angle[0]
    return new_angle


def orientation_to_angles(ori):
    """
    Covert an array of quaternions to an array of Euler angles. Calculations are from Android source code:

    https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java
    Function "getOrientation(float[] R, float[] values)"

    Note that this function DOES NOT consider singular configurations, such as Gimbal Lock.

    Args:
        ori: an array of N quaternions.

    Returns:
        A Nx3 array. With Android's game rotation vector or rotation vector, each group of three values
        correspond to: azimuth(yaw), pitch and roll, respectively.
    """
    if ori.dtype != quaternion.quaternion:
        ori = quaternion.from_float_array(ori)

    rm = quaternion.as_rotation_matrix(ori)
    angles = np.zeros([ori.shape[0], 3])
    angles[:, 0] = adjust_angle_array(np.arctan2(rm[:, 0, 1], rm[:, 1, 1]))
    angles[:, 1] = adjust_angle_array(np.arcsin(-rm[:, 2, 1]))
    angles[:, 2] = adjust_angle_array(np.arctan2(-rm[:, 2, 0], rm[:, 2, 2]))

    return angles


def angular_velocity_to_quaternion_derivative(q, w):
    omega = np.array([[0, -w[0], -w[1], -w[2]],
                      [w[0], 0, w[2], -w[1]],
                      [w[1], -w[2], 0, w[0]],
                      [w[2], w[1], -w[0], 0]]) * 0.5
    return np.dot(omega, q)


def gyro_integration(ts, gyro, init_q):
    """
    Integrate gyro into orientation.
    https://www.lucidar.me/en/quaternions/quaternion-and-gyroscope/
    """
    output_q = np.zeros((gyro.shape[0], 4))
    output_q[0] = init_q
    dts = ts[1:] - ts[:-1]
    for i in range(1, gyro.shape[0]):
        output_q[i] = output_q[i - 1] + angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1]) * dts[i - 1]
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q
