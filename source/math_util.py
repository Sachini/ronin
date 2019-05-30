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
        output_q[i] = output_q[i - 1] + angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1]) * dts[
            i - 1]
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q


def interpolate_quaternion_linear(data, ts_in, ts_out):
    """
    This function interpolate the input quaternion array into another time stemp.

    Args:
        data: Nx4 array containing N quaternions.
        ts_in: input_timestamp- N-sized array containing time stamps for each of the input quaternion.
        ts_out: output_timestamp- M-sized array containing output time stamps.
    Return:
        Mx4 array containing M quaternions.
    """

    assert np.amin(ts_in) <= np.amin(ts_out), 'Input time range must cover output time range'
    assert np.amax(ts_in) >= np.amax(ts_out), 'Input time range must cover output time range'
    pt = np.searchsorted(ts_in, ts_out)
    d_left = quaternion.from_float_array(data[pt - 1])
    d_right = quaternion.from_float_array(data[pt])
    ts_left, ts_right = ts_in[pt - 1], ts_in[pt]
    d_out = quaternion.quaternion_time_series.slerp(d_left, d_right, ts_left, ts_right, ts_out)
    return quaternion.as_float_array(d_out)


def icp_fit_transformation(source, target):
    """
    This function computes the best rigid transformation between two point sets. It assumes that "source" and
    "target" are with the same length and "source[i]" corresponds to "target[i]".

    :param source: Nxd array.
    :param target: Nxd array.
    :return: A transformation as (d+1)x(d+1) matrix; the rotation part as a dxd matrix and the translation
    part as a dx1 vector.
    """
    assert source.shape == target.shape
    center_source = np.mean(source, axis=0)
    center_target = np.mean(target, axis=0)
    m = source.shape[1]
    source_zeromean = source - center_source
    target_zeromean = target - center_target
    W = np.dot(source_zeromean.T, target_zeromean)
    U, S, Vt = np.linalg.svd(W)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = center_target.T - np.dot(R, center_source.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t


def dot_product_arr(v1, v2):
    if v1.ndim == 1:
        v1 = np.expand_dims(v1, axis=0)
    if v2.ndim == 1:
        v2 = np.expand_dims(v2, axis=0)
    assert v1.shape[0] == v2.shape[0], '{} {}'.format(v1.shape, v2.shape)
    dp = np.matmul(np.expand_dims(v1, axis=1), np.expand_dims(v2, axis=2))
    return np.squeeze(dp, axis=(1, 2))


def quaternion_from_two_vectors(v1, v2):
    """
    Compute quaternion from two vectors. v1 and v2 need not be normalized.

    :param v1: starting vector
    :param v2: ending vector
    :return Quaternion representation of rotation that rotate v1 to v2.
    """
    one_dim = False
    if v1.ndim == 1:
        v1 = np.expand_dims(v1, axis=0)
        one_dim = True
    if v2.ndim == 1:
        v2 = np.expand_dims(v2, axis=0)
    assert v1.shape == v2.shape
    v1n = v1 / np.linalg.norm(v1, axis=1)[:, None]
    v2n = v2 / np.linalg.norm(v2, axis=1)[:, None]
    w = np.cross(v1n, v2n)
    q = np.concatenate([1.0 + dot_product_arr(v1n, v2n)[:, None], w], axis=1)
    q /= np.linalg.norm(q, axis=1)[:, None]
    if one_dim:
        return q[0]
    return q


def align_3dvector_with_gravity_legacy(data, gravity, local_g_direction=np.array([0, 1, 0])):
    """
    Eliminate pitch and roll from a 3D vector by aligning gravity vector to local_g_direction.

    @:param data: N x 3 array
    @:param gravity: real gravity direction
    @:param local_g_direction: z direction before alignment
    @:return
    """
    assert data.ndim == 2, 'Expect 2 dimensional array input'
    assert data.shape[1] == 3, 'Expect Nx3 array input'
    assert data.shape[0] == gravity.shape[0], '{}, {}'.format(data.shape[0], gravity.shape[0])
    epsilon = 1e-03
    gravity_normalized = gravity / np.linalg.norm(gravity, axis=1)[:, None]
    output = np.copy(data)
    for i in range(data.shape[0]):
        # Be careful about two singular conditions where gravity[i] and local_g_direction are parallel.
        gd = np.dot(gravity_normalized[i], local_g_direction)
        if gd > 1. - epsilon:
            continue
        if gd < -1. + epsilon:
            output[i, [1, 2]] *= -1
            continue
        q = quaternion.from_float_array(quaternion_from_two_vectors(gravity[i], local_g_direction))
        output[i] = (q * quaternion.quaternion(1.0, *data[i]) * q.conj()).vec
    return output


def get_rotation_compensate_gravity(gravity, local_g_direction=np.array([0, 1, 0])):
    assert np.linalg.norm(local_g_direction) == 1.0
    gravity_normalized = gravity / np.linalg.norm(gravity, axis=1)[:, None]
    local_gs = np.stack([local_g_direction] * gravity.shape[0], axis=1).T
    dp = dot_product_arr(local_gs, gravity_normalized)
    flag_arr = np.zeros(dp.shape[0], dtype=np.int)
    flag_arr[dp < 0.0] = -1
    qs = quaternion_from_two_vectors(gravity_normalized, local_gs)
    return qs
