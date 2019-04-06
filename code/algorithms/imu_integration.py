import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

_GRAVITY = 9.81


# @jit(nopython=True)
def angular_velocity_to_quaternion_derivative_v2(q, w):
    omega = np.array([[0, -w[0], -w[1], -w[2]],
                      [w[0], 0, w[2], -w[1]],
                      [w[1], -w[2], 0, w[0]],
                      [w[2], w[1], -w[0], 0]]) * 0.5
    return np.dot(omega, q)


# @jit(nopython=True)
def gyro_integration(ts, gyro, init_q):
    """
    Integrate gyro into orientation.
    https://www.lucidar.me/en/quaternions/quaternion-and-gyroscope/
    """
    output_q = np.zeros((gyro.shape[0], 4))
    output_q[0] = init_q
    dts = ts[1:] - ts[:-1]
    for i in range(1, gyro.shape[0]):
        output_q[i] = output_q[i - 1] + angular_velocity_to_quaternion_derivative_v2(output_q[i - 1], gyro[i - 1]) * dts[i - 1]
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q
