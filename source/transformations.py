import numpy as np
import quaternion
import math

from numba import jit
from scipy.ndimage.filters import gaussian_filter1d


@jit
def change_cf(ori, vectors):
    """
    Euler-Rodrigous formula v'=v+2s(rxv)+2rx(rxv)
    :param ori: quaternion [n]x4
    :param vectors: vector nx3
    :return: rotated vector nx3
    """
    assert ori.shape[-1] == 4
    assert vectors.shape[-1] == 3

    if len(ori.shape) == 1:
        ori = np.repeat([ori], vectors.shape[0], axis=0)
    q_s = ori[:, :1]
    q_r = ori[:, 1:]

    tmp = np.cross(q_r, vectors)
    vectors = np.add(np.add(vectors, np.multiply(2, np.multiply(q_s, tmp))), np.multiply(2, np.cross(q_r, tmp)))
    return vectors


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, feat, targ, **kwargs):
        for t in self.transforms:
            feat, targ = t(feat, targ)
        return feat, targ


class RandomRotate:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, feat, targ, **kwargs):
        rv = np.random.random(3)
        na = np.linalg.norm(rv)
        if na < 1e-06:
            return feat

        angle = np.random.random() * self.max_angle * math.pi / 180
        rv = rv / na * math.sin(angle / 2.0)
        rot = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(rv))
        rows = feat.shape[0]
        feat = np.matmul(rot, feat.reshape([-1, 3]).T).T
        return feat.reshape([rows, -1]), targ


class RandomSmooth:
    def __init__(self, max_sigma):
        self.max_sigma = max_sigma

    def __call__(self, feat, targ, **kwargs):
        sigma = np.random.random() * self.max_sigma
        return gaussian_filter1d(feat, sigma=sigma, axis=0), targ


class RandomHoriRotate:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, feat, targ, **kwargs):
        angle = np.random.random() * self.max_angle
        rm = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
        feat_aug = np.copy(feat)
        targ_aug = np.copy(targ)
        feat_aug[:, :2] = np.matmul(rm, feat[:, :2].T).T
        feat_aug[:, 3:5] = np.matmul(rm, feat[:, 3:5].T).T
        targ_aug[:2] = np.matmul(rm, targ[:2].T).T

        return feat_aug, targ_aug


class RandomHoriRotateSeq:
    def __init__(self, input_format, output_format=None):
        """
        Rotate global input, global output by a random angle
        @:param input format - input feature vector(x,3) boundaries as array (E.g [0,3,6])
        @:param output format - output feature vector(x,2/3) boundaries as array (E.g [0,2,5])
                                if 2, 0 is appended as z.
        """
        self.i_f = input_format
        self.o_f = output_format

    @jit
    def __call__(self, feature, target):
        a = np.random.random() * 2 * np.math.pi
        # print("Rotating by {} degrees", a/np.math.pi * 180)
        t = np.array([np.cos(a), 0, 0, np.sin(a)])

        for i in range(len(self.i_f) - 1):
            feature[:, self.i_f[i]: self.i_f[i + 1]] = \
                change_cf(t, feature[:, self.i_f[i]: self.i_f[i + 1]])

        for i in range(len(self.o_f) - 1):
            if self.o_f[i + 1] - self.o_f[i] == 3:
                vector = target[:, self.o_f[i]: self.o_f[i + 1]]
                target[:, self.o_f[i]: self.o_f[i + 1]] = change_cf(t, vector)
            elif self.o_f[i + 1] - self.o_f[i] == 2:
                vector = np.concatenate([target[:, self.o_f[i]: self.o_f[i + 1]], np.zeros([target.shape[0], 1])],
                                        axis=1)
                target[:, self.o_f[i]: self.o_f[i + 1]] = change_cf(t, vector)[:, :2]

        return feature.astype(np.float32), target.astype(np.float32)
