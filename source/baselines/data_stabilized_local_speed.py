import json
import os
import random
import sys
from os import path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from scipy.ndimage.filters import gaussian_filter1d
from torch.utils.data import Dataset

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from math_util import *
from data_utils import CompiledSequence, select_orientation_source


class StabilizedLocalSpeedSequence(CompiledSequence):
    """
    Dataset: RoNIN
    """
    feature_dim = 6
    target_dim = 2
    aux_dim = 11

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.gravity = None
        self.info = {}
        self.mode = kwargs.get('mode', 'train')

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)
        self.info['path'] = osp.split(data_path)[-1]

        self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
            data_path, self.max_ori_error, self.grv_only)

        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            ts = np.copy(f['synced/time'])
            if 'synced/grav' in f:
                gravity = np.copy(f['synced/grav'])
            else:
                gravity = np.copy(f['synced/gravity'])
            tango_pos = np.copy(f['pose/tango_pos'])
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
            gyro = np.copy(f['synced/gyro'])
            linacce = np.copy(f['synced/linacce'])

        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        ori_q = init_rotor * ori_q

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = np.concatenate(
            [np.zeros([dt.shape[0], 1]), tango_pos[self.w:] - tango_pos[:-self.w]], axis=1) / dt

        local_v = ori_q[:-self.w].conj() * quaternion.from_float_array(glob_v) * ori_q[:-self.w]
        local_v = quaternion.as_float_array(local_v)[:, 1:]
        local_v_g = align_3dvector_with_gravity(local_v, gravity[:-self.w])

        start_frame = self.info.get('start_frame', 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([gyro, linacce], axis=1)[start_frame:]
        self.targets = local_v_g[start_frame:, [0, 2]]
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
        self.gt_pos = tango_pos[start_frame:]
        self.gravity = gravity[start_frame:]

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gravity, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class StabilizedLocalSpeedDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(StabilizedLocalSpeedDataset, self).__init__()
        self.shift = 0
        assert seq_type.aux_dim == 11
        self.feature_dim = seq_type.feature_dim
        self.aux_dim = seq_type.aux_dim
        self.target_dim = 2
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]

        self.ts, self.features, self.targets, self.gravity, self.orientations, self.gt_pos = [], [], [], [], [], []
        self.index_map = []

        feature_sigma = kwargs.get('feature_sigma', None)
        target_sigma = kwargs.get('target_sigma', None)

        for i in range(len(data_list)):
            seq = seq_type(osp.join(root_dir, data_list[i]), interval=1, **kwargs)
            self.features.append(seq.get_feature())
            self.targets.append(seq.get_target())
            aux = seq.get_aux()
            print(seq.get_meta())
            self.ts.append(aux[:, 0])
            self.orientations.append(aux[:, 1:5])
            self.gravity.append(aux[:, 5:8])
            self.gt_pos.append(aux[:, -3:])

            if feature_sigma is not None:
                self.features[i] = gaussian_filter1d(self.features[i], sigma=feature_sigma, axis=0)
            if target_sigma is not None:
                self.targets[i] = gaussian_filter1d(self.targets[i], sigma=target_sigma, axis=0)
            self.index_map += [[i, j] for j in range(window_size, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat = self.transform(feat)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)
