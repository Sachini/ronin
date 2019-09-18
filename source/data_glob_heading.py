import json
import random
import sys
from os import path as osp

import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from math_util import orientation_to_angles
from data_glob_speed import GlobSpeedSequence
from data_utils import load_cached_sequences


class HeadingSequence(GlobSpeedSequence):
    target_dim = 2
    aux_dim = 2     # velocity

    def __init__(self, data_path=None, **kwargs):
        super().__init__(data_path, **kwargs)

    def load(self, data_path):
        super().load(data_path)
        self.velocities = self.targets[:, :2]
        with open(osp.join(data_path, 'info.json')) as f:
            info = json.load(f)
            rot_tango_to_body = info['align_tango_to_body']
            start_frame = info.get('start_frame', 0)

        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            tango_ori = f['pose/tango_ori']
            body_ori_q = quaternion.from_float_array(tango_ori) * quaternion.from_float_array(rot_tango_to_body).conj()
            body_heading = orientation_to_angles(body_ori_q)[start_frame:, 0]
        self.targets = np.stack([np.sin(body_heading), np.cos(body_heading)], axis=-1)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return self.velocities

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class HeadingDataset(Dataset):
    # Input -imu
    # Targets - heading
    # Aux - velocity
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=1000,
                 random_shift=0, transform=None, **kwargs):
        super(HeadingDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, self.velocities = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i][:-1]
            self.velocities[i] = self.velocities[i]

            velocity = np.linalg.norm(self.velocities[i], axis=1)  # Remove outlier ground truth data
            bad_data = velocity > max_norm
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])
        vel = np.copy(self.velocities[seq_id][frame_id - self.window_size:frame_id])

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), vel.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def description(self):
        return {'features': self.feature_dim, 'target': self.target_dim, 'velocity': self.aux_dim}

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis, ], self.targets[i].astype(np.float32)
