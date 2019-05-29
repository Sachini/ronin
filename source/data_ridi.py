from os import path as osp
import sys

import pandas
import numpy as np
import quaternion

from data_utils import CompiledSequence


class RIDIGlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RIDI (can be downloaded from https://wustl.app.box.com/s/6lzfkaw00w76f8dmu0axax7441xcrzd9)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        self.info['ori_source'] = 'game_rv'

        if osp.exists(osp.join(path, 'processed/data.csv')):
            imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        else:
            imu_all = pandas.read_pickle(osp.join(path, 'processed/data.pkl'))

        ts = imu_all[['time']].values / 1e09
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        tango_pos = imu_all[['pos_x', 'pos_y', 'pos_z']].values

        # Use game rotation vector as device orientation.
        init_tango_ori = quaternion.quaternion(*imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[0])
        game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)

        init_rotor = init_tango_ori * game_rv[0].conj()
        ori = init_rotor * game_rv

        nz = np.zeros(ts.shape)
        gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

        gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
        acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        self.targets = (tango_pos[self.w:, :2] - tango_pos[:-self.w, :2]) / (ts[self.w:] - ts[:-self.w])
        self.gt_pos = tango_pos
        self.orientations = quaternion.as_float_array(game_rv)
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: orientation {}'.format(self.info['path'], self.info['ori_source'])
