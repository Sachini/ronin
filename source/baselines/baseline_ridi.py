import argparse
import json
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from os import path as osp

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from scipy.interpolate import interp1d
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from baselines.data_stabilized_local_speed import StabilizedLocalSpeedDataset, StabilizedLocalSpeedSequence
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error
from math_util import *

_feature_dim, _target_dim = 6, 2

""" 
 Implementation adapted from https://github.com/higerra/ridi_imu
 This part of the ridi code does not contain bias optimization, which did not make much differences in the latest dataset
"""


class SVRBase(ABC):
    def __init_(self, chn):
        self.chn = chn

    @abstractmethod
    def fit(self, feature, target):
        raise NotImplemented

    @abstractmethod
    def predict(self, feature):
        raise NotImplemented

    @abstractmethod
    def save(self, out_dir):
        raise NotImplemented

    @abstractmethod
    def load(self, model_dir):
        raise NotImplemented


class CvSVR(SVRBase):
    """
    SVR implementation from OpenCV.
    """

    def __init__(self, chn, c, e, **kwargs):
        self.chn = chn
        self.m = cv2.ml.SVM_create()
        self.m.setType(cv2.ml.SVM_EPS_SVR)
        self.m.setC(c)
        self.m.setDegree(1)
        self.m.setP(e)
        max_iter = kwargs.get('max_iter', 10000)
        self.m.setTermCriteria(
            (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, max_iter, 1e-09))

    def fit(self, feature, target):
        self.m.train(feature.astype(np.float32), cv2.ml.ROW_SAMPLE,
                     target.astype(np.float32))

    def predict(self, feature):
        return self.m.predict(feature.astype(np.float32))[1].astype(np.float).ravel()

    def save(self, out_dir):
        self.m.save(osp.join(out_dir, 'regressor_{}.yaml'.format(self.chn)))

    def load(self, model_dir):
        self.m = cv2.ml.SVM_load(osp.join(
            model_dir, 'regressor_{}.yaml'.format(self.chn)))


def get_dataset(root_dir, data_list, args, **kwargs):
    seq_type = {'rin': StabilizedLocalSpeedSequence}

    grv_only = False
    if kwargs.get('mode', 'train') == 'test':
        grv_only = True

    dataset = StabilizedLocalSpeedDataset(
        seq_type[args.dataset], root_dir, data_list, step_size=args.step_size, window_size=args.window_size,
        feature_sigma=args.feature_sigma, target_sigma=args.target_sigma,
        shuffle=False, grv_only=grv_only, **kwargs)

    total_mem = len(dataset) * (args.window_size * _feature_dim + _target_dim) * 32 / 1024 / 1024 / 1024
    print('Total memory: {:.3f}Gb'.format(total_mem))
    features_all = np.empty([len(dataset), dataset.feature_dim * args.window_size])
    targets_all = np.empty([len(dataset), dataset.target_dim])

    for i in range(len(dataset)):
        feat, targ, _, _ = dataset[i]
        features_all[i] = feat.T.flatten()
        targets_all[i] = targ

    return features_all, targets_all, dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        list_all = [s.strip().split(',' or ' ') for s in f.readlines() if len(s) > 0 and s[0] != '#']
        data_list = [d[0] for d in list_all if args.attachment is None or d[1] == args.attachment]
    return get_dataset(root_dir, data_list, args, **kwargs)


def train_chn(args):
    features, targets, _ = get_dataset_from_list(args.root_dir, args.train_list, args, mode=train)
    # Normalize feature
    mean, std = np.mean(features, axis=0), np.std(features, axis=0)
    features = (features - mean) / std
    targets = targets[:, args.chn]

    print('Number of training samples:', features.shape[0])

    start_t = time.time()
    print('Training for single channel {}'.format(args.chn))
    model = CvSVR(args.chn, args.c, args.e)
    model.fit(features, targets)
    end_t = time.time()

    print('Done. Time usage: {:.3f}s'.format(end_t - start_t))
    train_out = model.predict(features)
    train_loss = np.mean((train_out - targets) ** 2)
    print('Training loss for {}: {}'.format(args.chn, train_loss))

    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.exists(osp.join(args.out_dir, 'feat_mean.npy')):
            np.save(osp.join(args.out_dir, 'feat_mean.npy'), mean)
            np.save(osp.join(args.out_dir, 'feat_std.npy'), std)
        model.save(args.out_dir)
        print('Model saved to {}'.format(args.out_dir))


def train(args):
    features, targets, _ = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    # Normalize feature
    mean, std = np.mean(features, axis=0), np.std(features, axis=0)
    features = (features - mean) / std

    print('Number of training samples:', features.shape[0])
    cs = [args.c for _ in range(targets.shape[1])]
    es = [args.e for _ in range(targets.shape[1])]

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)['best_params']
        cs = [config['chn_{}'.format(i)]['param']['C'] for i in range(targets.shape[1])]
        es = [config['chn_{}'.format(i)]['param']['epsilon'] for i in range(targets.shape[1])]
        print('Config loaded from ' + args.config)
    print('Config: c: {}, e: {}'.format(cs, es))

    start_t = time.time()
    models = [CvSVR(chn, cs[chn], es[chn]) for chn in range(targets.shape[1])]
    for i in range(targets.shape[1]):
        print('Training for channel {}'.format(i))
        models[i].fit(features, targets[:, i])
    end_t = time.time()

    print('Done. Time usage: {:.3f}s'.format(end_t - start_t))
    for i in range(targets.shape[1]):
        train_out = models[i].predict(features)
        train_loss = np.mean((train_out - targets[:, i]) ** 2)
        print('Training loss for {}: {}'.format(i, train_loss))

    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        np.save(osp.join(args.out_dir, 'feat_mean.npy'), mean)
        np.save(osp.join(args.out_dir, 'feat_std.npy'), std)
        [m.save(args.out_dir) for m in models]
        print('Model saved to {}'.format(args.out_dir))


def grid_search(args):
    features, targets, _ = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    print('Number of training samples:', features.shape[0])

    # Data normalization
    mean, std = np.mean(features, axis=0), np.std(features, axis=0)
    features = (features - mean) / std

    if args.c < 0:
        c_opt = [0.1, 1.0, 10.0, 100.0]
    else:
        c_opt = [args.c]
    search_dict = {'C': c_opt,
                   'epsilon': [1e-04, 1e-03, 1e-02, 1e-01],
                   'gamma': ['auto']}
    start_t = time.time()

    best_params = {}
    for i in range(targets.shape[1]):
        print('Channel {}'.format(i))
        grid_searcher = GridSearchCV(
            svm.SVR(), search_dict, cv=3, scoring='neg_mean_squared_error', n_jobs=args.num_workers, verbose=2)
        grid_searcher.fit(features, targets[:, i])
        best_params['chn_{}'.format(i)] = {'param': grid_searcher.best_params_, 'score': grid_searcher.best_score_}
    end_t = time.time()
    print('Time usage: {:.3f}'.format(end_t - start_t))
    print(best_params)

    if args.out_path is not None:
        best_params = {'best_params': best_params}
        with open(args.out_path, 'w') as f:
            json.dump(best_params, f)


def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)
    ts = dataset.ts[seq_id]
    dt = np.mean(ts[ind[1:]] - ts[ind[:-1]])

    rot_local_to_grav = get_rotation_compensate_gravity(dataset.gravity[seq_id])
    ori = quaternion.from_float_array(dataset.orientations[seq_id])

    rot_grav_to_glob = (ori * quaternion.from_float_array(rot_local_to_grav).conj())[ind]

    za = np.zeros(preds.shape[0])
    preds_q = quaternion.from_float_array(np.stack([za, preds[:, 0], za, preds[:, 1]], axis=1))
    glob_v = quaternion.as_float_array(rot_grav_to_glob * preds_q * rot_grav_to_glob.conj())[:, 1:] * dt

    pos = np.zeros([glob_v.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(glob_v[:, :2], axis=0) + pos[0]
    pos[-1] = pos[-2]

    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


def test_sequence(args):
    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.root_dir
        with open(args.test_list) as f:
            list_all = [s.strip().split(',' or ' ') for s in f.readlines() if len(s) > 0 and s[0] != '#']
            data_list = [d[0] for d in list_all if args.attachment is None or d[1] == args.attachment]
    else:
        raise ValueError

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    _, dummy_targ, _ = get_dataset(root_dir, [data_list[0]], args)
    mean, std = 0.0, 1.0
    models = [CvSVR(i, 0., 0.) for i in range(dummy_targ.shape[1])]
    [m.load(args.model_path) for m in models]
    del dummy_targ

    try:
        mean = np.load(osp.join(args.model_path, 'feat_mean.npy'))
        std = np.load(osp.join(args.model_path, 'feat_std.npy'))
    except OSError:
        warnings.warn('Feature mean and std not found. Skip data normalization.')

    losses_all, ate_all, rte_all = [], [], []
    pred_per_min = 200 * 60
    for data in data_list:
        feat, targ, dataset = get_dataset(root_dir, [data], args, mode='test')
        feat = (feat - mean) / std
        preds = np.empty(targ.shape)
        for i in range(len(models)):
            preds[:, i] = models[i].predict(feat)
        loss = np.mean((preds - targ) ** 2, axis=0)
        losses_all.append(loss)

        ind = np.array([i[1] for i in dataset.index_map])
        pos_gt = dataset.gt_pos[0][:, :2]
        pos_pred = recon_traj_with_preds(dataset, preds, 0)
        pos_pred = pos_pred[:, :2]

        if args.align_length is not None:
            _, r, t = fit_transformation(pos_pred[:args.align_length], pos_gt[:args.align_length])
            pos_pred = np.matmul(r, (pos_pred - pos_pred[0]).T).T + pos_gt[0]

        pos_mse = np.linalg.norm(pos_pred - pos_gt, axis=1)

        ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
        ate_all.append(ate)

        if pos_pred.shape[0] < pred_per_min:
            ratio = pred_per_min / pos_pred.shape[0]
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
        else:
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
        rte_all.append(rte)

        print('{}: loss: {}/{:.6f}, ate: {:.6f}, rte: {:.6f}'.format(data, loss, np.mean(loss), ate, rte))

        targ_names = ['vx', 'vz']
        kp = preds.shape[1]

        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.title(data)
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.plot(pos_mse)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate, rte)])
        for i in range(kp):
            plt.subplot2grid((kp, 2), (i, 1))
            plt.plot(ind, preds[:, i])
            plt.plot(ind, targ[:, i])
            plt.legend(['Predicted', 'Ground truth'])
            plt.title('{}, error: {:.6f}'.format(targ_names[i], loss[i]))
        plt.tight_layout()

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            res_out = np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1)
            np.save(osp.join(args.out_dir, data + '_ridi.npy'), res_out)
            plt.savefig(osp.join(args.out_dir, data + '_ridi.png'))

        plt.close('all')

    losses_all = np.stack(losses_all, axis=0)
    losses_avg = np.mean(losses_all, axis=1)

    attach = 'unknown' if args.attachment is None else args.attachment
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
            for i in range(losses_all.shape[0]):
                f.write('{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                    data_list[i], attach, losses_all[i][0], losses_all[i][1], losses_avg[i], ate_all[i], rte_all[i]))

    print('Overall loss: {}/{:.6f}, avg ATE: {:.6f}, avg RTE: {:.6f}'.format(
        np.mean(losses_all, axis=0), np.mean(losses_avg), np.mean(ate_all), np.mean(rte_all)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ronin')
    parser.add_argument('--attachment', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'search'])
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default='None')
    parser.add_argument('--train_list', type=str, default=None)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--cache_path', type=str, default='None')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--chn', type=int, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--align_length', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--c', type=float, default=10)
    parser.add_argument('--e', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--use_ekf', action='store_true')
    parser.add_argument('--feature_sigma', type=float, default=2.0)
    parser.add_argument('--target_sigma', type=float, default=30.0)
    parser.add_argument('--show_plot', action='store_true')

    args = parser.parse_args()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        if args.chn is not None:
            train_chn(args)
        else:
            train(args)
    elif args.mode == 'test':
        test_sequence(args)
    elif args.mode == 'search':
        grid_search(args)
