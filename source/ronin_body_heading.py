import json
import os
import sys
import time
from os import path as osp
from shutil import copyfile

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))

from model_temporal import LSTMSeqNetwork
from data_glob_heading import HeadingSequence, HeadingDataset
from transformations import ComposeTransform, RandomHoriRotateSeq
from metric import compute_heading_error
from math_util import adjust_angle_array
from utils import load_config

torch.multiprocessing.set_sharing_strategy('file_system')
_input_channel, _output_channel = 6, 2
_device = 'cpu'
_min_moving_speed = 0.1


class HeadingNetwork(torch.nn.Module):
    def __init__(self, network, heading_dim=2, pre_norm=False, separate=True, get_prediction=False, weight=None):
        """
        Calculate heading angle with/out loss.
        The heading angle is absolute heading from the point person starts moving

        :param network: Network model
            - input - imu features
            - output - absolute_heading, prenorm_abs
        :param pre_norm: force network to output normalized values by adding a loss
        :param separate: report errors separately with total (The #losses can be obtained from get_channels()).
                        If False, only (weighted) sum of all losses will be returned.
        :param get_prediction: For testing phase, to get prediction values. In training only losses will be returned.
        :param weight: weight for each loss type (should ensure enough channels are given). If not all will be
                        weighed equally.
        """
        super(HeadingNetwork, self).__init__()
        self.network = network
        self.h_dim = heading_dim
        self.pre_norm = pre_norm
        self.concat_loss = not separate

        self.predict = get_prediction

        losses, channels = self.get_channels()
        if self.predict or weight is None or weight in ('None', 'none', False):
            self.weights = torch.ones(channels, dtype=torch.get_default_dtype(), device=_device)
        else:
            assert len(weight) == channels
            self.weights = torch.tensor(weight).to(_device)

    def forward(self, feature, heading=None, velocity=None):
        if self.predict:
            return self.forward_predict(feature)
        else:
            return self.forward_train(feature, heading, velocity)

    def forward_train(self, feature, gt_heading, velocity):
        output = self.network(feature)
        losses = []

        heading, norm_err = self.normalize_values(output)

        vel_mask = self.get_moving_mask(torch.norm(velocity, dim=2), _min_moving_speed)

        # absolute angle is valid from the first moving point onwards
        abs_angle_err = self.mse_of_2dvec(heading, gt_heading, vel_mask)
        losses.append(abs_angle_err)

        if self.pre_norm:
            losses.append(norm_err)

        losses = torch.stack(losses)
        weighted_loss = torch.sum(losses * self.weights)
        if self.concat_loss:
            return weighted_loss, weighted_loss
        else:
            return weighted_loss, losses

    def forward_predict(self, feature):
        output = self.network(feature)
        heading, norm_err = self.normalize_values(output)

        return heading, norm_err

    def normalize_values(self, values):
        norm_err = torch.norm(values, dim=2) - 1
        output = torch.nn.functional.normalize(values, dim=2)
        return output, torch.mean(torch.abs(norm_err))

    def consistent_find_leftmost(self, values, condition):
        indices = torch.arange(values.size(1), 0, -1, dtype=torch.float, device=values.device)
        mask = condition(values).float() * indices.unsqueeze(0).expand_as(values)
        return torch.argmax(mask, dim=1)

    def get_moving_mask(self, values, threshold):
        """
        Find mask for point along sequence where motion starts
        :param values: batch x length
        :param threshold: min value
        :return: mask of size values
        """
        assert len(values.shape) == 2
        min_i = self.consistent_find_leftmost(values, lambda x: x >= threshold).float().unsqueeze(-1)
        indices = torch.arange(0, values.size(1), dtype=torch.float, device=values.device).unsqueeze(0).expand_as(
            values)
        return indices >= min_i

    @staticmethod
    def mse_of_2dvec(pred, targ, mask=None):
        result = torch.norm(pred - targ, dim=2)
        if mask is None:
            return torch.mean(result)
        else:
            return torch.sum(result * mask.float()) / torch.sum(mask)

    def get_channels(self):
        l = 1
        c = 1
        if self.pre_norm:
            c += 1
        if self.concat_loss:
            return l, c
        return l + c, c


def write_config(args, **kwargs):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "seq2seq_heading_models"
            if kwargs:
                values['kwargs'] = kwargs
            json.dump(values, f, sort_keys=True)


def get_dataset(root_dir, data_list, args, **kwargs):
    input_format = [0, 3, 6]
    output_format = [0, _output_channel]
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, [], False

    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms.append(RandomHoriRotateSeq(input_format, output_format))
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True
    transforms = ComposeTransform(transforms)

    dataset = HeadingDataset(HeadingSequence, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
                             random_shift=random_shift, transform=transforms,
                             shuffle=shuffle, grv_only=grv_only, **kwargs)

    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split()[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def get_model(args, mode='train', **kwargs):
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')
    network = LSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, _device, lstm_layers=args.layers, lstm_size=args.layer_size,
                             **config).to(_device)

    model = HeadingNetwork(network, heading_dim=2, pre_norm=kwargs.get('heading_norm', False), separate=kwargs.get('separate_loss', False),
                           get_prediction=(mode != 'train'), weight=kwargs.get('weights'))

    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    return model


def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.data_dir, args.train_list, args, mode='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    end_t = time.time()

    print('Training set loaded. Time usage: {:.3f}s'.format(end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.data_dir, args.val_list, args, mode='val', **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('Validation set loaded')

    global _device
    _device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        copyfile(args.train_list, osp.join(args.out_dir, "train_list"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.out_dir, "validation_list"))
        write_config(args, **kwargs)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_model(args, **kwargs).to(_device)
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.6, verbose=True, eps=1e-12)
    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', False)

    log_file = None
    if args.out_dir:
        log_file = osp.join(args.out_dir, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.out_dir, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    if kwargs.get('force_lr', False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)
    _, channels = network.get_channels()
    print("Starting from epoch {}".format(start_epoch))
    try:
        for epoch in range(start_epoch, args.epochs):
            log_line = format_string(epoch, optimizer.param_groups[0]['lr'])
            network.train()
            train_loss = np.zeros(channels)
            start_t = time.time()
            w_losses = 0.0
            for bid, batch in enumerate(train_loader):
                feat, targ, vel, _, _ = batch
                feat, targ, vel = feat.to(_device), targ.to(_device), vel.to(_device)
                optimizer.zero_grad()
                w_loss, loss = network(feat, targ, vel)
                train_loss += loss.cpu().detach().numpy()
                w_losses += w_loss.cpu().detach().numpy()
                w_loss.backward()
                optimizer.step()
                step += 1

            train_loss /= train_mini_batches
            train_errs[epoch] = np.sum(w_losses / train_mini_batches)
            end_t = time.time()
            result = format_string(train_loss, train_errs[epoch])
            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss {} result: {} '.format(
                    epoch, end_t - start_t, train_errs[epoch], train_loss))
            log_line = format_string(log_line, result)

            saved_model = False
            if val_loader:
                network.eval()
                val_loss = np.zeros(channels)
                w_losses = 0.0
                for bid, batch in enumerate(val_loader):
                    feat, targ, vel, _, _ = batch
                    feat, targ, vel = feat.to(_device), targ.to(_device), vel.to(_device)
                    optimizer.zero_grad()
                    w_loss, loss = network(feat, targ, vel)
                    val_loss += loss.cpu().detach().numpy()
                    w_losses += w_loss.cpu().detach().numpy()
                val_loss /= val_mini_batches
                val_sum = w_losses / val_mini_batches
                result = format_string(val_loss, val_sum)
                log_line = format_string(log_line, result)
                if not quiet_mode:
                    print('Validation loss: {} result: {}'.format(val_sum, val_loss))

                if val_sum < best_val_loss:
                    best_val_loss = val_sum
                    saved_model = True
                    if args.out_dir:
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_errs[epoch],
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if use_scheduler:
                    scheduler.step(val_sum)

            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)


def traj_from_velocity(vel, freq=200):
    dts = 1 / freq
    pos = vel * dts
    pos = np.cumsum(pos, axis=0)
    return pos


def test(args, **kwargs):
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatch
    global _device

    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.data_dir if args.data_dir else osp.split(args.test_list)[0]
        with open(args.test_list) as f:
            test_data_list = [s.strip().split()[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    if args.out_dir and not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(osp.join(str(Path(args.model_path).parents[1]), 'config.json'), 'r') as f:
        model_data = json.load(f)

    _device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if _device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path, map_location={model_data['device']: args.device})

    seq_dataset = get_dataset(root_dir, test_data_list, args, mode='test', **kwargs)

    network = get_model(args, mode='test', **kwargs)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(_device)
    print('Model {} loaded to device {}.'.format(args.model_path, _device))

    log_file = None
    if args.test_list and args.out_dir:
        log_file = osp.join(args.out_dir, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.model_path + '\n')
            f.write('Seq mse angle_err norm_err\n')

    heading_mse, heading_angles = [], []

    for idx, data in enumerate(test_data_list):
        assert data == osp.split(seq_dataset.data_path[idx])[1]
        log_line = data

        feat, gt_heading = seq_dataset.get_test_seq(idx)
        feat = torch.Tensor(feat).to(_device)
        pred_heading, norm_err = network(feat)
        pred_heading, norm_err = pred_heading[0].cpu().detach().numpy(), norm_err.item()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, '{}_{}_heading.npy'.format(data, 'lstm')), np.concatenate([pred_heading, gt_heading], axis=1))

        vel = seq_dataset.velocities[idx]
        gt_vel_norm = np.clip(np.linalg.norm(vel, axis=1), a_max=1, a_min=0)

        mse, angle = compute_heading_error(pred_heading, gt_heading)

        heading_mse.append(mse)
        heading_angles.append(np.mean(angle))

        result = format_string(heading_mse[-1], heading_angles[-1])
        log_line = format_string(log_line, result, norm_err)
        if log_file is not None:
            with open(log_file, 'a') as f:
                log_line += '\n'
                f.write(log_line)

        print('{} :- heading mse: {:.3f} angle: {:.3f} norm: {:.3f}'.format(data, heading_mse[-1], heading_angles[-1], norm_err))

        absolute_angle = np.arctan2(pred_heading[:, 0], pred_heading[:, 1])
        gt_angle = np.arctan2(gt_heading[:, 0], gt_heading[:, 1])
        if not args.fast_test:
            plt.figure('Heading_Error {}'.format(data), figsize=(6, 8))

            plt.subplot(311)
            plt.title("Absolue Headings")
            plt.plot(adjust_angle_array(absolute_angle) * 180 / np.pi)
            plt.plot(adjust_angle_array(gt_angle) * 180 / np.pi)
            plt.legend(['predicted', 'gt'])

            plt.subplot(312)
            plt.title("Absolue Heading Errors")
            plt.plot(angle)
            plt.legend('mse: {:.3f}, angl: {:.3f}'.format(heading_mse[-1], heading_angles[-1]))

            plt.subplot(313)
            plt.title("Clipped gt velocity")
            plt.plot(np.clip(gt_vel_norm, a_min=0, a_max=1))
            plt.legend(['confidence', 'gt_velocity'])
            plt.tight_layout()

            if args.out_dir is not None:
                plt.savefig(osp.join(args.out_dir, args.prefix + data + '_output.png'))

        if args.use_trajectory_type == 'gt':
            traj = traj_from_velocity(vel)
        else:
            if osp.exists(osp.join(args.out_dir, '{}_{}.npy'.format(data, args.use_trajectory_type))):
                traj = np.load(osp.join(args.out_dir, '{}_{}.npy'.format(data, args.use_trajectory_type)))[:, :2]
            else:
                raise ValueError("Trajectory file {}_{}.npy is missing".format(data, args.use_trajectory_type))

        predicted = adjust_angle_array(absolute_angle)
        gt_angle = adjust_angle_array(gt_angle)
        g_l = {'m': ':', 'c': 'g'}
        p_l = {'m': '--', 'c': 'b'}
        handles =[mpatch.Patch(color=g_l['c'], label='Ground_truth'),
                  mpatch.Patch(color=p_l['c'], label='Predicted')]

        sh, l, h_w, window = 25, 2, 1, 75
        if not args.fast_test:
            plt.figure('Results_Plot {}'.format(data), figsize=(12, 12))

            plt.plot(traj[:, 0], traj[:, 1], color='black')
            for i in range(window, gt_heading.shape[0] - (window+sh), 1500):
                gt_i = np.median(gt_angle[i - window:i + window])
                p_i = np.median(predicted[i + sh - window:i + sh + window])
                # gt_i = gt_angle[i]
                # p_i = predicted[i + sh]
                plt.arrow(traj[i, 0], traj[i, 1], np.sin(gt_i) * l, np.cos(gt_i) * l, head_width=h_w, overhang=0.8, linestyle=g_l['m'], color=g_l['c'])
                plt.arrow(traj[i + sh, 0], traj[i + sh, 1], np.sin(p_i) * l, np.cos(p_i) * l, head_width=h_w, overhang=0.8, linestyle=p_l['m'],
                         color=p_l['c'])
            plt.axis('equal')
            plt.legend(handles=handles)
            plt.tight_layout()
            if args.out_dir is not None:
                plt.savefig(osp.join(args.out_dir, args.prefix + data + '_plot.png'))
            if args.show_plot:
                plt.show()

        plt.close('all')

    heading_mse = np.array(heading_mse)
    heading_angles = np.array(heading_angles)

    measure = format_string("MSE", "angle_err", sep='\t')
    values = format_string(np.mean(heading_mse), np.mean(heading_angles),
                           sep='\t')
    print(measure, '\n', values)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(measure + '\n')
            f.write(values)


if __name__ == '__main__':
    """
    Run file with individual arguments or/and config file. If argument appears in both config file and args, 
    args is given precedence.
    """
    default_config_file = osp.abspath(osp.join(osp.abspath(__file__), '../../config/heading_model_defaults.json'))

    import argparse

    parser = argparse.ArgumentParser(description="Run seq2seq heading model in train/test mode [required]. Optional "
                                                 "configurations can be specified as --key [value..] pairs",
                                     add_help=True)
    parser.add_argument('--config', type=str, help='Configuration file [Default: {}]'.format(default_config_file),
                        default=default_config_file)
    # common
    parser.add_argument('--data_dir', type=str, help='Directory for data files if different from list path.')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, help='Gaussian for smoothing features')
    parser.add_argument('--target_sigma', type=float, help='Gaussian for smoothing target')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, help='Cuda device e.g:- cuda:0')
    parser.add_argument('--cpu', action='store_const', dest='device', const='cpu')
    # lstm
    lstm_cmd = parser.add_argument_group('lstm', 'configuration for LSTM')
    lstm_cmd.add_argument('--layers', type=int)
    lstm_cmd.add_argument('--layer_size', type=int)

    mode = parser.add_subparsers(title='mode', dest='mode', help='Operation: [train] train model, [test] evaluate model')
    mode.required = True
    # train
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str)
    train_cmd.add_argument('--val_list', type=str)
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int)
    train_cmd.add_argument('--save_interval', type=int)
    train_cmd.add_argument('--lr', '--learning_rate', type=float)
    # test
    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_path', type=str, default=None)
    test_cmd.add_argument('--test_list', type=str, default=None)
    test_cmd.add_argument('--model_path', type=str, default=None)
    test_cmd.add_argument('--fast_test', action='store_true')
    test_cmd.add_argument('--show_plot', action='store_true')
    test_cmd.add_argument('--prefix', type=str, default='', help='prefix to add when saving files.')
    test_cmd.add_argument('--use_trajectory_type', type=str, default='gt',
                          help='Trajectory type to use when rendering the headings. (Default: gt). If not gt, the trajectory file is taken as <args.out_dir>/<data_name>_<use_trajectory_type>.npy with files generated in ronin_lstm_tcn.py or ronin_resnet.py')

    '''
    Extra arguments
    Set True: use_scheduler, quite (no output on stdout)
              force_lr (force lr when a model is loaded from continue_from),
              heading_norm (normalize heading),
              separate_loss (report loss separately for logging)
    float: dropout, max_ori_error (err. threshold for priority grv in degrees)
           max_velocity_norm (filter outliers in training)
           weights (array of float values) 
    '''
    args, unknown_args = parser.parse_known_args()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    args, kwargs = load_config(default_config_file, args, unknown_args)
    if args.mode == "train" and kwargs.get('weights') and type(kwargs.get('weights')) != list:
        kwargs['weights'] = [float(i) for i in kwargs.get('weights').split(',')]

    print(args, kwargs)
    if args.mode == 'train':
        train(args, **kwargs)
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("Model path required")
        args.batch_size = 1
        test(args, **kwargs)
