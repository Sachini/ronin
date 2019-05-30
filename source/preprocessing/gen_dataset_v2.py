# pylint: disable=C0103,C0111,C0301

import argparse
import os
import sys
from os import path as osp

import numpy as np
import pandas
import scipy.interpolate

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from math_util import interpolate_quaternion_linear
from preprocessing.write_trajectory_to_ply import write_ply_to_file

_nano_to_sec = 1e09


def interpolate_vector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.

    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def process_data_source(raw_data, output_time, method):
    input_time = raw_data[:, 0]
    if method == 'vector':
        output_data = interpolate_vector_linear(raw_data[:, 1:], input_time, output_time)
    elif method == 'quaternion':
        assert raw_data.shape[1] == 5
        output_data = interpolate_quaternion_linear(raw_data[:, 1:], input_time, output_time)
    else:
        raise ValueError('Interpolation method must be "vector" or "quaternion"')
    return output_data


def compute_output_time(all_sources, sample_rate=200):
    """
    Compute the output reference time from all data sources. The reference time range must be within the time range of
    all data sources.
    :param data_all:
    :param sample_rate:
    :return:
    """
    interval = 1. / sample_rate
    min_t = max([data[0, 0] for data in all_sources.values()]) + interval
    max_t = min([data[-1, 0] for data in all_sources.values()]) - interval
    return np.arange(min_t, max_t, interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default=None, help='Path to a list file.')
    parser.add_argument('--path', type=str, default=None, help='Path to a dataset folder.')
    parser.add_argument('--skip_front', type=int, default=200, help='Number of discarded records at beginning.')
    parser.add_argument('--skip_end', type=int, default=200, help='Numbef of discarded records at end')
    parser.add_argument('--output_samplerate', type=int, default=200, help='Output sample rate. Default is 200Hz')
    parser.add_argument('--recompute', action='store_true',
                        help='When set, the previously computed results will be over-written.')
    parser.add_argument('--no_trajectory', action='store_true',
                        help='When set, no ply files will be written.')
    parser.add_argument('--no_remove_duplicate', action='store_true')
    parser.add_argument('--clear_result', action='store_true')
    parser.add_argument('--fast_mode', action='store_true')

    args = parser.parse_args()

    dataset_list = []
    root_dir = ''
    if args.path:
        dataset_list.append(args.path)
    elif args.list:
        root_dir = os.path.dirname(args.list) + '/'
        with open(args.list) as f:
            for s in f.readlines():
                if s[0] is not '#':
                    dataset_list.append(s.strip('\n'))
    else:
        raise ValueError('No data specified')

    print(dataset_list)

    total_length = 0.0
    length_dict = {}
    for dataset in dataset_list:
        if len(dataset.strip()) == 0:
            continue
        if dataset[0] == '#':
            continue
        info = dataset.split(',')
        motion_type = 'unknown'
        if len(info) == 2:
            motion_type = info[1]
        data_root = root_dir + info[0]
        length = 0
        if os.path.exists(data_root + '/processed/data.csv') and not args.recompute:
            data_pandas = pandas.read_csv(data_root + '/processed/data.csv')
        elif os.path.exists(data_root + '/processed/data.csv') and not args.recompute:
            data_pandas = pandas.read_pickle(data_root + '/processed/data.pkl')
        else:
            print('------------------\nProcessing ' + data_root, ', type: ' + motion_type)
            all_sources = {}
            source_vector = {'gyro', 'gyro_uncalib', 'acce', 'linacce', 'gravity', 'magnet'}
            source_quaternion = {'rv', 'game_rv'}

            reference_time = 0
            source_all = source_vector.union(source_quaternion)
            for source in source_all:
                try:
                    source_data = np.genfromtxt(osp.join(data_root, source + '.txt'))
                    source_data[:, 0] = (source_data[:, 0] - reference_time) / _nano_to_sec
                    all_sources[source] = source_data
                except OSError:
                    print('Can not find file for source {}. Please check the dataset.'.format(source))
                    exit(1)

            if osp.exists(osp.join(data_root, 'pose.txt')):
                source_data = np.genfromtxt(osp.join(data_root, 'pose.txt'))
                source_data[:, 0] = (source_data[:, 0] - reference_time) / _nano_to_sec
                all_sources['pose'] = source_data
            if osp.exists(osp.join(data_root, 'pose_adf.txt')):
                source_data = np.genfromtxt(osp.join(data_root, 'pose_adf.txt'))
                source_data[:, 0] = (source_data[:, 0] - reference_time) / _nano_to_sec
                all_sources['pose_adf'] = source_data

            for src_id, src in all_sources.items():
                print('Source: %s,  start time: %d, end time: %d' % (src_id, src[0, 0], src[-1, 0]))

            output_time = compute_output_time(all_sources, args.output_samplerate)
            if motion_type not in length_dict:
                length_dict[motion_type] = 0
            length_dict[motion_type] += output_time[-1] - output_time[0]

            processed_source = {}
            for source in all_sources.keys():
                if 'pose' in source:
                    continue
                raw_data = all_sources[source]
                input_sr = (raw_data.shape[0] - 1) / (raw_data[-1, 0] - raw_data[0, 0])
                if source in source_vector:
                    processed_source[source] = process_data_source(all_sources[source], output_time, 'vector')
                elif source in source_quaternion:
                    # The Android API gives quaternion in the order of xyzw, we need to convert it to wxyz.
                    all_sources[source][:, [1, 2, 3, 4]] = all_sources[source][:, [4, 1, 2, 3]]
                    processed_source[source] = process_data_source(all_sources[source], output_time, 'quaternion')
                print('{} found. Input sampling rate: {}Hz. Channel size:{}'.format(source, input_sr,
                                                                                    processed_source[source].shape[1]))


            # The only file that could be missing is the Tango pose. If so, we fill two arrays with zero values.
            def add_pose_data(key_id, output_name):
                processed_source[output_name + '_pos'] = np.zeros([output_time.shape[0], 3])
                processed_source[output_name + '_ori'] = np.zeros([output_time.shape[0], 4])
                if key_id in all_sources:
                    pose_data = all_sources[key_id][:, :4]
                    ori_data = all_sources[key_id][:, [0, -1, -4, -3, -2]]
                    input_sr = pose_data.shape[0] / (pose_data[-1, 0] - pose_data[0, 0])
                    print('{} found. Input sampling rate: {}Hz'.format(output_name, input_sr))
                    processed_source[output_name + '_pos'] = process_data_source(pose_data, output_time, 'vector')
                    processed_source[output_name + '_ori'] = process_data_source(ori_data, output_time, 'quaternion')


            add_pose_data('pose', 'tango')
            add_pose_data('pose_adf', 'tango_adf')
            processed_source['magnetic_rv'] = np.zeros([output_time.shape[0], 4])
            processed_source['pressure'] = np.zeros([output_time.shape[0], 1])

            # construct a Pandas DataFrame
            column_list = 'time,' \
                          'gyro_x,gyro_y,gyro_z,' \
                          'gyro_uncalib_x,gyro_uncalib_y,gyro_uncalib_z,' \
                          'acce_x,acce_y,acce_z,' \
                          'linacce_x,linacce_y,linacce_z,' \
                          'grav_x,grav_y,grav_z,' \
                          'magnet_x,magnet_y,magnet_z,' \
                          'rv_w,rv_x,rv_y,rv_z,' \
                          'game_rv_w,game_rv_x,game_rv_y,game_rv_z,' \
                          'magnetic_rv_w,magnetic_rv_x,magnetic_rv_y,magnetic_rv_z,' \
                          'pressure,' \
                          'tango_pos_x,tango_pos_y,tango_pos_z,' \
                          'tango_ori_w,tango_ori_x,tango_ori_y,tango_ori_z,' \
                          'tango_adf_pos_x,tango_adf_pos_y,tango_adf_pos_z,' \
                          'tango_adf_ori_w,tango_adf_ori_x,tango_adf_ori_y,tango_adf_ori_z'.split(',')
            data_mat = np.concatenate([output_time[:, None],
                                       processed_source['gyro'],
                                       processed_source['gyro_uncalib'][:, :3],
                                       processed_source['acce'],
                                       processed_source['linacce'],
                                       processed_source['gravity'],
                                       processed_source['magnet'],
                                       processed_source['rv'],
                                       processed_source['game_rv'],
                                       processed_source['magnetic_rv'],
                                       processed_source['pressure'],
                                       processed_source['tango_pos'],
                                       processed_source['tango_ori'],
                                       processed_source['tango_adf_pos'],
                                       processed_source['tango_adf_ori']], axis=1)
            data_mat = data_mat[args.skip_front:-args.skip_end]
            output_folder = osp.join(data_root, 'processed')
            if not osp.isdir(output_folder):
                os.makedirs(output_folder)
            data_pandas = pandas.DataFrame(data_mat, columns=column_list)
            data_pandas.to_pickle(output_folder + '/data.pkl')
            print('Dataset written to ' + output_folder + '/data.csv')

            if not args.fast_mode and not args.no_trajectory and 'pose' in all_sources:
                tango_pos = processed_source['tango_pos'][args.skip_front:-args.skip_end]
                tango_ori = processed_source['tango_ori'][args.skip_front:-args.skip_end]
                print("Writing trajectory to ply file")
                write_ply_to_file(path=output_folder + '/trajectory.ply', position=tango_pos, orientation=tango_ori,
                                  num_axis=0)

                if osp.exists(osp.join(args.path, 'pose_adf.txt')):
                    tango_adf_pos = processed_source['tango_adf_pos'][args.skip_front:-args.skip_end]
                    tango_adf_ori = processed_source['tango_adf_ori'][args.skip_front:-args.skip_end]
                    print("Writing adf trajectory to ply file")
                    write_ply_to_file(path=output_folder + '/trajectory_adf.ply', position=tango_adf_pos,
                                      orientation=tango_adf_ori, trajectory_color=(200, 128, 255),
                                      num_axis=0)
    print('All done. Total length: {:.2f}s ({:.2f}min)'.format(total_length, total_length / 60.0))
    for k, v in length_dict.items():
        print(k + ': {:.2f}s ({:.2f}min)'.format(v, v / 60.0))
