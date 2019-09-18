import argparse
import json
import os
import shutil
import sys
from os import path as osp

import h5py
import numpy as np
import pandas

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from preprocessing.gen_dataset_v2 import compute_output_time, process_data_source

'''
HDF5 data format
* data.hdf5
     |---raw
     |    |---tango
     |          |---gyro, gyro_uncalib, acce, magnet, game_rv, gravity, linacce, step, tango_pose, rv, 
     pressure, (optional) wifi, gps, magnetic_rv, magnet_uncalib
     |    |--- imu
     |          |---gyro, gyro_uncalib, acce, magnet, game_rv, gravity, linacce, step. rv, pressure, (optional) wifi, 
     gps, magnetic_rv, magnet_uncalib
     |--synced
     |    |---gyro, gyro_uncalib, acce, magnet, game_rv, rv, gravity, linacce, step
     |---pose
     |    |---tango_pos, tango_ori, (optional)ekf_ori
  The HDF5 file stores all data. "raw" subgroup store all unprocessed data. "synced" subgroup stores synchronized data
  (previous stores as "processed/data.pkl"). "pose" subgroup store all pose information, including corrected tango pose
  and (optional) EKF orientation.

* info.json
  Stores meta information, such as reference time, synchronization, calibration and orientation errors.

To read a HDF5 dataset:

import h5py
with h5py.File(<path-to-hdf5-file>) as f:
     gyro = f['synced/gyro']
     acce = f['synced/acce']
     tango_pos = f['pose/tango_pos']
     tango_ori = f['pose/tango_ori']
     .....

NOTICE: the HDF5 library will not read the data until it's actually used. For example, all data domains in the
        above code are NOT actually read from the disk. This means that if you try to access "gyro" or "acce" 
        etc. after the "with" closure is released, an error will occur. To avoid this issue, use:
               gyro = np.copy(f['synced/gyro'])
        to force reading.
'''

_raw_data_sources = ['gyro', 'gyro_uncalib', 'acce', 'magnet', 'game_rv', 'linacce', 'gravity', 'step', 'rv',
                     'pressure']
_optional_data_sources = ['wifi', 'gps', 'magnetic_rv', 'magnet_uncalib']
_synced_columns = {'time': 'time',
                   'gyro': ['gyro_x', 'gyro_y', 'gyro_z'],
                   'gyro_uncalib': ['gyro_uncalib_x', 'gyro_uncalib_y', 'gyro_uncalib_z'],
                   'acce': ['acce_x', 'acce_y', 'acce_z'],
                   'magnet': ['magnet_x', 'magnet_y', 'magnet_z'],
                   'game_rv': ['game_rv_w', 'game_rv_x', 'game_rv_y', 'game_rv_z'],
                   'rv': ['rv_w', 'rv_x', 'rv_y', 'rv_z'],
                   'grav': ['grav_x', 'grav_y', 'grav_z'],
                   'linacce': ['linacce_x', 'linacce_y', 'linacce_z']}
_device_list = ['asus1', 'asus2', 'asus3', 'pixel', 'samsung1', 'samsung2']
_nano_to_sec = 1e09
_micro_to_nano = 1000


def load_wifi_dataset(path):
    columns = ['scan', 'last_timestamp', 'BSSID', 'level']
    df = pandas.DataFrame(columns=columns)
    scan_no = 0
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            values = line.split()
            if len(values) == 1:
                scan_no += 1
            else:
                df = df.append(pandas.Series([scan_no, int(values[0]) * _micro_to_nano, values[1], int(values[2])],
                                             index=columns), ignore_index=True)
    df_mac_addr = np.array(df.values[:, 2])
    df_num = np.array(df.values[:, [0, 1, 3]], dtype=np.int)
    return df_num, df_mac_addr


def compile_annotated_sequence(root_dir, data_list, args):
    fail_list = []
    for data in data_list:
        try:
            data_path = osp.join(root_dir, data)
            out_path = osp.join(args.out_dir, data)
            if osp.isdir(out_path):
                if args.overwrite:
                    shutil.rmtree(out_path)
                    os.makedirs(out_path)
                else:
                    print('-- {} exists.'.format(data))
                    continue
                    # raise ValueError('Output folder {} exists.'.format(out_path))
            else:
                os.makedirs(out_path)
            print('Compiling {} to {}'.format(data_path, out_path))

            imu_all = pandas.read_pickle(osp.join(data_path, 'processed/data.pkl'))
            gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
            gyro_uncalib = imu_all[['gyro_uncalib_x', 'gyro_uncalib_y', 'gyro_uncalib_z']].values
            init_gyro_bias = gyro_uncalib[0] - gyro[0]
            end_gyro_bias = np.loadtxt(osp.join(data_path, 'imu/gyro_bias.txt'))

            # Write meta info
            length = imu_all['time'].values[-1] - imu_all['time'].values[0]
            ori_error = json.load(open(osp.join(data_path, 'processed/ori_error.json')))
            calibrations = np.loadtxt(osp.join(data_path, 'processed/calibration.txt'))
            imu_time_offset = np.loadtxt(osp.join(data_path, 'processed/imu_time_offset.txt'))
            body_align = np.loadtxt(osp.join(data_path, 'processed/align_tango_to_body.txt'))
            device = 'unknown'
            with open(osp.join(data_path, 'imu/acce_calib.txt')) as f:
                line = f.readline().split()
                date = line[3]
                if line[-1] in _device_list:
                    device = line[-1]
            acce_calib = np.loadtxt(osp.join(data_path, 'imu/acce_calib.txt'))

            meta_info = {'type': 'annotated',
                         'length': length,
                         'date': date,
                         'device': device,
                         'imu_reference_time': np.loadtxt(osp.join(data_path, 'imu/reference_time.txt')).tolist(),
                         'tango_reference_time': np.loadtxt(osp.join(data_path,
                                                                     'tango/reference_time.txt')).tolist(),
                         'imu_init_gyro_bias': init_gyro_bias.tolist(),
                         'imu_end_gyro_bias': end_gyro_bias.tolist(),
                         'imu_acce_bias': acce_calib[0].tolist(),
                         'imu_acce_scale': acce_calib[1].tolist(),
                         'gyro_integration_error': ori_error['gyro_integration'],
                         'grv_ori_error': ori_error['game_rv'],
                         'ekf_ori_error': ori_error['ekf'],
                         'start_calibration': calibrations[0].tolist(),
                         'end_calibration': calibrations[1].tolist(),
                         'imu_time_offset': imu_time_offset.tolist(),
                         'align_tango_to_body': body_align.tolist()}
            json.dump(meta_info, open(osp.join(out_path, 'info.json'), 'w'))

            with h5py.File(osp.join(out_path, 'data.hdf5'), 'x') as f:
                f.create_group('raw/imu')
                for source in _raw_data_sources:
                    f.create_dataset('raw/imu/' + source,
                                     data=np.genfromtxt(osp.join(data_path, 'imu/' + source + '.txt')))
                for source in _optional_data_sources:
                    if osp.isfile(osp.join(data_path, 'imu/' + source + '.txt')):
                        if source == 'wifi':
                            data_num, data_string = load_wifi_dataset(osp.join(data_path, 'imu/' + source + '.txt'))
                            f.create_dataset('raw/imu/wifi_values', data=data_num)
                            f.create_dataset('raw/imu/wifi_address', data=data_string,
                                             dtype=h5py.special_dtype(vlen=str))
                        else:
                            f.create_dataset('raw/imu/' + source,
                                             data=np.genfromtxt(osp.join(data_path, 'imu/' + source + '.txt')))

                f.create_group('raw/tango')
                for source in _raw_data_sources:
                    f.create_dataset('raw/tango/' + source,
                                     data=np.genfromtxt(osp.join(data_path, 'tango/' + source + '.txt')))
                for source in _optional_data_sources:
                    if osp.isfile(osp.join(data_path, 'tango/' + source + '.txt')):
                        if source == 'wifi':
                            data_num, data_string = load_wifi_dataset(osp.join(data_path, 'tango/' + source + '.txt'))
                            f.create_dataset('raw/tango/wifi_values', data=data_num)
                            f.create_dataset('raw/tango/wifi_address', data=data_string, dtype=h5py.special_dtype(
                                vlen=str))
                        else:
                            f.create_dataset('raw/tango/' + source,
                                             data=np.genfromtxt(osp.join(data_path, 'tango/' + source + '.txt')))

                f.create_dataset('raw/tango/tango_pose',
                                 data=np.genfromtxt(osp.join(data_path, 'tango/pose.txt')))
                f.create_dataset('raw/tango/tango_adf_pose',
                                 data=np.genfromtxt(osp.join(data_path, 'tango/pose_adf.txt')))

                f.create_group('synced')
                for synced_key, columns in _synced_columns.items():
                    arr = imu_all[columns].values
                    f.create_dataset('synced/' + synced_key, data=arr)

                f.create_group('pose')
                corrected_tango = np.load(osp.join(data_path, 'processed/corrected_tango.npy'))
                f.create_dataset('pose/tango_pos', data=corrected_tango[:, :3])
                f.create_dataset('pose/tango_ori', data=corrected_tango[:, 3:7])

                if osp.exists(osp.join(data_path, 'processed/ekf_ori.npy')):
                    ekf_ori = np.load(osp.join(data_path, 'processed/ekf_ori.npy'))
                    f.create_dataset('pose/ekf_ori', data=ekf_ori)
        except (FileNotFoundError, OSError, TypeError) as e:
            print(e)
            fail_list.append(data)

    print('Fail list:')
    [print(data) for data in fail_list]
    with open(osp.join(root_dir, 'compile_fail_list.txt'), 'w') as f:
        for data in fail_list:
            f.write(data + '\n')


def compile_unannotated_sequence(root_dir, data_list, args):
    """
    Compile unannotated(or imu_only) sequence directly from raw files.
    """
    source_vector = {'gyro', 'gyro_uncalib', 'acce', 'linacce', 'gravity', 'magnet'}
    source_quaternion = {'game_rv', 'rv'}
    source_all = source_vector.union(source_quaternion)
    fail_list = []
    for data in data_list:
        try:
            data_path = osp.join(root_dir, data)
            out_path = osp.join(args.out_dir, data)
            if osp.isdir(out_path):
                if args.overwrite:
                    shutil.rmtree(out_path)
                    os.makedirs(out_path)
                else:
                    raise ValueError('Output folder {} exists.'.format(out_path))
            else:
                os.makedirs(out_path)

            print('Compiling {} to {}'.format(data_path, out_path))

            all_sources = {}
            # We assume that all IMU/magnetic/pressure files must exist.
            reference_time = np.loadtxt(osp.join(data_path, 'reference_time.txt')).tolist()
            for source in source_all:
                try:
                    source_path = osp.join(root_dir, data, source + '.txt')
                    source_data = np.genfromtxt(source_path)
                    source_data[:, 0] = (source_data[:, 0] - reference_time) / _nano_to_sec
                    all_sources[source] = source_data
                except OSError:
                    print('Can not find file for source {}. Please check the dataset.'.format(source_path))
                    continue

            output_time = compute_output_time(all_sources)
            processed_sources = {}
            for source in all_sources.keys():
                if source in source_vector:
                    processed_sources[source] = process_data_source(all_sources[source], output_time, 'vector')
                else:
                    processed_sources[source] = process_data_source(
                        all_sources[source][:, [0, 4, 1, 2, 3]], output_time, 'quaternion')

            init_gyro_bias = processed_sources['gyro_uncalib'][0] - processed_sources['gyro'][0]
            end_gyro_bias = np.loadtxt(osp.join(data_path, 'gyro_bias.txt'))
            device = 'unknown'
            with open(osp.join(data_path, 'acce_calib.txt')) as f:
                line = f.readline().split()
                date = line[3]
                if line[-1] in _device_list:
                    device = line[-1]
            acce_calib = np.loadtxt(osp.join(data_path, 'acce_calib.txt'))

            meta_info = {'type': 'unannotated',
                         'length': output_time[-1] - output_time[0],
                         'date': date,
                         'device': device,
                         'imu_reference_time': reference_time,
                         'imu_init_gyro_bias': init_gyro_bias.tolist(),
                         'imu_end_gyro_bias': end_gyro_bias.tolist(),
                         'imu_acce_bias': acce_calib[0].tolist(),
                         'imu_acce_scale': acce_calib[1].tolist()}
            json.dump(meta_info, open(osp.join(out_path, 'info.json'), 'w'))

            with h5py.File(osp.join(out_path, 'data.hdf5'), 'x') as f:
                f.create_group('raw/imu')
                for source in _raw_data_sources:
                    f.create_dataset('raw/imu/' + source,
                                     data=np.genfromtxt(osp.join(data_path, source + '.txt')))
                f.create_group('synced')
                f.create_dataset('synced/time', data=output_time)
                for source in source_all:
                    if source == 'gravity':
                        f.create_dataset('synced/' + source, data=processed_sources['gravity'])
                    else:
                        f.create_dataset('synced/' + source, data=processed_sources[source])

        except (OSError, FileNotFoundError, TypeError) as e:
            print(e)
            fail_list.append(data)

    print('Fail list:')
    [print(data) for data in fail_list]
    with open(osp.join(root_dir, 'compile_fail_list.txt'), 'w') as f:
        for data in fail_list:
            f.write(data + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--list', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--type', type=str, choices=['annotated', 'unannotated'], default='annotated')

    args = parser.parse_args()

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.path is not None:
        if args.path[-1] == '/':
            args.path = args.path[:-1]
        root_dir = osp.split(args.path)[0]
        data_list = [osp.split(args.path)[1]]
    elif args.list is not None:
        root_dir = osp.split(args.list)[0]
        with open(args.list) as f:
            data_list = [s.strip().split()[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either --path or --list must be specified')

    if args.type == 'annotated':
        compile_annotated_sequence(root_dir, data_list, args)
    elif args.type == 'unannotated':
        compile_unannotated_sequence(root_dir, data_list, args)
    else:
        raise ValueError
