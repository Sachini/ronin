import numpy as np
import plyfile
import quaternion


def write_ply_to_file(path, position, orientation, acceleration=None,
                      global_rotation=np.identity(3, float), local_axis=None,
                      trajectory_color=None, num_axis=3,
                      length=1.0, kpoints=100, interval=100):
    """
    Visualize camera trajectory as ply file
    :param path: path to save
    :param position: Nx3 array of positions
    :param orientation: Nx4 array or orientation as quaternion
    :param acceleration: (optional) Nx3 array of acceleration
    :param global_rotation: (optional) global rotation
    :param local_axis: (optional) local axis vector
    :param trajectory_color: (optional) the color of the trajectory. The default is [255, 0, 0] (red)
    :return: None
    """
    num_cams = position.shape[0]
    assert orientation.shape[0] == num_cams

    max_acceleration = 1.0
    if acceleration is not None:
        assert acceleration.shape[0] == num_cams
        max_acceleration = max(np.linalg.norm(acceleration, axis=1))
        print('max_acceleration: ', max_acceleration)
        num_axis = 4

    sample_pt = np.arange(0, num_cams, interval, dtype=int)
    num_sample = sample_pt.shape[0]

    # Define the optional transformation. Default is set w.r.t tango coordinate system
    position_transformed = np.matmul(global_rotation, np.array(position).transpose()).transpose()
    if local_axis is None:
        local_axis = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]])
    if trajectory_color is None:
        trajectory_color = [0, 255, 255]
    axis_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]
    vertex_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    positions_data = np.empty((position_transformed.shape[0],), dtype=vertex_type)
    positions_data[:] = [tuple([*i, *trajectory_color]) for i in position_transformed]

    app_vertex = np.empty([num_axis * kpoints], dtype=vertex_type)
    for i in range(num_sample):
        q = quaternion.quaternion(*orientation[sample_pt[i]])
        if acceleration is not None:
            local_axis[:, -1] = acceleration[sample_pt[i]].flatten() / max_acceleration
        global_axes = np.matmul(global_rotation, np.matmul(quaternion.as_rotation_matrix(q), local_axis))
        for k in range(num_axis):
            for j in range(kpoints):
                axes_pts = position_transformed[sample_pt[i]].flatten() +\
                           global_axes[:, k].flatten() * j * length / kpoints
                app_vertex[k*kpoints + j] = tuple([*axes_pts, *axis_color[k]])

        positions_data = np.concatenate([positions_data, app_vertex], axis=0)
    vertex_element = plyfile.PlyElement.describe(positions_data, 'vertex')
    plyfile.PlyData([vertex_element], text=True).write(path)
