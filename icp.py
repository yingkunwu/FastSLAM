# Iterative Closest Point Algorithm
# Ref: https://github.com/AtsushiSakai/PythonRobotics/blob/53eae53b5a78a08b7ce4c6ffeed727c1d6a0ab2e/SLAM/iterative_closest_point/iterative_closest_point.py

import numpy as np
import matplotlib.pyplot as plt


EPS = 0.0001
MAX_ITER = 100
show_animation = False

if show_animation:
    fig = plt.figure()

def icp_matching(previous_points, current_points):
    """
    Iterative Closest Point matching
    - input
    previous_points: 2D or 3D points in the previous frame
    current_points: 2D or 3D points in the current frame
    - output
    R: Rotation matrix
    T: Translation vector
    """
    H = np.diag([1, 1, 1])  # homogeneous transformation matrix

    dError = np.inf
    preError = np.inf
    count = 0

    while dError >= EPS:
        count += 1

        previous_points_matched, current_points_matched, total_error = nearest_neighbor_association(previous_points, current_points)
        if previous_points_matched.shape[1] < 5:
            return None, [100, 100]

        if show_animation:  # pragma: no cover
            plot_points(previous_points_matched, current_points_matched, fig)

        # perform RANSAC
        min_error = np.float('inf')
        best_Rt = None
        best_Tt = None
        for _ in range(15):
            sample = np.random.choice(current_points_matched.shape[1], 5, replace=False)
            
            Rt, Tt = svd_motion_estimation(previous_points_matched[:, sample], current_points_matched[:, sample])
            temp_points = (Rt @ previous_points_matched) + Tt[:, np.newaxis]
            _, _, error = nearest_neighbor_association(temp_points, current_points_matched)
            if error < min_error:
                min_error = error
                best_Rt = Rt
                best_Tt = Tt

        # update current points
        previous_points = (best_Rt @ previous_points) + best_Tt[:, np.newaxis]

        dError = preError - total_error
        #print("Residual:", error)

        preError = total_error
        H = update_homogeneous_matrix(H, best_Rt, best_Tt)

        if MAX_ITER <= count:
            break

    R = np.array(H[0:-1, 0:-1])
    T = np.array(H[0:-1, -1])

    return R, T


def update_homogeneous_matrix(Hin, R, T):
    H = np.zeros((3, 3))

    H[0:2, 0:2] = R
    H[0:2, 2] = T
    H[2, 2] = 1.0

    return Hin @ H


def nearest_neighbor_association(prev_points, curr_points):
    d = np.linalg.norm(np.repeat(curr_points, prev_points.shape[1], axis=1)
                       - np.tile(prev_points, (1, curr_points.shape[1])), axis=0)
    d = d.reshape(curr_points.shape[1], prev_points.shape[1])

    indexes = np.argmin(d, axis=1)
    error = np.min(d, axis=1)

    return prev_points[:, indexes], curr_points, np.sum(error)


def svd_motion_estimation(previous_points, current_points):
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)

    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]

    W = c_shift @ p_shift.T
    u, s, vh = np.linalg.svd(W)

    R = u @ vh.T
    t = cm - (R @ pm)

    return R, t


def plot_points(previous_points, current_points, figure):
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    if previous_points.shape[0] == 3:
        plt.clf()
        axes = figure.add_subplot(111, projection='3d')
        axes.scatter(previous_points[0, :], previous_points[1, :],
                    previous_points[2, :], c="r", marker=".")
        axes.scatter(current_points[0, :], current_points[1, :],
                    current_points[2, :], c="b", marker=".")
        axes.scatter(0.0, 0.0, 0.0, c="r", marker="x")
        figure.canvas.draw()
    else:
        plt.cla()
        plt.plot(previous_points[0, :], previous_points[1, :], ".r", markersize=1)
        plt.plot(current_points[0, :], current_points[1, :], ".b", markersize=1)
        plt.plot(0.0, 0.0, "xr")
        plt.axis("equal")

    plt.pause(0.01)
    plt.draw()
    plt.clf()
