# Iterative Closest Point Algorithm
# Ref: https://github.com/AtsushiSakai/PythonRobotics/blob/53eae53b5a78a08b7ce4c6ffeed727c1d6a0ab2e/SLAM/iterative_closest_point/iterative_closest_point.py

import numpy as np
import matplotlib.pyplot as plt

from utils import wrapAngle


EPS = 0.0001
MAX_ITER = 100
show_animation = False

if show_animation:
    fig = plt.figure()


def icp_matching(edges, scan, pose):
    if len(scan) < 5 or len(edges) < len(scan):
        return None
    
    # delete duplicate scans
    scan = np.unique(scan, axis=0)

    # transpose edges and scan to match the implementation of algorithm
    edges = edges.T
    scan = scan.T

    H = np.diag([1, 1, 1])  # homogeneous transformation matrix

    dError = np.inf
    preError = np.inf
    count = 0

    while dError >= EPS:
        count += 1

        indexes, total_error = nearest_neighbor_association(edges, scan)
        edges_matched = edges[:, indexes]

        if show_animation:
            plot_points(edges_matched, scan, fig)

        # perform RANSAC
        min_error = np.float('inf')
        best_Rt = None
        best_Tt = None
        for _ in range(15):
            sample = np.random.choice(scan.shape[1], 5, replace=False)
            
            Rt, Tt = svd_motion_estimation(edges_matched[:, sample], scan[:, sample])
            temp_points = (Rt @ scan) + Tt[:, np.newaxis]
            _, error = nearest_neighbor_association(edges_matched, temp_points)
            if error < min_error:
                min_error = error
                best_Rt = Rt
                best_Tt = Tt

        # update current scan for iterative refinement
        scan = (best_Rt @ scan) + best_Tt[:, np.newaxis]

        dError = preError - total_error
        #print("Residual:", error)

        preError = total_error
        H = update_homogeneous_matrix(H, best_Rt, best_Tt)

        if MAX_ITER <= count:
            break

    R = np.array(H[0:-1, 0:-1])
    T = np.array(H[0:-1, -1])

    if abs(T[0]) > 5 or abs(T[1]) > 5:
        return None
    else:
        x = pose[0] + T[0]
        y = pose[1] + T[1]
        orientation = wrapAngle(pose[2] + np.arctan2(R[1][0], R[0][0]))

        return np.array((x, y, orientation))


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

    return indexes, np.sum(error)


def svd_motion_estimation(previous_points, current_points):
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)

    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]

    W = c_shift @ p_shift.T
    u, s, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

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
