import numpy as np
import matplotlib.pyplot as plt

from icp import icp_matching


# Bresenhams Line Generation Algorithm
# ref: https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
def bresenham(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    steep = 0
    if dx <= dy:
        steep = 1
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    pk = 2 * dy - dx

    loc = []
    for _ in range(0, dx + 1):
        if steep == 0:
            loc.append([x1, y1])
        else:
            loc.append([y1, x1])

        if x1 < x2:
            x1 = x1 + 1
        else:
            x1 = x1 - 1

        if (pk < 0):
            if steep == 0:
                pk = pk + 2 * dy
            else:
                pk = pk + 2 * dy
        else:
            if y1 < y2:
                y1 = y1 + 1
            else:
                y1 = y1 - 1
    
            pk = pk + 2 * dy - 2 * dx

    return loc


def data_association(prev_points, curr_points, prev_idx, curr_idx):
    assert len(prev_idx) == len(curr_idx)

    prev_points = np.array(prev_points)
    curr_points = np.array(curr_points)

    prev_points_matched, curr_points_matched = [], []
    for i in range(len(prev_idx)):
        min_range = min(len(prev_idx[i]), len(curr_idx[i]))
        prev_points_matched.extend(prev_points[np.array(prev_idx[i])[:min_range]])
        curr_points_matched.extend(curr_points[np.array(curr_idx[i])[:min_range]])

    return np.array(prev_points_matched), np.array(curr_points_matched)


def scan_matching(prev_points, curr_points, prev_idx, curr_idx, pose):
    if len(prev_points) < 5 or len(curr_points) < 5:
        return None

    prev_points, curr_points = data_association(prev_points, curr_points, prev_idx, curr_idx)

    R, t = icp_matching(prev_points.T, curr_points.T)

    if t[0] > 5 or t[1] > 5:
        return None
    
    x = pose[0] + t[0]
    y = pose[1] + t[1]
    orientation = wrapAngle(pose[2] + np.arctan2(R[1][0], R[0][0]))

    return np.array((x, y, orientation))


def wrapAngle(radian):
    radian = radian - 2 * np.pi * np.floor((radian + np.pi) / (2 * np.pi))
    return radian
    

def prob2logodds(prob):
    return np.log(prob / (1 - prob + 1e-15))


def logodds2prob(logodds):
    return 1 - 1 / (1 + np.exp(logodds) + 1e-15)


def normalDistribution(mean, variance):
    return np.exp(-(np.power(mean, 2) / variance / 2.0) / np.sqrt(2.0 * np.pi * variance))


def create_rotation_matrix(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    R_inv = np.linalg.inv(R)

    return R, R_inv


def rotate(center, vector, R):
    vector = (vector - center) @ R.T + center
    return vector


def visualize(robot, particles, best_particle, world, radar_list, step, offset):
    plt.suptitle("Fast SLAM 1.0", y=0.9)
    plt.title("number of particles:{}, step:{}".format(len(particles), step + 1))
    grid_size = best_particle.grid_size
    plt.xlim(0, grid_size[0])
    plt.ylim(0, grid_size[1])

    # draw map
    #world_map = 1 - robot.grid
    world_map = 1 - best_particle.grid
    plt.imshow(world_map, cmap='gray')

    # draw radar beams
    for (x, y) in radar_list:
        plt.plot(x + offset[0], y + offset[1], "yo", markersize=1)

    # draw tragectory
    true_path = np.array(robot.trajectory)
    estimated_path = np.array(best_particle.trajectory)
    plt.plot(true_path[:, 0] + offset[0], true_path[:, 1] + offset[1], "b")
    plt.plot(estimated_path[:, 0], estimated_path[:, 1], "g")

    # draw robot position
    plt.plot(robot.x + offset[0], robot.y + offset[1], "bo")

    # draw particles position
    for p in particles:
        plt.plot(p.x, p.y, "go", markersize=1)

    plt.pause(0.01)
    plt.draw()
    plt.clf()
