import numpy as np
import matplotlib.pyplot as plt


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


# Iterative Closest Point Algorithm
# Ref: https://github.com/AtsushiSakai/PythonRobotics/blob/53eae53b5a78a08b7ce4c6ffeed727c1d6a0ab2e/SLAM/iterative_closest_point/iterative_closest_point.py
def icp(prev_scan, curr_scan, pose):
    prev_points, curr_points = [], []
    for p_s, c_s in zip(prev_scan, curr_scan):
        if p_s and c_s:
            prev_points.append(p_s)
            curr_points.append(c_s)

    if len(prev_points) == 0:
        return None
    
    prev_points = np.array(prev_points) - pose[:2]
    curr_points = np.array(curr_points) - pose[:2]

    prev_mean = np.mean(prev_points, axis=0)
    curr_mean = np.mean(curr_points, axis=0)
    
    prev_shift = prev_points - prev_mean
    curr_shift = curr_points - curr_mean

    W = curr_shift.T @ prev_shift
    u, s, vh = np.linalg.svd(W)

    if np.sum(s == 0) > 0:
        return None
    
    R = (u @ vh).T
    t = curr_mean - (R @ prev_mean)

    x = pose[0] + t[0]
    y = pose[1] + t[1]
    orientation = wrapAngle(pose[2] + np.arctan2(R[1][0], R[0][0]))

    return (x, y, orientation)


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


def visualize(robot, particles, best_particle, world, radar_list, true_path, estimated_path, step):
    plt.suptitle("Fast SLAM 1.0", y=0.9)
    plt.title("number of particles:{}, step:{}".format(len(particles), step + 1))
    plt.xlim(0, world.size_x)
    plt.ylim(0, world.size_y)

    # draw map
    #world_map = 1 - robot.grid
    world_map = 1 - best_particle.grid
    plt.imshow(world_map, cmap='gray')

    # draw radar beams
    for (x, y) in radar_list:
        plt.plot(x, y, "yo", markersize=1)

    # draw tragectory
    true_path = np.array(true_path)
    estimated_path = np.array(estimated_path)
    plt.plot(true_path[:, 0], true_path[:, 1], "b")
    plt.plot(estimated_path[:, 0], estimated_path[:, 1], "g")


    # draw robot position
    plt.plot(robot.x, robot.y, "bo")

    # draw particles position
    for p in particles:
        plt.plot(p.x, p.y, "go", markersize=1)

    plt.pause(0.01)
    plt.draw()
    plt.clf()
