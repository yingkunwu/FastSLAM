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


def scan_matching(prev_points, curr_points, pose):
    if len(prev_points) < 5 or len(curr_points) < 5 or len(prev_points) < len(curr_points):
        return None
    
    # delete duplicates
    curr_points = np.unique(curr_points, axis=0)

    R, t = icp_matching(prev_points.T, curr_points.T)

    if abs(t[0]) > 5 or abs(t[1]) > 5:
        return None
    else:
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


def absolute2relative(position, states):
    x, y, theta = states
    pose = np.array([x, y])

    R, R_inv = create_rotation_matrix(theta)
    position = position - pose
    position = np.array(position) @ R_inv.T

    return position


def relative2absolute(position, states):
    x, y, theta = states
    pose = np.array([x, y])

    R, R_inv = create_rotation_matrix(theta)
    position = np.array(position) @ R.T
    position = position + pose

    return position


def visualize(robot, particles, best_particle, radar_list, config, step, title, output_path, visualize=False):
    plt.clf()
    plt.suptitle(title)
    plt.title("number of particles:{}, step:{}".format(len(particles), step + 1))
    grid_size = best_particle.grid_size
    plt.xlim(0, grid_size[1])
    plt.ylim(0, grid_size[0])

    Rx, Ry, _ = config['R_init']
    px, py, _ = config['p_init']
    offset = [px - Rx, py - Ry]

    # draw map
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

    if step % 10 == 0:
        plt.savefig('{}_{}.png'.format(output_path, step))

    if visualize:
        plt.draw()
        plt.pause(0.01)
