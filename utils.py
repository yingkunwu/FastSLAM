import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))


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


def visualize(robot, particles, best_particle, radar_list, step, title, output_path, visualize=False):
    ax1.clear()
    ax2.clear()
    fig.suptitle("{}\n\n number of particles:{}, step:{}".format(title, len(particles), step + 1))
    ax1.set_title("Estimated by Particles")
    ax2.set_title("Ground Truth")
    ax1.axis("off")
    ax2.axis("off")

    grid_size = best_particle.grid_size
    ax1.set_xlim(0, grid_size[1])
    ax1.set_ylim(0, grid_size[0])

    grid_size = robot.grid_size
    ax2.set_xlim(0, grid_size[1])
    ax2.set_ylim(0, grid_size[0])

    # draw map
    world_map = 1 - best_particle.grid
    ax1.imshow(world_map, cmap='gray')
    world_map = 1 - robot.grid
    ax2.imshow(world_map, cmap='gray')

    # draw radar beams
    for (x, y) in radar_list:
        ax2.plot(x, y, "yo", markersize=1)

    # draw tragectory
    true_path = np.array(robot.trajectory)
    ax2.plot(true_path[:, 0], true_path[:, 1], "b")
    estimated_path = np.array(best_particle.trajectory)
    ax1.plot(estimated_path[:, 0], estimated_path[:, 1], "g")

    # draw particles position
    for p in particles:
        ax1.plot(p.x, p.y, "go", markersize=1)

    # draw robot position
    ax2.plot(robot.x, robot.y, "bo")

    if step % 10 == 0:
        plt.savefig('{}_{}.png'.format(output_path, step), bbox_inches='tight')

    if visualize:
        plt.draw()
        plt.pause(0.01)
