import numpy as np
import matplotlib.pyplot as plt


def visualize(robot, particles, world, step):
    plt.title("MCL, step " + str(step + 1))
    plt.xlim(0, world.size_x)
    plt.ylim(0, world.size_y)

    world_map = world.get_map()
    plt.imshow(world_map)

    for p in particles:
        plt.plot(p.x, p.y, "go", markersize=1)

    plt.plot(robot.x, robot.y, "bo")

    # draw radar beams
    radar_src, radar_dest = robot.build_radar_beams()
    r = np.linspace(radar_src, radar_dest, robot.radar_length)
    for dr in r.T:  
        for (x, y) in dr.T:
            plt.plot(x, y, "ko", markersize=1)

    plt.pause(0.01)
    plt.draw()
    plt.clf()
