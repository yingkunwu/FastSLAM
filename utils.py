import numpy as np
import matplotlib.pyplot as plt


def visualize(robot, particles, world, radar_list, step):
    plt.title("MCL, step " + str(step + 1))
    plt.xlim(0, world.size_x)
    plt.ylim(0, world.size_y)

    # draw map
    world_map = world.get_map()
    plt.imshow(world_map)

    # draw radar beams
    for dr in radar_list:  
        for (x, y) in dr:
            plt.plot(x, y, "ko", markersize=1)

    # draw robot position
    plt.plot(robot.x, robot.y, "bo")

    # draw particles position
    for p in particles:
        plt.plot(p.x, p.y, "go", markersize=1)

    plt.pause(0.01)
    plt.draw()
    plt.clf()
