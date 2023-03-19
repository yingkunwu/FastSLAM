import matplotlib.pyplot as plt


def visualize(robot, particles, world, step):
    plt.title("MCL, step " + str(step + 1))
    plt.xlim(0, world.size_x)
    plt.ylim(0, world.size_y)

    world_map = world.get_map()
    plt.imshow(world_map)

    for p in particles:
        plt.plot(p.x, p.y, "go", markersize=1)

    for mark in world.landmarks:
        plt.plot(mark[0], mark[1], "ro")

    plt.plot(robot.x, robot.y, "bo")
    plt.pause(0.01)

    plt.draw()
    plt.clf()
