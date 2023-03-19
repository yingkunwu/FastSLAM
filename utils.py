import matplotlib.pyplot as plt


def visualize(robot, particles, world, step):
    plt.title("MCL, step " + str(step))
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    for p in particles:
        plt.plot(p.x, p.y, "go", markersize=1)

    for mark in world.landmarks:
        plt.plot(mark[0], mark[1], "ro")

    plt.plot(robot.x, robot.y, "bo")
    plt.pause(0.1)

    plt.clf()
    plt.draw()
