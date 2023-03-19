import numpy as np
import random

from world import World
from robot import Robot
from utils import *


if __name__ == "__main__":
    # create a world map
    world = World(100, 100)

    # create landmarks positions
    landmarks = [[20.0, 20.0], [20.0, 80.0], [20.0, 50.0], [50.0, 20.0], [50.0, 80.0], [80.0, 80.0], [80.0, 20.0], [80.0, 50.0]]

    for (x, y) in landmarks:
        world.set_landmarks(x, y)

    # create a robot
    R = Robot(world.size_x, world.size_y, world.landmarks)
    # set robot noise
    R.set_noise(0.2, 0.1, 3.0)
    # set robot position inside the world
    R.set_states(40.0, 40.0, np.pi / 2)

    # move the robot
    R.motion(turn=-np.pi / 2, forward=10.0)

    NUMBER_OF_PARTICLES = 1000
    p = []
    for i in range(NUMBER_OF_PARTICLES):
        r = Robot(world.size_x, world.size_y, world.landmarks)
        r.set_noise(0.2, 0.1, 3.0)
        p.append(r)

    NUMBER_OF_ITERATIONS = 1000
    for idx in range(1, NUMBER_OF_ITERATIONS):
        R.motion(turn=0.1, forward=0.5)
        z = R.sense()

        # Simulate a robot motion for each of these particles
        p_tmp = [None] * NUMBER_OF_PARTICLES
        for i in range(NUMBER_OF_PARTICLES):
            p_tmp[i] = p[i].motion(turn=0.1, forward=0.5)
            p[i] = p_tmp[i]

        if idx % 1 == 0:
            # Generate particle weights depending on robot's measurement
            w = [0.0] * NUMBER_OF_PARTICLES
            for i in range(NUMBER_OF_PARTICLES):
                w[i] = p[i].measurement_prob(z);
            
            # normalize
            w = w / sum(w)

            # Resample the particles with a sample probability proportional to the importance weight
            new_p = [None] * NUMBER_OF_PARTICLES
            J_inv = 1 / NUMBER_OF_PARTICLES
            r = random.random() * J_inv
            c = w[0]

            i = 0
            for j in range(NUMBER_OF_PARTICLES):
                U = r + j * J_inv
                while (U > c):
                    i += 1
                    c += w[i]
                new_p[j] = p[i]

            p = new_p

        visualize(R, p, world, idx)
