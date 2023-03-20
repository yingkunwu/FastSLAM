import numpy as np
import random
import copy

from world import World
from robot import Robot
from utils import *
from config import SCENCES


if __name__ == "__main__":
    config = SCENCES['scene-1']

    # create a world map
    world = World()
    world.read_map(config['map'])
    occupancy = world.get_occupancy()

    world_grid = world.get_grid()
    map_size = world_grid.shape

    # create landmarks positions
    landmarks = [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [0.0, 50.0], [200.0, 50.0], [0.0, 100.0], [100.0, 100.0], [200.0, 100.0]]

    for (x, y) in landmarks:
        world.set_landmarks(x, y)

    # create a robot
    (x, y, orientation) = config['init']
    R = Robot(x, y, orientation, world.landmarks, map_size)
    # set robot noise
    R.set_noise(0.2, 0.1, 3.0)

    # initialize particles
    p = []
    NUMBER_OF_PARTICLES = 1000
    for i in range(NUMBER_OF_PARTICLES):
        location = random.choice(occupancy)
        r = Robot(location[0], location[1], random.random() * 2 * np.pi, world.landmarks, map_size)
        r.set_noise(0.2, 0.1, 3.0)
        p.append(r)

    # monte carlo localization
    for idx, (forward, turn) in enumerate(config['paths']):
        R.motion(turn=turn, forward=forward)
        z = R.sense(world_grid)

        # Simulate a robot motion for each of these particles
        for i in range(NUMBER_OF_PARTICLES):
            p[i].motion(turn=turn, forward=forward, noise=True)

        if idx % 1 == 0:
            # Generate particle weights depending on robot's measurement
            w = [0.0] * NUMBER_OF_PARTICLES
            for i in range(NUMBER_OF_PARTICLES):
                w[i] = p[i].measurement_prob(z, world_grid);
            
            # normalize
            w = w / sum(w)

            # Resample the particles with a sample probability proportional to the importance weight
            # Use low variance sampling method
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
                new_p[j] = copy.deepcopy(p[i])

            p = new_p

        visualize(R, p, world, idx)
