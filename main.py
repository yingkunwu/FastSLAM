import numpy as np
import random
import copy

from world import World
from robot import Robot
from utils import *
from config import *


if __name__ == "__main__":
    config = SCENCES['scene-1']

    # create a world map
    world = World()
    world.read_map(config['map'])
    world_grid = world.get_grid()
    occupancy = world.get_occupancy()

    # create a robot
    (x, y, orientation) = config['init']
    R = Robot(x, y, orientation, world_grid.shape, world_grid)
    # set robot noise
    R.set_noise(0.5, 0.25, 3.0)

    # initialize particles
    p = []
    for i in range(NUMBER_OF_PARTICLES):
        location = random.choice(occupancy)
        r = Robot(x, y, orientation, world_grid.shape)
        r.set_noise(0.5, 0.25, 3.0)
        p.append(r)

    # store path
    true_path, estimated_path = [], []

    # monte carlo localization
    for idx, (forward, turn) in enumerate(config['paths']):
        R.motion(turn=turn, forward=forward)
        true_path.append([R.x, R.y])
        z, free_grid, occupy_grid, free_grid_offset, occupy_grid_offset = R.sense(world_grid)

        # Simulate a robot motion for each of these particles
        for i in range(NUMBER_OF_PARTICLES):
            p[i].motion(turn=turn, forward=forward, noise=True)

        if idx % 1 == 0:
            # Generate particle weights depending on robot's measurement
            w = [0.0] * NUMBER_OF_PARTICLES
            for i in range(NUMBER_OF_PARTICLES):
                w[i] = p[i].measurement_model(z, world_grid);
                p[i].update_occupancy_grid(free_grid_offset, occupy_grid_offset)
            
            # normalize
            w = w / sum(w)

            # select best particle
            best_id = np.argmax(w)
            best_particle = copy.deepcopy(p[best_id])
            estimated_path.append([best_particle.x, best_particle.y])

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

        visualize(R, p, best_particle, world, free_grid, true_path, estimated_path, idx)
