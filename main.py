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
    R.set_noise(sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()

    # initialize particles
    p = []
    for i in range(NUMBER_OF_PARTICLES):
        r = Robot(x, y, orientation, world_grid.shape)
        r.set_noise(sense_noise=0)
        p.append(r)

    # initial guess
    estimated_R = copy.deepcopy(p[0])

    # store path
    true_path, estimated_path = [], []

    # monte carlo localization
    for idx, (forward, turn) in enumerate(config['controls']):
        R.move(turn=turn, forward=forward)
        curr_odo = R.get_state()
        true_path.append([R.x, R.y])
        z_star, free_grid_star, occupy_grid_star = R.sense()
        free_grid_offset_star, occupy_grid_offset_star = R.absolute2relative(free_grid_star, occupy_grid_star)

        w = [0.0] * NUMBER_OF_PARTICLES
        for i in range(NUMBER_OF_PARTICLES):
            # Simulate a robot motion for each of these particles
            p[i].motion_update(prev_odo, curr_odo)

            z, free_grid, occupy_grid = p[i].sense(estimated_R.grid)
    
            # Calculate particle's weights depending on robot's measurement
            w[i] = p[i].measurement_model(z_star, z)

            # Update occupancy grid based on the true measurements
            p[i].update_occupancy_grid(free_grid_offset_star, occupy_grid_offset_star)
        
        # normalize
        w = w / sum(w)

        # select best particle
        best_id = np.argmax(w)
        estimated_R = copy.deepcopy(p[best_id])
        estimated_path.append([estimated_R.x, estimated_R.y])

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

        prev_odo = curr_odo

        visualize(R, p, estimated_R, world, free_grid_star, true_path, estimated_path, idx)
