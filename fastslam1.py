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

    # create a robot
    (x, y, theta) = config['init']
    R = Robot(x, y, theta, world_grid.shape, world_grid, sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()

    offset = [config['grid_size'][0] / 2 - x, config['grid_size'][1] / 2 - y]

    # initialize particles
    p = []
    for i in range(NUMBER_OF_PARTICLES):
        r = Robot(config['grid_size'][0] / 2, config['grid_size'][1] / 2, theta, config['grid_size'])
        r.w = 1 / NUMBER_OF_PARTICLES
        p.append(r)

    # initial estimate
    estimated_R = copy.deepcopy(p[0])

    # monte carlo localization
    for idx, (forward, turn) in enumerate(config['controls']):
        R.move(turn=turn, forward=forward)
        curr_odo = R.get_state()
        R.update_trajectory()

        z_star, free_grid_star, occupy_grid_star, scan_star = R.sense()
        free_grid_offset_star, occupy_grid_offset_star = R.absolute2relative(free_grid_star, occupy_grid_star)

        total_weight = 0
        best_id, best_w = 0, 0
        for i in range(NUMBER_OF_PARTICLES):
            # Perform scan matching
            # _, free_grid, occupy_grid, scan = p[i].sense() // For simplicity I use ground truth map for scan matching

            free_grid_prime, occupy_grid_prime = p[i].relative2absolute(free_grid_offset_star, occupy_grid_offset_star)

            # Simulate a robot motion for each of these particles
            p[i].motion_update(prev_odo, curr_odo)
            p[i].update_trajectory()
    
            # Calculate particle's weights depending on robot's measurement
            z, _, _, _ = p[i].sense()
            w = p[i].measurement_model(z_star, z)

            # Update weight
            p[i].w = w

            total_weight += p[i].w
            if p[i].w > best_w:
                best_w = p[i].w
                best_id = i

            # Update occupancy grid based on the true measurements
            free_grid, occupy_grid = p[i].relative2absolute(free_grid_offset_star, occupy_grid_offset_star)
            p[i].update_occupancy_grid(free_grid, occupy_grid)

        # normalize
        for i in range(NUMBER_OF_PARTICLES):
            p[i].w /= total_weight

        # select best particle
        estimated_R = copy.deepcopy(p[best_id])

        # Resample the particles with a sample probability proportional to the importance weight
        # Use low variance sampling method
        new_p = [None] * NUMBER_OF_PARTICLES
        J_inv = 1 / NUMBER_OF_PARTICLES
        r = random.random() * J_inv
        c = p[0].w

        i = 0
        for j in range(NUMBER_OF_PARTICLES):
            U = r + j * J_inv
            while (U > c):
                i += 1
                c += p[i].w
            new_p[j] = copy.deepcopy(p[i])

        p = new_p
        prev_odo = curr_odo

        visualize(R, p, estimated_R, world, free_grid_star, idx, offset)
