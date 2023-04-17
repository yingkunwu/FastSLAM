import numpy as np
import random
import copy
import os

from world import World
from robot import Robot
from motion_model import MotionModel
from measurement_model import MeasurementModel
from utils import absolute2relative, relative2absolute, visualize
from config import *


if __name__ == "__main__":
    name = 'scene-2'
    config = SCENCES[name]

    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, "fastslam1")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, name)

    # create a world map
    world = World()
    world.read_map(config['map'])
    world_grid = world.get_grid()

    # create a robot
    (x, y, theta) = config['R_init']
    R = Robot(x, y, theta, config, world_grid, sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()

    # initialize particles
    p = [None] * NUMBER_OF_PARTICLES
    (x, y, theta) = config['p_init']
    for i in range(NUMBER_OF_PARTICLES):
        p[i] = Robot(x, y, theta, config)

    # create motion model
    motion_model = MotionModel(config['motion_model'])

    # create measurement model
    measurement_model = MeasurementModel(config['measurement_model'], config['radar_range'])

    # FastSLAM1.0
    for idx, (forward, turn) in enumerate(config['controls']):
        R.move(turn=turn, forward=forward)
        curr_odo = R.get_state()
        R.update_trajectory()

        z_star, free_grid_star, occupy_grid_star = R.sense()
        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)

        w = np.zeros(NUMBER_OF_PARTICLES)
        for i in range(NUMBER_OF_PARTICLES):
            # Simulate a robot motion for each of these particles
            prev_pose = p[i].get_state()
            x, y, theta = motion_model.sample_motion_model(prev_odo, curr_odo, prev_pose)
            p[i].set_states(x, y, theta)
            p[i].update_trajectory()
    
            # Calculate particle's weights depending on robot's measurement
            z, _, _ = p[i].sense()
            w[i] = measurement_model.measurement_model(z_star, z)

            # Update occupancy grid based on the true measurements
            curr_pose = p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            p[i].update_occupancy_grid(free_grid, occupy_grid)

        # normalize
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]

        # select best particle
        estimated_R = copy.deepcopy(p[best_id])

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

        visualize(R, p, estimated_R, free_grid_star, idx, "FastSLAM 1.0", output_path, False)
