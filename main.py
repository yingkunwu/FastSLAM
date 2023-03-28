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
        r.w = 1 / NUMBER_OF_PARTICLES
        p.append(r)

    # initial estimate
    estimated_R = copy.deepcopy(p[0])
    _, _, _, scan = estimated_R.sense()
    prev_scan = curr_scan = scan

    # store path
    true_path, estimated_path = [], []

    # monte carlo localization
    for idx, (forward, turn) in enumerate(config['controls']):
        R.move(turn=turn, forward=forward)
        curr_odo = R.get_state()
        true_path.append([R.x, R.y])
        z_star, free_grid_star, occupy_grid_star, curr_scan = R.sense()
        free_grid_offset_star, occupy_grid_offset_star = R.absolute2relative(free_grid_star, occupy_grid_star)

        total_weight = 0
        best_id, best_w = 0, 0
        for i in range(NUMBER_OF_PARTICLES):
            # Perform scan matching
            _, _, _, prev_scan = p[i].sense(estimated_R.grid)
            pose = np.array(p[i].get_state())
            pose_hat = icp(prev_scan, curr_scan, pose)

            # If the scan matching fails, the pose and the weights are computed according to the motion model
            if pose_hat is not None:
                # Sample around the mode
                samples = np.random.multivariate_normal(pose_hat, mode_sample_cov, NUMBER_OF_MODE_SAMPLES)

                # Compute gaussain proposal
                likelihoods = np.zeros(NUMBER_OF_MODE_SAMPLES)
                for j in range(NUMBER_OF_MODE_SAMPLES):
                    motion_prob = p[i].motion_model(prev_odo, curr_odo, samples[j])

                    x, y, orientation = samples[j]
                    tmp_r = Robot(x, y, orientation, world_grid.shape)
                    z, _, _, _ = tmp_r.sense(p[i].grid)
                    measurement_prob = p[i].measurement_model(z_star, z)

                    prob = motion_prob * measurement_prob
                    likelihoods[j] = prob

                eta = np.sum(likelihoods)

                pose_mean = np.sum(samples * likelihoods[:, np.newaxis], axis=0)
                pose_mean = pose_mean / eta

                tmp = samples - pose_mean
                pose_cov = tmp.T @ (tmp * likelihoods[:, np.newaxis])
                pose_cov = pose_cov / eta

                # Sample new pose of the particle from the gaussian proposal
                new_pose = np.random.multivariate_normal(pose_mean, pose_cov, 1)
                x, y, orientation = new_pose[0]
                p[i].set_states(x, y, orientation)

                # Update weight
                p[i].w = p[i].w * eta
                
            else:
                # Simulate a robot motion for each of these particles
                p[i].motion_update(prev_odo, curr_odo)
        
                # Calculate particle's weights depending on robot's measurement
                z, _, _, _ = p[i].sense(estimated_R.grid)
                w = p[i].measurement_model(z_star, z)

                # Update weight
                p[i].w = p[i].w * w

            total_weight += p[i].w
            if p[i].w > best_w:
                best_w = p[i].w
                best_id = i

            # Update occupancy grid based on the true measurements
            p[i].update_occupancy_grid(free_grid_offset_star, occupy_grid_offset_star)

        # normalize
        for i in range(NUMBER_OF_PARTICLES):
            p[i].w /= total_weight

        # select best particle
        estimated_R = copy.deepcopy(p[best_id])
        estimated_path.append([estimated_R.x, estimated_R.y])

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
        prev_scan = curr_scan

        visualize(R, p, estimated_R, world, free_grid_star, true_path, estimated_path, idx)
