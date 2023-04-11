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
    (x, y, theta) = config['R_init']
    R = Robot(x, y, theta, config, world_grid, sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()

    # initialize particles
    p = [None] * NUMBER_OF_PARTICLES
    (x, y, theta) = config['p_init']
    for i in range(NUMBER_OF_PARTICLES):
        p[i] = Robot(x, y, theta, config)

    w = [1 / NUMBER_OF_PARTICLES] * NUMBER_OF_PARTICLES

    # monte carlo localization
    for idx, (forward, turn) in enumerate(config['controls']):
        R.move(turn=turn, forward=forward)
        curr_odo = R.get_state()
        R.update_trajectory()

        z_star, free_grid_star, occupy_grid_star = R.sense()
        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)

        for i in range(NUMBER_OF_PARTICLES):
            # Perform scan matching
            #_, free_grid, occupy_grid, scan = p[i].sense() #// For simplicity I use ground truth map for scan matching
            #if len(free_grid) > 0 and len(occupy_grid) > 0:
            #    free_grid, occupy_grid = p[i].absolute2relative(free_grid, occupy_grid)
            #    free_grid_prime, occupy_grid_prime = R.relative2absolute(free_grid, occupy_grid)

            #    pose = np.array(p[i].get_state())
            #    pose_hat = scan_matching(occupy_grid_prime, occupy_grid_star, pose)
                #plot_points(occupy_grid_offset_star.T, occupy_grid.T, fig)
            #else:
            #    pose_hat = None

            # If the scan matching fails, the pose and the weights are computed according to the motion model
            if True:
                # Sample around the mode
                #samples = np.random.multivariate_normal(pose_hat, mode_sample_cov, NUMBER_OF_MODE_SAMPLES)

                tmp_samples = [None] * NUMBER_OF_MODE_SAMPLES * 2
                z_list = np.zeros(NUMBER_OF_MODE_SAMPLES * 2)
                tmp_r = Robot(0, 0, 0, config, p[i].grid)
                for j in range(NUMBER_OF_MODE_SAMPLES * 2):
                    x, y, theta = p[i].sample_motion_model(prev_odo, curr_odo)
                    tmp_samples[j] = (x, y, theta)
                    tmp_r.set_states(x, y, theta)
                    z, _, _ = tmp_r.sense()
                    z_list[j] = p[i].measurement_model(z_star, z)

                tmp_idx = np.argsort(z_list)[:NUMBER_OF_MODE_SAMPLES]
                z_list = z_list[tmp_idx]
                samples = [None] * NUMBER_OF_MODE_SAMPLES
                for l, k in enumerate(tmp_idx):
                    samples[l] = tmp_samples[k]

                # Compute gaussain proposal
                likelihoods = z_list
                for j in range(NUMBER_OF_MODE_SAMPLES):
                    motion_prob = p[i].motion_model(prev_odo, curr_odo, samples[j])

                    #x, y, theta = samples[j]
                    #tmp_r = Robot(x, y, theta, world_grid.shape)
                    #z, _, _, _ = tmp_r.sense(p[i].grid)
                    #measurement_prob = p[i].measurement_model(z_star, z)

                    #likelihoods[j] = motion_prob * measurement_prob
                    likelihoods[j] = motion_prob * z_list[j]

                eta = np.sum(likelihoods)

                pose_mean = np.sum(samples * likelihoods[:, np.newaxis], axis=0)
                pose_mean = pose_mean / eta

                tmp = samples - pose_mean
                pose_cov = tmp.T @ (tmp * likelihoods[:, np.newaxis])
                pose_cov = pose_cov / eta

                # Sample new pose of the particle from the gaussian proposal
                new_pose = np.random.multivariate_normal(pose_mean, pose_cov, 1)
                x, y, theta = new_pose[0]
                p[i].set_states(x, y, theta)

                # Update weight
                w[i] *= eta
                
            else:
                # Simulate a robot motion for each of these particles
                p[i].motion_update(prev_odo, curr_odo)
        
                # Calculate particle's weights depending on robot's measurement
                z, _, _, _ = p[i].sense()
                w = p[i].measurement_model(z_star, z)

                # Update weight
                p[i].w = p[i].w * w

            p[i].update_trajectory()

            # Update occupancy grid based on the true measurements
            p_odo = p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, p_odo).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, p_odo).astype(np.int32)
            p[i].update_occupancy_grid(free_grid, occupy_grid)

        # normalize
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]

        # select best particle
        estimated_R = copy.deepcopy(p[best_id])

        # adaptive resampling
        N_eff = 1 / np.sum(w ** 2)
        if N_eff < NUMBER_OF_PARTICLES / 2:
            print("Resample!")
            # Resample the particles with a sample probability proportional to the importance weight
            # Use low variance sampling method
            new_p = [None] * NUMBER_OF_PARTICLES
            new_w = [None] * NUMBER_OF_PARTICLES
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
                new_w[j] = w[i]

            p = new_p
            w = new_w

        prev_odo = curr_odo

        visualize(R, p, estimated_R, free_grid_star, config, idx)