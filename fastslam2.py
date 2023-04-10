import numpy as np
import random
import copy

from world import World
from robot import Robot
from utils import *
from config import *

from icp import plot_points


if __name__ == "__main__":
    config = SCENCES['scene-1']

    # create a world map
    world = World()
    world.read_map(config['map'])
    world_grid = world.get_grid()
    occupancy = world.get_occupancy()

    # create a robot
    (x, y, theta) = config['init']
    R = Robot(x, y, theta, world_grid.shape, world_grid, sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()
    fig = plt.figure()

    offset = [0, 0]

    # initialize particles
    p = []
    for i in range(NUMBER_OF_PARTICLES):
        r = Robot(x, y, theta, world_grid.shape)
        r.w = 1 / NUMBER_OF_PARTICLES
        p.append(r)

    # initial estimate
    estimated_R = copy.deepcopy(p[0])
    prev_scan = curr_scan = []
    prev_idx = curr_idx = []

    # monte carlo localization
    for idx, (forward, turn) in enumerate(config['controls']):
        R.move(turn=turn, forward=forward)
        curr_odo = R.get_state()
        R.update_trajectory()

        z_star, free_grid_star, occupy_grid_star, scan_star = R.sense()
        curr_scan = free_grid_star
        curr_idx = scan_star

        free_grid_offset_star, occupy_grid_offset_star = R.absolute2relative(free_grid_star, occupy_grid_star)

        total_weight = 0
        best_id, best_w = 0, 0

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
                for j in range(NUMBER_OF_MODE_SAMPLES * 2):
                    x, y, theta = p[i].motion_sample(prev_odo, curr_odo)
                    tmp_samples[j] = (x, y, theta)
                    tmp_r = Robot(x, y, theta, world_grid.shape)
                    z, _, _, _ = tmp_r.sense(p[i].grid)
                    measurement_prob = p[i].measurement_model(z_star, z)
                    z_list[j] = measurement_prob

                idx = np.argsort(z_list)[:NUMBER_OF_MODE_SAMPLES]
                z_list = z_list[idx]
                samples = [None] * NUMBER_OF_MODE_SAMPLES
                for l, k in enumerate(idx):
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
                p[i].w = p[i].w * eta
                
            else:
                # Simulate a robot motion for each of these particles
                p[i].motion_update(prev_odo, curr_odo)
        
                # Calculate particle's weights depending on robot's measurement
                z, _, _, _ = p[i].sense()
                w = p[i].measurement_model(z_star, z)

                # Update weight
                p[i].w = p[i].w * w

            p[i].update_trajectory()

            total_weight += p[i].w
            if p[i].w > best_w:
                best_w = p[i].w
                best_id = i

            # Update occupancy grid based on the true measurements
            free_grid, occupy_grid = p[i].relative2absolute(free_grid_offset_star, occupy_grid_offset_star)
            p[i].update_occupancy_grid(free_grid, occupy_grid)

        # normalize
        N_eff = 0
        for i in range(NUMBER_OF_PARTICLES):
            p[i].w /= total_weight
            N_eff += p[i].w ** 2
        N_eff = 1 / N_eff

        # select best particle
        estimated_R = copy.deepcopy(p[best_id])

        # adaptive resampling
        if N_eff < NUMBER_OF_PARTICLES / 2:
            print("Resample!")
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
        prev_idx = curr_idx

        visualize(R, p, estimated_R, world, free_grid_star, idx, offset)
