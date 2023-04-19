import numpy as np
import random
import copy
import os
import argparse
import yaml

from world import World
from robot import Robot
from motion_model import MotionModel
from measurement_model import MeasurementModel
from utils import absolute2relative, relative2absolute, degree2radian, visualize
from icp import icp_matching


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", type=str, default="scene-1", help="map that robot navigates in")
    parser.add_argument("-p", "--particles", type=int, default=100, help="number of particles")
    args = parser.parse_args()

    maps = ['scene-1', 'scene-2']
    assert args.map in maps, "Please specify one of the map in {}.".format(maps)
    assert 0 < args.particles < 200, "The number of particles should be larger than 0 and smaller than a reasonable value."

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ROBOT = config['robot']
    SCENE = config[args.map]
    NUMBER_OF_PARTICLES = args.particles
    NUMBER_OF_MODE_SAMPLES = config['num_mode_samples']
    MODE_SAMPLE_COV = np.diag(config['mode_sample_cov'])

    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, "fastslam2")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, args.map)

    # create a world map
    world = World()
    world.read_map(SCENE['map'])
    world_grid = world.get_grid()

    # create a robot
    (x, y, theta) = SCENE['R_init']
    R = Robot(x, y, degree2radian(theta), world_grid, ROBOT, sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()

    # initialize particles
    p = [None] * NUMBER_OF_PARTICLES
    (x, y, theta) = SCENE['p_init']
    init_grid = np.ones(SCENE['grid_size']) * ROBOT['prior_prob']
    for i in range(NUMBER_OF_PARTICLES):
        p[i] = Robot(x, y, degree2radian(theta), copy.deepcopy(init_grid), ROBOT)

    # create motion model
    motion_model = MotionModel(config['motion_model'])

    # create measurement model
    measurement_model = MeasurementModel(config['measurement_model'], ROBOT['radar_range'])

    # initialize particles' weight
    w = [1 / NUMBER_OF_PARTICLES] * NUMBER_OF_PARTICLES

    # FastSLAM2.0
    for idx, (forward, turn) in enumerate(SCENE['controls']):
        R.move(turn=degree2radian(turn), forward=forward)
        curr_odo = R.get_state()
        R.update_trajectory()

        z_star, free_grid_star, occupy_grid_star = R.sense()
        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)

        for i in range(NUMBER_OF_PARTICLES):
            prev_pose = p[i].get_state()
            tmp_r = Robot(0, 0, 0, p[i].grid, ROBOT)

            # generate initial guess from motion model
            guess_pose = motion_model.sample_motion_model(prev_odo, curr_odo, prev_pose)
            scan = relative2absolute(occupy_grid_offset_star, guess_pose).astype(np.int32)
            tmp = np.where(p[i].grid >= 0.9)
            edges = np.stack((tmp[1], tmp[0])).T
            # refine guess pose by scan matching
            pose_hat = icp_matching(edges, scan, guess_pose)

            # If the scan matching fails, the pose and the weights are computed according to the motion model
            if pose_hat is not None:
                # Sample around the mode
                samples = np.random.multivariate_normal(pose_hat, MODE_SAMPLE_COV, NUMBER_OF_MODE_SAMPLES)
                # Compute gaussain proposal
                likelihoods = np.zeros(NUMBER_OF_MODE_SAMPLES)
 
                for j in range(NUMBER_OF_MODE_SAMPLES):
                    motion_prob = motion_model.motion_model(prev_odo, curr_odo, prev_pose, samples[j])

                    x, y, theta = samples[j]
                    tmp_r.set_states(x, y, theta)
                    z, _, _ = tmp_r.sense()
                    measurement_prob = measurement_model.measurement_model(z_star, z)

                    likelihoods[j] = motion_prob * measurement_prob
                    
                eta = np.sum(likelihoods)
                if eta > 0:
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
                print("scan: ", w[i])
                
            else:
                # Simulate a robot motion for each of these particles
                x, y, theta = motion_model.sample_motion_model(prev_odo, curr_odo, prev_pose)
                p[i].set_states(x, y, theta)
        
                # Calculate particle's weights depending on robot's measurement
                z, _, _ = p[i].sense()
                w[i] *= measurement_model.measurement_model(z_star, z)
                print("not scan: ", w[i])

            p[i].update_trajectory()

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
            c = w[0]

            i = 0
            for j in range(NUMBER_OF_PARTICLES):
                U = r + j * J_inv
                while (U > c):
                    i += 1
                    c += w[i]
                new_p[j] = copy.deepcopy(p[i])
                new_w[j] = w[i]

            p = new_p
            w = new_w

        prev_odo = curr_odo

        visualize(R, p, estimated_R, free_grid_star, idx, "FastSLAM 2.0", output_path, False)
