import random
import numpy as np

from utils import *


random.seed(10)
np.random.rand(10)


class Robot(object):
    def __init__(self, x, y, theta, config, grid=None, sense_noise=None):
        # initialize robot pose
        self.x = x
        self.y = y
        self.theta = theta
        self.trajectory = []

        # probability for updating occupancy map
        self.prior_prob = config['prior_prob']
        self.occupy_prob = config['occupy_prob']
        self.free_prob = config['free_prob']

        # initialize map occupancy probability
        if grid is not None:
            self.grid = 1 - grid
            self.grid_size = self.grid.shape
        else:
            self.grid_size = config['grid_size']
            self.grid = np.ones(self.grid_size) * self.prior_prob

        # sensing noise for trun robot measurement
        self.sense_noise = sense_noise if sense_noise is not None else 0.0

        # coefficient for motion model
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.alpha3 = config['alpha3']
        self.alpha4 = config['alpha4']

        # coefficient for measurement probability
        self.p_hit = config['p_hit']
        self.sigma_hit = config['sigma_hit']
        self.p_short = config['p_short']
        self.p_max = config['p_max']
        self.p_rand = config['p_rand']
        self.lambda_short = config['lambda_short']

        # parameters for beam range sensor
        self.num_sensors = config['num_sensors']
        self.radar_theta = (np.arange(0, self.num_sensors)) * (2 * np.pi / self.num_sensors)
        self.radar_length = config['radar_length']
        self.radar_range = config['radar_range']

    def set_states(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def get_state(self):
        return (self.x, self.y, self.theta)
    
    def update_trajectory(self):
        self.trajectory.append([self.x, self.y])

    def move(self, turn, forward):
        self.theta = self.theta + turn
        self.theta = wrapAngle(self.theta)

        self.x = self.x + forward * np.cos(self.theta)
        self.y = self.y + forward * np.sin(self.theta)

    def sample_motion_model(self, prev_odo, curr_odo):
        rot1 = np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2]
        rot1 = wrapAngle(rot1)
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        rot2 = curr_odo[2] - prev_odo[2] - rot1
        rot2 = wrapAngle(rot2)

        rot1 = rot1 - np.random.normal(0, self.alpha1 * rot1 ** 2 + self.alpha2 * trans ** 2)
        rot1 = wrapAngle(rot1)
        trans = trans - np.random.normal(0, self.alpha3 * trans ** 2 + self.alpha4 * (rot1 ** 2 + rot2 ** 2))
        rot2 = rot2 - np.random.normal(0, self.alpha1 * rot2 ** 2 + self.alpha2 * trans ** 2)
        rot2 = wrapAngle(rot2)

        x = self.x + trans * np.cos(self.theta + rot1)
        y = self.y + trans * np.sin(self.theta + rot1)
        theta = self.theta + rot1 + rot2

        return (x, y, theta)

    def sense(self, world_grid=None):
        measurements, free_grid, occupy_grid = self.ray_casting(world_grid)
        measurements = np.clip(measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors), 0.0, self.radar_range)
        
        return measurements, free_grid, occupy_grid

    def build_radar_beams(self):
        radar_src = np.array([[self.x] * self.num_sensors, [self.y] * self.num_sensors])
        radar_theta = self.radar_theta + self.theta
        radar_rel_dest = np.stack(
            (
                np.cos(radar_theta) * self.radar_length,
                np.sin(radar_theta) * self.radar_length
            ), axis=0
        )

        radar_dest = np.zeros_like(radar_rel_dest)
        radar_dest[0, :] = np.clip(radar_rel_dest[0, :] + radar_src[0, :], 0, self.grid_size[1] - 1)
        radar_dest[1, :] = np.clip(radar_rel_dest[1, :] + radar_src[1, :], 0, self.grid_size[0] - 1)

        beams = [None] * self.num_sensors
        for i in range(self.num_sensors):
            x1, y1 = radar_src[:, i]
            x2, y2 = radar_dest[:, i]
            beams[i] = bresenham(x1, y1, x2, y2)

        return beams
    
    def ray_casting(self, world_grid=None):
        beams = self.build_radar_beams()

        loc = np.array([self.x, self.y])
        measurements = [self.radar_range] * self.num_sensors
        free_grid, occupy_grid = [], []

        for i, beam in enumerate(beams):
            dist = np.linalg.norm(beam - loc, axis=1)
            beam = np.array(beam)

            obstacle_position = np.where(self.grid[beam[:, 1], beam[:, 0]] > 0.5)[0]
            if len(obstacle_position) > 0:
                idx = obstacle_position[0]
                occupy_grid.append(list(beam[idx]))
                free_grid.extend(list(beam[:idx]))
                measurements[i] = dist[idx]
            else:
                free_grid.extend(list(beam))

        return measurements, free_grid, occupy_grid
    
    def motion_model(self, prev_odo, curr_odo, curr_pose):
        rot1 = np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2]
        rot1 = wrapAngle(rot1)
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        rot2 = curr_odo[2] - prev_odo[2] - rot1
        rot2 = wrapAngle(rot2)

        rot1_prime = np.arctan2(curr_pose[1] - self.y, curr_pose[0] - self.x) - self.theta
        rot1_prime = wrapAngle(rot1_prime)
        trans_prime = np.sqrt((curr_pose[0] - self.x) ** 2 + (curr_pose[1] - self.y) ** 2)
        rot2_prime = curr_pose[2] - self.theta - rot1_prime
        rot2_prime = wrapAngle(rot2_prime)
        
        p1 = normalDistribution(wrapAngle(rot1 - rot1_prime), self.alpha1 * rot1_prime ** 2 + self.alpha2 * trans_prime ** 2)
        p2 = normalDistribution(trans - trans_prime, self.alpha3 * trans_prime ** 2 + self.alpha4 * (rot1_prime ** 2 + rot2_prime ** 2))
        p3 = normalDistribution(wrapAngle(rot2 - rot2_prime), self.alpha1 * rot2_prime ** 2 + self.alpha2 * trans_prime ** 2)

        return p1 * p2 * p3

    def measurement_model(self, z_star, z):
        z_star, z = np.array(z_star), np.array(z)

        # probability of measuring correct range with local measurement noise
        prob_hit = normalDistribution(z - z_star, np.power(self.sigma_hit, 2))

        # probability of hitting unexpected objects
        prob_short = self.lambda_short * np.exp(-self.lambda_short * z)
        prob_short[np.greater(z, z_star)] = 0

        # probability of not hitting anything or failures
        prob_max = np.zeros_like(z)
        prob_max[z == self.radar_range] = 1

        # probability of random measurements
        prob_rand = 1 / self.radar_range

        # total probability (p_hit + p_shot + p_max + p_rand = 1)
        prob = self.p_hit * prob_hit + self.p_short * prob_short + self.p_max * prob_max + self.p_rand * prob_rand
        prob = np.prod(prob)

        return prob
    
    def update_occupancy_grid(self, free_grid, occupy_grid):
        mask1 = np.logical_and(0 < free_grid[:, 0], free_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < free_grid[:, 1], free_grid[:, 1] < self.grid_size[0])
        free_grid = free_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(False)
        l = prob2logodds(self.grid[free_grid[:, 1], free_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[free_grid[:, 1], free_grid[:, 0]] = logodds2prob(l)

        mask1 = np.logical_and(0 < occupy_grid[:, 0], occupy_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < occupy_grid[:, 1], occupy_grid[:, 1] < self.grid_size[0])
        occupy_grid = occupy_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(True)
        l = prob2logodds(self.grid[occupy_grid[:, 1], occupy_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[occupy_grid[:, 1], occupy_grid[:, 0]] = logodds2prob(l)
    
    def inverse_sensing_model(self, occupy):
        if occupy:
            return self.occupy_prob
        else:
            return self.free_prob
