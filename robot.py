import random
import numpy as np

from utils import *


random.seed(0)
np.random.rand(0)


class Robot(object):
    def __init__(self, x, y, orientation, grid_size, grid=None):
        # initialize robot pose
        self.x = x
        self.y = y
        self.orientation = orientation

        # probability for updating occupancy map
        self.prior_prob = 0.5
        self.occupy_prob = 0.9
        self.free_prob = 0.35

        # initialize map occupancy probability
        self.grid_size = grid_size
        self.grid = 1 - grid if grid is not None else np.ones(grid_size) * self.prior_prob

        # coefficient for measurement probability
        self.p_hit = 0.8
        self.p_short = 0.05
        self.p_max = 0.1
        self.p_rand = 0.05
        self.lambda_short = 0.15

        # motion noise for robot particals motion and sensing noise for robot measurement
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

        # parameters for beam range sensor
        self.num_sensors = 11
        self.radar_theta = (np.arange(0, self.num_sensors) - self.num_sensors // 2) * (np.pi / self.num_sensors)
        self.radar_length = 50
        self.radar_range = 60

    def set_noise(self, forward_noise, turn_noise, sense_noise):
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise

    def motion(self, turn, forward, noise=False):
        self.orientation = self.orientation + turn
        if noise:
            self.orientation = self.orientation + random.gauss(0.0, self.turn_noise)
        self.orientation = self.orientation - 2 * np.pi if self.orientation > np.pi else self.orientation
        self.orientation = self.orientation + 2 * np.pi if self.orientation <= -np.pi else self.orientation

        if noise:
            forward = forward + random.gauss(0.0, self.forward_noise)
        self.x = self.x + forward * np.cos(self.orientation)
        self.y = self.y + forward * np.sin(self.orientation)

    def sense(self, world_grid):
        measurements, free_grid, occupy_grid = self.ray_casting(world_grid)
        measurements = np.clip(measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors), 0.0, self.radar_range)

        # calculate map location relative to the robot pose
        pose = np.array([self.x, self.y])
        R, R_inv = create_rotation_matrix(self.orientation)
        free_grid_tmp = rotate(pose, np.array(free_grid), R_inv)
        occupy_grid_tmp = rotate(pose, np.array(occupy_grid), R_inv)
        free_grid_offset = free_grid_tmp - pose
        occupy_grid_offset = occupy_grid_tmp - pose
        
        return measurements, free_grid, occupy_grid, free_grid_offset, occupy_grid_offset
    
    def build_radar_beams(self):
        radar_src = np.array([[self.x] * self.num_sensors, [self.y] * self.num_sensors])
        radar_theta = self.radar_theta + self.orientation
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

        return radar_src.T, beams
    
    def ray_casting(self, world_grid=None):
        radar_src, beams = self.build_radar_beams()

        measurements = [self.radar_range] * self.num_sensors
        free_grid, occupy_grid = [], []
        for i, beam in enumerate(beams):
            dist = np.sqrt(np.sum(np.power(beam - radar_src[i], 2), axis=1))

            for b, d in zip(beam, dist):
                if self.grid[b[1]][b[0]] > 0.5 and d < measurements[i]: # TODO 這個world_grid不能是已知
                    measurements[i] = d

                if d < measurements[i]:
                    free_grid.append(b)
                else:
                    occupy_grid.append(b)
                    break  # no need to iterate the rest if we hit an object

        return measurements, free_grid, occupy_grid

    def measurement_model(self, z_star, world_grid=None):
        z, _, _ = self.ray_casting(world_grid)
        z, z_star = np.array(z), np.array(z_star)

        # probability of measuring correct range with local measurement noise
        prob_hit = np.exp(-(np.power(z - z_star, 2) / np.power(self.sense_noise, 2) / 2.0) / np.sqrt(2.0 * np.pi * np.power(self.sense_noise, 2)))

        # probability of hitting unexpected objects
        prob_short = self.lambda_short * np.exp(-self.lambda_short * z)
        prob_short[np.greater(z, z_star)] = 0

        # probability of not hitting anything or failures
        prob_max = np.ones_like(z)
        prob_max[z == self.radar_range] = 0

        # probability of random measurements
        prob_rand = 1 / self.radar_range

        # total probability (p_hit + p_shot + p_max + p_rand = 1)
        prob = self.p_hit * prob_hit + self.p_short * prob_short + self.p_max * prob_max + self.p_rand * prob_rand
        prob = np.prod(prob)

        return prob
    
    def update_occupancy_grid(self, free_grid_offset, occupy_grid_offset):
        # calculate map locaiton based on the current measurement and robot pose
        pose = np.array([self.x, self.y])
        free_grid_tmp = free_grid_offset + pose
        occupy_grid_tmp = occupy_grid_offset + pose
        R, R_inv = create_rotation_matrix(self.orientation)
        free_grid = rotate(pose, np.array(free_grid_tmp), R).astype(np.int32)
        occupy_grid = rotate(pose, np.array(occupy_grid_tmp), R).astype(np.int32)

        # update occupancy grid
        mask1 = np.logical_and(0 < free_grid[:, 0], free_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < free_grid[:, 1], free_grid[:, 1] < self.grid_size[0])
        free_grid = free_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(False)
        l = self.prob2logodds(self.grid[free_grid[:, 1], free_grid[:, 0]]) + self.prob2logodds(inverse_prob) - self.prob2logodds(self.prior_prob)
        self.grid[free_grid[:, 1], free_grid[:, 0]] = self.logodds2prob(l)

        mask1 = np.logical_and(0 < occupy_grid[:, 0], occupy_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < occupy_grid[:, 1], occupy_grid[:, 1] < self.grid_size[0])
        occupy_grid = occupy_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(True)
        l = self.prob2logodds(self.grid[occupy_grid[:, 1], occupy_grid[:, 0]]) + self.prob2logodds(inverse_prob) - self.prob2logodds(self.prior_prob)
        self.grid[occupy_grid[:, 1], occupy_grid[:, 0]] = self.logodds2prob(l)

    
    def inverse_sensing_model(self, occupy):
        if occupy:
            return self.occupy_prob
        else:
            return self.free_prob
    
    @staticmethod
    def prob2logodds(prob):
        return np.log(prob / (1 - prob + 1e-15))
    
    @staticmethod
    def logodds2prob(logodds):
        return 1 - 1 / (1 + np.exp(logodds) + 1e-15)
