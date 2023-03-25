import random
import numpy as np

from utils import *


class Robot(object):
    def __init__(self, x, y, orientation, grid_size, grid=None):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.grid_size = grid_size

        # coefficient for measurement probability
        self.p_hit = 0.8
        self.p_short = 0.05
        self.p_max = 0.1
        self.p_rand = 0.05
        self.lambda_short = 0.15

        # probability for updating occupancy map
        self.prior_prob = 0.5
        self.occupy_prob = 0.9
        self.free_prob = 0.35
        if grid is not None:
            self.grid = 1 - grid
        else:
            self.grid = np.ones(grid_size) * self.prior_prob # initialize map occupancy probability

        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

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
    
    def ray_casting(self, world_grid):
        radar_src, beams = self.build_radar_beams()

        measurements = [self.radar_range] * self.num_sensors

        free_grid, occupy_grid = [], []
        for i, beam in enumerate(beams):
            for b in beam:
                dist = np.sqrt(np.sum(np.power(b - radar_src[i], 2), axis=0))

                if self.grid[b[1]][b[0]] > 0.5 and dist < measurements[i]: # TODO 這個world_grid不能是已知
                    measurements[i] = dist

                if dist < measurements[i]:
                    free_grid.append(b)
                else:
                    occupy_grid.append(b)
                    break  # no need to iterate the rest if we hit an object

        return measurements, free_grid, occupy_grid

    def measurement_prob(self, measurement, world_grid):
        dist, _, _ = self.ray_casting(world_grid)
        prob = self.get_gaussian_probability(np.array(measurement), np.power(self.sense_noise, 2), np.array(dist))
        prob = np.prod(prob)

        return prob
    
    def update_occupancy_grid(self, free_grid_offset, occupy_grid_offset):
        # TODO 把旋轉角度也考慮進去
        pose = np.array([self.x, self.y])
        free_grid_tmp = free_grid_offset + pose
        occupy_grid_tmp = occupy_grid_offset + pose
        R, R_inv = create_rotation_matrix(self.orientation)
        free_grid = rotate(pose, np.array(free_grid_tmp), R).astype(np.int32)
        occupy_grid = rotate(pose, np.array(occupy_grid_tmp), R).astype(np.int32)

        for (x, y) in free_grid:
            if 0 < x < self.grid_size[1] and 0 < y < self.grid_size[0]:
                inverse_prob = self.inverse_sensing_model(False)
                l = self.prob2logodds(self.grid[y][x]) + self.prob2logodds(inverse_prob) - self.prob2logodds(self.prior_prob)
                self.grid[y][x] = self.logodds2prob(l)

        for (x, y) in occupy_grid:
            if 0 < x < self.grid_size[1] and 0 < y < self.grid_size[0]:
                inverse_prob = self.inverse_sensing_model(True)
                l = self.prob2logodds(self.grid[y][x]) + self.prob2logodds(inverse_prob) - self.prob2logodds(self.prior_prob)
                self.grid[y][x] = self.logodds2prob(l)

    
    def inverse_sensing_model(self, occupy):
        if occupy:
            return self.occupy_prob
        else:
            return self.free_prob

    def get_gaussian_probability(self, mean, var, z):
        # probability of measuring correct range with local measurement noise
        prob_hit = np.exp(-(np.power(z - mean, 2) / var / 2.0) / np.sqrt(2.0 * np.pi * var))

        # probability of hitting unexpected objects
        prob_short = self.lambda_short * np.exp(-self.lambda_short * z)
        prob_short[np.greater(prob_short, mean)] = 0

        # probability of not hitting anything or failures
        prob_max = np.ones_like(z)
        prob_max[z == self.radar_range] = 0

        # probability of random measurements
        prob_rand = 1 / self.radar_range

        # total probability (p_hit + p_shot + p_max + p_rand = 1)
        prob = self.p_hit * prob_hit + self.p_short * prob_short + self.p_max * prob_max + self.p_rand * prob_rand
        return prob
    
    @staticmethod
    def prob2logodds(prob):
        return np.log(prob / (1 - prob + 1e-15))
    
    @staticmethod
    def logodds2prob(logodds):
        return 1 - 1 / (1 + np.exp(logodds) + 1e-15)
