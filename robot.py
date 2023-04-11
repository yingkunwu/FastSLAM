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
