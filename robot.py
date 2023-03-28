import random
import numpy as np

from utils import *


random.seed(12)
np.random.rand(12)


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

        # probability weight
        self.w = 0

        # coefficient for motion model
        self.alpha1 = 0.005
        self.alpha2 = 0.005
        self.alpha3 = 0.01
        self.alpha4 = 0.01

        # coefficient for measurement probability
        self.p_hit = 0.8
        self.sigma_hit = 20
        self.p_short = 0.05
        self.p_max = 0.1
        self.p_rand = 0.05
        self.lambda_short = 0.15

        # motion noise for robot particals motion and sensing noise for robot measurement
        self.sense_noise = 0.0

        # parameters for beam range sensor
        self.num_sensors = 11
        self.radar_theta = (np.arange(0, self.num_sensors) - self.num_sensors // 2) * (np.pi / self.num_sensors)
        self.radar_length = 50
        self.radar_range = 60

    def set_states(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

    def set_noise(self, sense_noise):
        self.sense_noise = sense_noise

    def get_state(self):
        return (self.x, self.y, self.orientation)

    def move(self, turn, forward):
        self.orientation = self.orientation + turn
        self.orientation = wrapAngle(self.orientation)

        self.x = self.x + forward * np.cos(self.orientation)
        self.y = self.y + forward * np.sin(self.orientation)

    def motion_update(self, prev_odo, curr_odo):
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

        self.x = self.x + trans * np.cos(self.orientation + rot1)
        self.y = self.y + trans * np.sin(self.orientation + rot1)
        self.orientation = self.orientation + rot1 + rot2

    def sense(self, world_grid=None):
        measurements, free_grid, occupy_grid, scan = self.ray_casting(world_grid)
        measurements = np.clip(measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors), 0.0, self.radar_range)
        
        return measurements, free_grid, occupy_grid, scan
    
    def absolute2relative(self, free_grid, occupy_grid):
        # calculate map location relative to the robot pose
        pose = np.array([self.x, self.y])
        R, R_inv = create_rotation_matrix(self.orientation)
        free_grid_tmp = rotate(pose, np.array(free_grid), R_inv)
        occupy_grid_tmp = rotate(pose, np.array(occupy_grid), R_inv)
        free_grid_offset = free_grid_tmp - pose
        occupy_grid_offset = occupy_grid_tmp - pose

        return free_grid_offset, occupy_grid_offset
    
    def relative2absolute(self, free_grid_offset, occupy_grid_offset):
        # calculate map locaiton based on the current measurement and robot pose
        pose = np.array([self.x, self.y])
        free_grid_tmp = free_grid_offset + pose
        occupy_grid_tmp = occupy_grid_offset + pose
        R, R_inv = create_rotation_matrix(self.orientation)
        free_grid = rotate(pose, np.array(free_grid_tmp), R).astype(np.int32)
        occupy_grid = rotate(pose, np.array(occupy_grid_tmp), R).astype(np.int32)

        return free_grid, occupy_grid

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
        scan = [None] * self.num_sensors
        for i, beam in enumerate(beams):
            dist = np.sqrt(np.sum(np.power(beam - radar_src[i], 2), axis=1))

            for b, d in zip(beam, dist):
                if world_grid is not None:
                    if world_grid[b[1]][b[0]] > 0.5 and d < measurements[i]:
                        measurements[i] = d
                else:
                    if self.grid[b[1]][b[0]] > 0.5 and d < measurements[i]:
                        measurements[i] = d

                if d < measurements[i]:
                    free_grid.append(b)
                else:
                    occupy_grid.append(b)
                    scan[i] = b
                    break  # no need to iterate the rest if we hit an object

        return measurements, free_grid, occupy_grid, scan

    def measurement_model(self, z_star, z):
        z_star, z = np.array(z_star), np.array(z)

        # probability of measuring correct range with local measurement noise
        prob_hit = normalDistribution(z - z_star, np.power(self.sigma_hit, 2))

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
    
    def motion_model(self, prev_odo, curr_odo, curr_pose):
        rot1 = np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2]
        rot1 = wrapAngle(rot1)
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        rot2 = curr_odo[2] - prev_odo[2] - rot1
        rot2 = wrapAngle(rot2)

        rot1_prime = np.arctan2(curr_pose[1] - self.y, curr_pose[0] - self.x) - self.orientation
        rot1_prime = wrapAngle(rot1_prime)
        trans_prime = np.sqrt((curr_pose[0] - self.x) ** 2 + (curr_pose[1] - self.y) ** 2)
        rot2_prime = curr_pose[2] - self.orientation - rot1_prime
        rot2_prime = wrapAngle(rot2_prime)

        p1 = normalDistribution(wrapAngle(rot1 - rot1_prime), self.alpha1 * rot1_prime ** 2 + self.alpha2 * trans_prime ** 2)
        p2 = normalDistribution(trans - trans_prime, self.alpha3 * trans_prime ** 2 + self.alpha4 * (rot1_prime ** 2 + rot2_prime ** 2))
        p3 = normalDistribution(wrapAngle(rot2 - rot2_prime), self.alpha1 * rot2_prime ** 2 + self.alpha2 * trans_prime ** 2)

        return p1 * p2 * p3
    
    def update_occupancy_grid(self, free_grid_offset, occupy_grid_offset):
        free_grid, occupy_grid = self.relative2absolute(free_grid_offset, occupy_grid_offset)

        # update occupancy grid
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
