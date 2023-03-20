import random
import numpy as np


class Robot(object):
    def __init__(self, x, y, orientation, landmarks, map_size):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.landmarks = landmarks
        self.map_size = map_size

        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

        self.num_sensors = 11
        self.radar_theta = (np.arange(0, self.num_sensors) - self.num_sensors // 2) * (np.pi / self.num_sensors)
        self.radar_length = 30

    def set_noise(self, forward_noise, turn_noise, sense_noise):
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise

    def motion(self, turn, forward, noise=False):
        self.orientation = self.orientation + turn
        if noise:
            self.orientation = self.orientation + random.gauss(0.0, self.turn_noise)
        self.orientation = self.orientation % (2 * np.pi)

        if noise:
            forward = forward + random.gauss(0.0, self.forward_noise)
        self.x = self.x + forward * np.cos(self.orientation)
        self.y = self.y + forward * np.sin(self.orientation)

    def sense(self, world_grid):
        measurements = self.ray_casting(world_grid)
        measurements = measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors)
        
        return measurements
    
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
        radar_dest[0, :] = np.clip(radar_rel_dest[0, :] + radar_src[0, :], 0, self.map_size[1] - 1)
        radar_dest[1, :] = np.clip(radar_rel_dest[1, :] + radar_src[1, :], 0, self.map_size[0] - 1)

        return radar_src, radar_dest
    
    def ray_casting(self, world_grid):
        radar_src, radar_dest = self.build_radar_beams()
        r = np.linspace(radar_src, radar_dest, self.radar_length)

        measurements = [self.radar_length] * self.num_sensors
        for dr in r:
            dist = np.sqrt(np.sum(np.power(dr - radar_src, 2), axis=0))

            for i, (x, y) in enumerate(dr.T):
                if not world_grid[int(y)][int(x)] and (dist[i] < measurements[i]):
                    measurements[i] = dist[i]

        return measurements

    def measurement_prob(self, measurement, world_grid):
        prob = 1.0

        #for i, (x, y) in enumerate(self.landmarks):
        #    dist = np.sqrt(np.power(x - self.x, 2) + np.power(y - self.y, 2))
        #    prob *= self.get_gaussian_probability(dist, self.sense_noise, measurement[i])

        dist = self.ray_casting(world_grid)
        for d, m in zip(dist, measurement):
            prob *= self.get_gaussian_probability(d, self.sense_noise, m)
        
        return prob
    
    @staticmethod
    def get_gaussian_probability(mean, var, z):
        return np.exp(-(np.power(mean - z, 2) / np.power(var, 2) / 2.0) / np.sqrt(2.0 * np.pi * np.power(var, 2)))
