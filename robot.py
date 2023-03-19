import random
import numpy as np


class Robot(object):
    def __init__(self, x, y, orientation, landmarks):
        self.x = x
        self.y = y
        self.orientation = orientation

        self.landmarks = landmarks

        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set_noise(self, forward_noise, turn_noise, sense_noise):
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise

    def sense(self):
        measurements = []

        for (x, y) in self.landmarks:
            # get Euclidean distance to each landmark and add noise to simulate range finder data
            m = np.sqrt(np.power(x - self.x, 2) + np.power(y - self.y, 2)) + random.gauss(0.0, self.sense_noise)
            measurements.append(m)

        return measurements
    
    def motion(self, turn, forward, noise=False):
        self.orientation = self.orientation + turn
        if noise:
            self.orientation = self.orientation + random.gauss(0.0, self.turn_noise)
        self.orientation = self.orientation % (2 * np.pi)

        if noise:
            forward = forward + random.gauss(0.0, self.forward_noise)
        self.x = self.x + forward * np.cos(self.orientation)
        self.y = self.y + forward * np.sin(self.orientation)

    def measurement_prob(self, measurement):
        prob = 1.0

        for i, (x, y) in enumerate(self.landmarks):
            dist = np.sqrt(np.power(x - self.x, 2) + np.power(y - self.y, 2))
            prob *= self.get_gaussian_probability(dist, self.sense_noise, measurement[i])
        
        return prob
    
    @staticmethod
    def get_gaussian_probability(mean, var, z):
        return np.exp(-(np.power(mean - z, 2) / np.power(var, 2) / 2.0) / np.sqrt(2.0 * np.pi * np.power(var, 2)))
