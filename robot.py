import random
import numpy as np


class Robot(object):
    def __init__(self, x, y, landmarks):
        self.x = random.random() * x
        self.y = random.random() * y
        self.orientation = random.random() * 2 * np.pi

        self.landmarks = landmarks

        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set_states(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

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
    
    def motion(self, turn, forward):
        self.orientation = (self.orientation + turn + random.gauss(0.0, self.turn_noise)) % (2 * np.pi)

        dist = forward + random.gauss(0.0, self.forward_noise)
        self.x = self.x + dist * np.cos(self.orientation)
        self.y = self.y + dist * np.sin(self.orientation)

        # 不知道為啥一定要重新initialize才會work
        r = Robot(self.x, self.y, self.landmarks)
        r.set_noise(0.2, 0.1, 3.0)
        r.set_states(self.x, self.y, self.orientation)
        return r

    def measurement_prob(self, measurement):
        prob = 1.0

        for i, (x, y) in enumerate(self.landmarks):
            dist = np.sqrt(np.power(x - self.x, 2) + np.power(y - self.y, 2))
            prob *= self.get_gaussian_probability(dist, self.sense_noise, measurement[i])
        
        return prob
    
    @staticmethod
    def get_gaussian_probability(mean, var, z):
        return np.exp(-(np.power(mean - z, 2) / np.power(var, 2) / 2.0) / np.sqrt(2.0 * np.pi * np.power(var, 2)))
