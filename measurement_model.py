import numpy as np

from utils import normalDistribution

class MeasurementModel(object):
    def __init__(self, config, radar_range):
        self.p_hit = config['p_hit']
        self.sigma_hit = config['sigma_hit']
        self.p_short = config['p_short']
        self.p_max = config['p_max']
        self.p_rand = config['p_rand']
        self.lambda_short = config['lambda_short']
        self.radar_range = radar_range

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
