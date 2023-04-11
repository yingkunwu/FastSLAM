import numpy as np

from utils import wrapAngle, normalDistribution

class MotionModel(object):
    def __init__(self, config):
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.alpha3 = config['alpha3']
        self.alpha4 = config['alpha4']

    def sample_motion_model(self, prev_odo, curr_odo, prev_pose):
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

        x = prev_pose[0] + trans * np.cos(prev_pose[2] + rot1)
        y = prev_pose[1] + trans * np.sin(prev_pose[2] + rot1)
        theta = prev_pose[2] + rot1 + rot2

        return (x, y, theta)
    
    def motion_model(self, prev_odo, curr_odo, prev_pose, curr_pose):
        rot1 = np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2]
        rot1 = wrapAngle(rot1)
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        rot2 = curr_odo[2] - prev_odo[2] - rot1
        rot2 = wrapAngle(rot2)

        rot1_prime = np.arctan2(curr_pose[1] - prev_pose[1], curr_pose[0] - prev_pose[0]) - prev_pose[2]
        rot1_prime = wrapAngle(rot1_prime)
        trans_prime = np.sqrt((curr_pose[0] - prev_pose[0]) ** 2 + (curr_pose[1] - prev_pose[1]) ** 2)
        rot2_prime = curr_pose[2] - prev_pose[2] - rot1_prime
        rot2_prime = wrapAngle(rot2_prime)
        
        p1 = normalDistribution(wrapAngle(rot1 - rot1_prime), self.alpha1 * rot1_prime ** 2 + self.alpha2 * trans_prime ** 2)
        p2 = normalDistribution(trans - trans_prime, self.alpha3 * trans_prime ** 2 + self.alpha4 * (rot1_prime ** 2 + rot2_prime ** 2))
        p3 = normalDistribution(wrapAngle(rot2 - rot2_prime), self.alpha1 * rot2_prime ** 2 + self.alpha2 * trans_prime ** 2)

        return p1 * p2 * p3