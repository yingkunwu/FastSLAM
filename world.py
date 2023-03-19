import numpy as np
import cv2

class World(object):
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.landmarks = []
        self.map = None

    def set_landmarks(self, x, y):
        self.landmarks.append([x, y])

