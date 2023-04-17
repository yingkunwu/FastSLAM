import numpy as np
import cv2

class World(object):
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.grid = None
        self.edges = None

    def read_map(self, filename):
        img = cv2.imread(filename)
        img = np.flip(img, 0)
        color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.height, self.width = color.shape[:2]

        self.grid = np.zeros((self.height, self.width))
        self.grid[(color != (255, 255, 255)).any(axis=2)] = 1

    def get_grid(self):
        return self.grid
