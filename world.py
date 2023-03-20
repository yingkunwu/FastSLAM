import numpy as np
import cv2

class World(object):
    def __init__(self, size_x=0, size_y=0):
        self.size_x = size_x
        self.size_y = size_y
        self.landmarks = []
        self.map = None
        self.grid = None
        self.occupancy = []

    def set_landmarks(self, x, y):
        self.landmarks.append([x, y])

    def read_map(self, filename):
        self.map = cv2.imread(filename)
        self.map = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)
        height, width, _ = self.map.shape
        self.size_y, self.size_x = height, width

        self.map = np.flip(self.map, 0)

        self.grid = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                if (self.map[i][j] != [255, 255, 255]).any():
                    self.grid[i][j] = 0
                else:
                    self.occupancy.append([j, i])

    def get_map(self):
        return self.map
    
    def get_grid(self):
        return self.grid
    
    def get_occupancy(self):
        return self.occupancy