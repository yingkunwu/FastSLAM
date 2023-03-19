import numpy as np
import cv2

class World(object):
    def __init__(self, size_x=0, size_y=0):
        self.size_x = size_x
        self.size_y = size_y
        self.landmarks = []
        self.map = None
        self.occupancy = []

    def set_landmarks(self, x, y):
        self.landmarks.append([x, y])

    def read_map(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        self.size_y, self.size_x = height, width

        img = np.flip(img, 0)

        self.map = np.ones_like(img) * 255
        for i in range(height):
            for j in range(width):
                if (img[i][j] != [255, 255, 255]).any():
                    self.map[i][j] = [0, 0, 0]
                else:
                    self.occupancy.append([j, i])

    def get_map(self):
        return self.map
    
    def get_occupancy(self):
        return self.occupancy