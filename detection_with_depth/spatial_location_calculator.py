import numpy as np
import math

class SpatialLocationCalculator():
    def __init__(self, HFOV, horizontal_length, vertical_length):
        self.image_width = horizontal_length
        self.image_height = vertical_length

        # Setting of values
        self.DELTA = 6 # defines padding around the center point of the (ROI) 
        self.THRESH_LOW = 100 # 10cm
        self.THRESH_HIGH = 15000 # 15m

        # Compute distance (in pixels) from the projection center to the image center
        self.focal_length = self.image_width / (2.0 * math.tan(HFOV / 2.0))

    def calc_location(self, roi, depthMap):
        # Take 10x10 depth pixels around center of bounding box for depth averaging
        cx, cy = (roi[0]+roi[2])//2, (roi[1]+roi[3])//2
        # ensures that the bounding box for averaging, defined by xmin, ymin, xmax, ymax, 
        # does not exceed the image boundaries or is too close to the edges
        x = int(min(max(cx, self.DELTA), self.image_width - self.DELTA))
        y = int(min(max(cy, self.DELTA), self.image_height - self.DELTA))
        xmin, ymin, xmax, ymax = x-self.DELTA, y-self.DELTA, x+self.DELTA, y+self.DELTA

        # Calculate the average depth in the region
        depthROI = depthMap[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)
        averageDepth = np.mean(depthROI[inRange])

        u, v, z = cx - self.image_width // 2, cy - self.image_height // 2, self.focal_length # Spatial coordinates on image in pixels
        f = lambda x : x * averageDepth / self.focal_length

        return f(u), f(v), f(z)
    
    """Project a point in the spatial coordinate onto the retinal plane"""
    def retinal_projection(self, X, Y, Z):
        f_inverse = lambda x : x * self.focal_length / Z
        rX, rY = f_inverse(X), f_inverse(Y)
        if not (rX == rX and rY == rY):
            return 0, 0
        return int(rX + self.image_width // 2), int(rY + self.image_height // 2)


    def calc_distance(self, roi, depthMap):
        X, Y, Z = self.calc_location(roi, depthMap)
        return np.sqrt(X**2 + Y**2 + Z**2)