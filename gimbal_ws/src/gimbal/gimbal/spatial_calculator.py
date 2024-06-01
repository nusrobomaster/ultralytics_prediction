from collections import defaultdict
import math
import numpy as np
from spatial_location_calculator import SpatialLocationCalculator

class SpatialCalculator:
    def __init__(self, image_width, image_height):
        self.spatial_location_calculator = SpatialLocationCalculator(image_width, image_height)
        self.HFOV = 1.23918 # 71.0 degrees
        self.VFOV = 0.767945 # 44.0 degrees
        print("Horizontal FOV: ", self.HFOV)
        print("Vertical FOV: ", self.VFOV)

        # Dictionary to store previous positions of tracked objects
        self.previous_positions = defaultdict(lambda: None)

    def calculate_object_location(self, camera_coords, relative_object_coords, object_euclidean_distance):
        cam_x, cam_y, cam_yaw = camera_coords
        obj_pos_x, obj_depth = relative_object_coords
        theta = cam_yaw + math.atan2(obj_pos_x, obj_depth)
        theta = theta % (2 * math.pi)
        vector_from_cam_to_object = np.array([
            object_euclidean_distance * math.cos(theta),
            object_euclidean_distance * math.sin(theta)
        ])
        object_coords = vector_from_cam_to_object + camera_coords[:2]
        return object_coords
    
    def determine_direction_of_movement(self, tracker_id, current_position):
        previous_position = self.previous_positions[tracker_id]
        if previous_position is None:
            self.previous_positions[tracker_id] = current_position
            return "Stationary"

        movement = current_position[0] - previous_position[0]
        self.previous_positions[tracker_id] = current_position

        movement_buffer = 2
        if movement > movement_buffer:
            return "Moving Right"
        elif movement < -movement_buffer:
            return "Moving Left"
        else:
            return "Stationary"
        