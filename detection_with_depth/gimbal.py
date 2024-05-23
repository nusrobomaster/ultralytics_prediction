class Gimbal:
    def __init__(self, IMAGE_WIDTH, IMAGE_HEIGHT, HFOV, VFOV):
        # Initial orientation (home position)
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.image_width = IMAGE_WIDTH
        self.image_height = IMAGE_HEIGHT
        self.hfov = HFOV
        self.vfov = VFOV

    def calculate_expected_gimbal_orientation(self, bbox):
        # Calculate the center of the bounding box
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2

        # Calculate the center of the image frame
        image_center_x = self.image_width / 2
        image_center_y = self.image_height / 2

        # Calculate the difference between the centers
        delta_x = bbox_center_x - image_center_x
        delta_y = bbox_center_y - image_center_y

        # Calculate the corresponding yaw and pitch adjustments
        yaw_adjustment = (delta_x / self.image_width) * self.hfov
        pitch_adjustment = (delta_y / self.image_height) * self.vfov

        # Calculate the new orientation
        new_yaw = self.current_yaw + yaw_adjustment
        new_pitch = self.current_pitch + pitch_adjustment

        return new_yaw, new_pitch

    def update_orientation(self, target_yaw, target_pitch):
        self.current_yaw = target_yaw
        self.current_pitch = target_pitch

    def get_current_orientation(self):
        return self.current_yaw, self.current_pitch