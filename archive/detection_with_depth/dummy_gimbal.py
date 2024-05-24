import time

class DummyGimbal:
    def __init__(self, target_buffer=0.02):
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.moving = False
        self.has_target = False
        self.target_buffer = target_buffer

    def move_to(self, yaw, pitch):
        self.target_yaw = yaw
        self.target_pitch = pitch
        self.has_target = True
        self.moving = True

    def update(self):
        time.sleep(0.05)
        angular_increment = 0.01
        if self.moving:
            # Incrementally move towards the target orientation
            if self.current_yaw < self.target_yaw:
                self.current_yaw += angular_increment
                if self.current_yaw > self.target_yaw:
                    self.current_yaw = self.target_yaw
            elif self.current_yaw > self.target_yaw:
                self.current_yaw -= angular_increment
                if self.current_yaw < self.target_yaw:
                    self.current_yaw = self.target_yaw

            if self.current_pitch < self.target_pitch:
                self.current_pitch += angular_increment
                if self.current_pitch > self.target_pitch:
                    self.current_pitch = self.target_pitch
            elif self.current_pitch > self.target_pitch:
                self.current_pitch -= angular_increment
                if self.current_pitch < self.target_pitch:
                    self.current_pitch = self.target_pitch

            # Check if the gimbal has reached the target orientation within the buffer range
            yaw_reached = abs(self.current_yaw - self.target_yaw) <= self.target_buffer
            pitch_reached = abs(self.current_pitch - self.target_pitch) <= self.target_buffer

            if yaw_reached and pitch_reached:
                self.moving = False

    def get_orientation(self):
        return self.current_yaw, self.current_pitch
