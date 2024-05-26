import numpy as np
import pyrealsense2 as rs

class Camera:
    def __init__(self, image_width, image_height):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.image_width = image_width
        self.image_height = image_height
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        print("Depth Scale is: ", self.depth_scale)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
    
    def get_camera_coords(self):
        camera_coords = input("Enter coordinates for camera position and yaw in meters and radians in the format x,y,yaw: ")
        camera_coords = np.array(camera_coords.split(','), dtype=np.float32)
        return camera_coords

    def stop(self):
        self.pipeline.stop()