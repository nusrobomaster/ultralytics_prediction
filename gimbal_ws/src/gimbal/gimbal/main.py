import numpy as np
from spatial_calculator import SpatialCalculator
from gimbal_handler import Gimbal
from detection_processor import DetectionProcessor
import rclpy
from rclpy.node import Node

class Main(Node):
    def __init__(self):
        self.image_width = 640
        self.image_height = 480
        
    def run(self):
        spatial_calculator = SpatialCalculator(self.image_width, self.image_height)
        gimbal = Gimbal(self.image_width, self.image_height, spatial_calculator.HFOV, spatial_calculator.VFOV)
        front_camera_detection_processor = DetectionProcessor(gimbal, spatial_calculator)
        
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(gimbal)
        executor.add_node(front_camera_detection_processor)

        while rclpy.ok():
            rclpy.spin_once(front_camera_detection_processor, timeout_sec=0.05)


def main(args=None):
    rclpy.init(args=args)
    main_app = Main()
    main_app.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
