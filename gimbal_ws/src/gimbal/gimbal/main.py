from spatial_calculator import SpatialCalculator
from gimbal_handler import Gimbal
from detection_processor import DetectionProcessor
import rclpy
from rclpy.node import Node
import cv2

class Main(Node):
    def __init__(self):
        self.image_width = 640
        self.image_height = 480
        
    def run(self):
        spatial_calculator = SpatialCalculator(self.image_width, self.image_height)
        gimbal = Gimbal(self.image_width, self.image_height, spatial_calculator.HFOV, spatial_calculator.VFOV)

        # detection process objects for front and back cameras
        front_camera_detection_processor = DetectionProcessor(gimbal, spatial_calculator, 'front_camera')
        # back_camera_detection_processor = DetectionProcessor(gimbal, spatial_calculator, 'back_camera')
        
        # running rosnodes using multithreading
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(gimbal)
        executor.add_node(front_camera_detection_processor)
        # executor.add_node(back_camera_detection_processor)

        while rclpy.ok():
            rclpy.spin_once(front_camera_detection_processor, timeout_sec=0.02)
            # rclpy.spin_once(back_camera_detection_processor, timeout_sec=0.02)

            # visualise detection results
            if front_camera_detection_processor.get_annotated_frame() is not None:
                cv2.imshow("Front Camera", front_camera_detection_processor.get_annotated_frame())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # if back_camera_detection_processor.get_annotated_frame() is not None:
            #     cv2.imshow("Back Camera", back_camera_detection_processor.get_annotated_frame())
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

def main(args=None):
    rclpy.init(args=args)
    main_app = Main()
    main_app.run()
    main_app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
