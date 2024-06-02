import time
import numpy as np
from spatial_calculator import SpatialCalculator
from gimbal_handler import Gimbal
from detection_processor import DetectionProcessor
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
import cv2

class Main(Node):
    def __init__(self):
        super().__init__('main_loop')
        self.image_width = 640
        self.image_height = 480
        self.camera_coords = np.array([0, 0, 0], dtype=np.float32)
        
        self.spatial_calculator = SpatialCalculator(self.image_width, self.image_height)
        self.gimbal = Gimbal(self.image_width, self.image_height, self.spatial_calculator.HFOV, self.spatial_calculator.VFOV)
        self.front_camera_detection_processor = DetectionProcessor(self.gimbal, self.spatial_calculator)

        # yolov8 subscriber
        self.yolov8_subscription = self.create_subscription(
            Detection2DArray,
            'detections_output',
            self.yolov8_detections_callback,
            10)
        self.yolov8_subscription  # prevent unused variable warning
        self.get_logger().info('Subscriber for YOLOv8 detections created.')
        self.supervision_results = None

    def yolov8_detections_callback(self, yolov8_detection_msg):
        self.get_logger().info('Received YOLOv8 detections')
        self.supervision_results = self.front_camera_detection_processor.convert_detections_format_for_supervision(yolov8_detection_msg)
        self.front_camera_detection_processor.update_supervision_results(self.supervision_results)
        self.get_logger().info(f'Received {len(self.supervision_results.cls)} detections')
     
    def run(self):
        num_frames_processed = 0
        start = time.time()
        elapsed_time_lst = []

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.gimbal)
        executor.add_node(self.front_camera_detection_processor)
        executor.add_node(self)

        self.get_logger().info('Starting main loop')
        try:
            while rclpy.ok():
                # depth_image, color_image = self.camera.get_frames()
                # if depth_image is None or color_image is None:
                #     continue

                num_frames_processed += 1
                rclpy.spin_once(self.front_camera_detection_processor, timeout_sec=0.05)

                # Check if there are results to display
                if self.supervision_results:
                    detections = self.front_camera_detection_processor.process_supervision(self.supervision_results)
                    annotated_frame = self.front_camera_detection_processor.annotate_frames(
                        detections,
                        self.supervision_results,
                        self.supervision_results.orig_img  # Assuming this is where the original image is stored
                    )
                    cv2.imshow('Object Detection', annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        finally:
            # self.camera.stop()
            cv2.destroyAllWindows()
            end = time.time()
            elapsed = end - start
            fps = num_frames_processed / elapsed
            print(f"FPS: {num_frames_processed} / {elapsed:.2f} = {fps:.2f}")
            print(f"Average elapsed time: {sum(elapsed_time_lst) / len(elapsed_time_lst):.2f}")

def main(args=None):
    rclpy.init(args=args)
    main_app = Main()
    main_app.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
