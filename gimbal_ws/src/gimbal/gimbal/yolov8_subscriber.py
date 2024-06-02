from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ultralytics.engine.results import Results, Boxes
import numpy as np
import rclpy
import torch

class_names = {
    0: "blue-base",
    1: "blue-hero",
    2: "blue-sentry",
    3: "blue-standard-3",
    4: "blue-standard-4",
    5: "red-base",
    6: "red-hero",
    7: "red-sentry",
    8: "red-standard-3",
    9: "red-standard-4",
}

class Yolov8DetectionSubscriber(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        self.subscription = self.create_subscription(
            Detection2DArray,
            'detections_output',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.results = None
        
        self.image_width = 640
        self.image_height = 480

    def listener_callback(self, msg):
        self.results = self.convert_detections_format_for_supervision(msg)

    def convert_detections_format_for_supervision(self, msg):
        boxes = []
        for detection in msg.detections:
            if len(detection.results) == 0:
                continue  
            
            x_center = detection.bbox.center.position.x
            y_center = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            confidence = detection.results[0].hypothesis.score
            class_id = int(detection.results[0].hypothesis.class_id)
            boxes.append([x_min, y_min, x_max, y_max, confidence, class_id])

        if not boxes:
            self.get_logger().info('No valid detections to process')
            return None  

        boxes = np.array(boxes, dtype=np.float32)
        self.get_logger().info(f'Boxes: {boxes}')

        try:
            results = Results(
                orig_img=np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8),
                path='',
                names=class_names,
                boxes=boxes,
            )
        except Exception as e:
            self.get_logger().error(f'Error creating Results object: {e}')
            return None

        return results

    def get_results(self):
        return self.results

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8DetectionSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
