from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ultralytics.engine.results import Results 
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
        self.bridge = CvBridge()
        
        self.detections_subscription = self.create_subscription(
            Detection2DArray,
            'detections_output',
            self.detections_subscription_callback,
            10)
        self.detections_subscription  # prevent unused variable warning

        self.image_subscription = self.create_subscription(
            Image,
            'rgb_footage',
            self.image_subscription_callback,
            10)
        self.image_subscription  # prevent unused variable warning

        self.depth_subscription = self.create_subscription(
            Image,
            'depth_map',
            self.depth_subscription_callback,
            10)
        self.depth_subscription  # prevent unused variable warning

        self.yolov8_results = None
        
        self.image_width = 640
        self.image_height = 480
        
        self.color_image, self.depth_map = None, None

    def detections_subscription_callback(self, msg):
        self.yolov8_results = self.convert_detections_format_for_supervision(msg)

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
            # self.get_logger().info('No valid detections to process')
            return None  

        boxes = np.array(boxes, dtype=np.float32)
        boxes_tensor = torch.from_numpy(boxes).to(torch.device('cuda'))
        # self.get_logger().info(f'Boxes: {boxes_tensor}')

        try:
            results = Results(
                orig_img=np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8),
                path='',
                names=class_names,
                boxes=boxes_tensor,
            )
            # print(results.boxes.cls)
        except Exception as e:
            self.get_logger().error(f'Error creating Results object: {e}')
            return None

        return results

    def image_subscription_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # print(self.color_image)
        
    def depth_subscription_callback(self, msg):
        self.depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        # print(self.depth_image)

    def get_detection_results(self):
        return self.yolov8_results

    def get_color_image(self):
        return self.color_image
            
    def get_depth_map(self):
        return self.depth_map

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8DetectionSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
