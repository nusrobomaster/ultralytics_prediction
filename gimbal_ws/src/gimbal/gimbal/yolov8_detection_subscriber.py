import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ultralytics.engine.results import Results
import numpy as np

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

    def listener_callback(self, msg):
        self.results = self.convert_detections(msg)
        self.get_logger().info(f'Received {len(self.results.cls)} detections')
        # feed results into the supervision prediction .from_ultralytics(results)

    def convert_detections(self, msg):
        boxes = []
        confidences = []
        class_ids = []
        for detection in msg.detections:
            x_center = detection.bbox.center.position.x
            y_center = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            boxes.append([x_min, y_min, x_max, y_max])
            class_ids.append(int(detection.results[0].hypothesis.class_id))
            confidences.append(detection.results[0].hypothesis.score)

        boxes = np.array(boxes)
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)

        results = Results(
            orig_img=np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8),
            path='',
            names={i: f'class_{i}' for i in range(len(class_ids))},
            boxes=boxes,
            conf=confidences,
            cls=class_ids
        )
        
        return results