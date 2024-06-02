
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from object_detector import ObjectDetector
from camera import Camera


class DummyYOLOv8DetectionPublisher(Node):
    def __init__(self):
        super().__init__('yolov8_detection_publisher')
        self.publisher_ = self.create_publisher(Detection2DArray, 'detections_output', 10)
        self.object_detector = ObjectDetector('yolov8n.pt')
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.camera = Camera(640, 480)

    def timer_callback(self):
        color_image = self.get_image()

        # Get YOLOv8 results
        results = self.object_detector.detect_objects(color_image)

        # Create and populate the Detection2DArray message
        detections_msg = Detection2DArray()
        detections_msg.header = self.create_header()

        for box in results.boxes:
            detection = Detection2D()
            center_x, center_y, width, height = box.xywh[0] # Access the bounding box coordinates

            detection.bbox.center.position.x = float(center_x)
            detection.bbox.center.position.y = float(center_y)
            detection.bbox.size_x = float(width)
            detection.bbox.size_y = float(height)
            
            hypothesis = ObjectHypothesisWithPose()
            # print(hypothesis)
            hypothesis.hypothesis.class_id = str(int(box.cls[0].item()))  # Class ID
            hypothesis.hypothesis.score = float(box.conf[0].item())  # Confidence score
            detection.results.append(hypothesis)

            detections_msg.detections.append(detection)

        self.publisher_.publish(detections_msg)
        self.get_logger().info('Identified %d armour plates' % len(detections_msg.detections))

    def create_header(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'
        return header

    def get_image(self):
        depth_image, color_image = self.camera.get_frames()
        return color_image

def main(args=None):
    rclpy.init(args=args)
    node = DummyYOLOv8DetectionPublisher()
    rclpy.spin(node)
    node.camera.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
