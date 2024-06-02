
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from object_detector import ObjectDetector
from camera import Camera
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DummyYOLOv8DetectionPublisher(Node):
    def __init__(self):
        super().__init__('yolov8_detection_publisher')
        self.bridge = CvBridge()

        self.detection_publisher = self.create_publisher(Detection2DArray, 'detections_output', 10)
        self.image_publisher = self.create_publisher(Image, 'rgb_footage', 10)        
        self.depth_publisher = self.create_publisher(Image, 'depth_map', 10)        
        self.timer = self.create_timer(0.02, self.timer_callback)

        self.object_detector = ObjectDetector('yolov8n_010624_7.pt')
        self.camera = Camera(640, 480)

    def timer_callback(self):
        depth_image, color_image = self.camera.get_frames()
        if depth_image is None or color_image is None:
            return

        # Get YOLOv8 results
        results = self.object_detector.detect_objects(color_image)
        # print(results.boxes.cls)

        # Create and populate the Detection2DArray message
        detections_msg = Detection2DArray()
        detections_msg.header = self.create_header()

        for box in results.boxes:
            detection = Detection2D()
            center_x, center_y, width, height = box.xywh[0] 

            detection.bbox.center.position.x = float(center_x)
            detection.bbox.center.position.y = float(center_y)
            detection.bbox.size_x = float(width)
            detection.bbox.size_y = float(height)
            
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(box.cls[0].item())) 
            hypothesis.hypothesis.score = float(box.conf[0].item())
            detection.results.append(hypothesis)

            detections_msg.detections.append(detection)

        self.detection_publisher.publish(detections_msg)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8"))
        self.depth_publisher.publish(self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1"))

        self.get_logger().info('Identified %d armour plates' % len(detections_msg.detections))

    def create_header(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'
        return header
    
def main(args=None):
    rclpy.init(args=args)
    node = DummyYOLOv8DetectionPublisher()
    rclpy.spin(node)
    node.camera.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
