from rclpy.node import Node
import supervision as sv
import numpy as np
from std_msgs.msg import Float32MultiArray
import cv2
from yolov8_subscriber import Yolov8DetectionSubscriber
import rclpy

# Perform object tracking using supervision library
# Determine the euclidean distance of the detected objects from the camera
# Determine the yaw and pitch adjustments for the gimbal
# Publishes this as the gimbal orientation to rotate relative to the camera, together with the euclidean distance
class DetectionProcessor(Node):
    def __init__(self, gimbal, spatial_calculator, namespace):
        super().__init__('detection_processor', namespace=namespace)
        self.target_information_publisher = self.create_publisher(Float32MultiArray, 'target_information', 10)
        self.timer = self.create_timer(0.02, self.publish_target_information)
        self.get_logger().info('Gimbal Publisher Node has been started.')
        
        # Supervision components
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

        self.gimbal = gimbal
        self.spatial_calculator = spatial_calculator

        self.yolov8_detector = Yolov8DetectionSubscriber(namespace)
        self.yolov8_results, self.color_image, self.depth_map = None, None, None
        self.target_information = Float32MultiArray()
        self.annotated_frame = None
        self.supervision_detections = None

        self.conf_threshold = 0.5
        self.camera_coords = np.array([0, 0, 0], dtype=np.float32)

    def publish_target_information(self):
        rclpy.spin_once(self.yolov8_detector, timeout_sec=0.02)
        
        self.yolov8_results = self.yolov8_detector.get_detection_results()
        self.color_image = self.yolov8_detector.get_color_image()
        self.depth_map = self.yolov8_detector.get_depth_map()

        self.annotated_frame = None # reset the annotated frame for visualisation purposes
        if self.yolov8_results and self.color_image is not None and self.depth_map is not None:
            self.supervision_detections = self.process_supervision()
            self.annotated_frame = self.annotate_frames(self.supervision_detections)
            # Target information is [euclidean_distance, yaw_adjustment + yaw_offset, pitch_adjustment + pitch_offset]
            data = self.get_target_information_data(self.supervision_detections)
            if all(value is not None for value in data):
                self.target_information.data = data
            
            self.target_information_publisher.publish(self.target_information)
        else:
            if self.color_image is None:
                print('Color image is None')
            if self.depth_map is None:
                print('Depth map is None')
            if self.yolov8_results is None:
                print('Yolov8 results is None')
    
    def process_supervision(self):
        supervision_detections = sv.Detections.from_ultralytics(self.yolov8_results)
        supervision_detections = self.tracker.update_with_detections(supervision_detections)
        return supervision_detections

    def annotate_frames(self, detections):
        annotated_frame = self.box_annotator.annotate(self.color_image.copy(), detections=detections)
        if len(detections) > 0:
            labels = [
                f"#{tracker_id} {self.yolov8_results.names[class_id]}"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
            ]
            if labels:
                annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame
    
    def get_target_information_data(self, detections):
        detection_information, detection_distance_info = [], []
        last_valid_depth_value = 0
        for i in range(len(detections.xyxy)):
            confidence = detections.confidence[i]
            if confidence > self.conf_threshold:
                bbox = detections.xyxy[i]
                class_name = detections.data['class_name'][i]

                xmin, ymin, xmax, ymax = map(int, bbox)
                current_position = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                yaw_offset, pitch_offset = 0, 0
                
                if len(detections.tracker_id) > 0:
                    tracker_id = detections.tracker_id[i]
                    direction = self.spatial_calculator.determine_direction_of_movement(tracker_id, current_position)
                    yaw_offset, pitch_offset = self.gimbal.calculate_gimbal_offsets(direction)
                    cv2.putText(self.annotated_frame, direction, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                centroid_x, centroid_y, depth_value = self.spatial_calculator.spatial_location_calculator.calc_location_relative_to_camera((xmin, ymin, xmax, ymax), self.depth_map)
                distance_from_camera = self.spatial_calculator.spatial_location_calculator.calc_distance_from_camera(centroid_x, centroid_y, last_valid_depth_value)
                pos_x, pos_y = self.spatial_calculator.calculate_object_location(self.camera_coords, (centroid_x, last_valid_depth_value), distance_from_camera)
                yaw_adjustment, pitch_adjustment = self.gimbal.calculate_gimbal_adjustment((xmin, ymin, xmax, ymax))

                is_valid_depth_value = not np.isnan(depth_value) and not np.isinf(depth_value)
                if is_valid_depth_value:
                    last_valid_depth_value = depth_value
                    detection_distance_info.append(distance_from_camera)
                    detection_information.append(((centroid_x, centroid_y), (yaw_adjustment, pitch_adjustment), (yaw_offset, pitch_offset), (pos_x, pos_y)))

                object_str = f"objects: {class_name}"
                confidence_str = f"conf: {confidence:.2f}"
                depth_str = f"distance: {last_valid_depth_value:.2f} meters"

        # Print detection information 
        if detection_information:
            target_index = np.argmin(detection_distance_info)
            (target_centroid_x, target_centroid_y), (yaw_adjustment, pitch_adjustment), (yaw_offset, pitch_offset), (target_pos_x, target_pos_y) = detection_information[target_index]
            euclidean_dist = detection_distance_info[target_index]
            yaw_relative_to_gimbal = yaw_adjustment + yaw_offset
            pitch_relative_to_gimbal = pitch_adjustment + pitch_offset
            
            print(f"Target {object_str} is at x = {target_centroid_x:.2f}m, y = {target_centroid_y:.2f}m, z = {last_valid_depth_value:.2f}m")
            print(f"Euclidean distance away: {euclidean_dist:.2f}m")
            print(f"{object_str} has coordinates x = {target_pos_x:.2f}m, y = {target_pos_y:.2f}m relative to the map")
            print(f"Angle of target relative to camera: yaw = {yaw_adjustment:.2f} radians, pitch = {pitch_adjustment:.2f} radians")
            print(f"Gimbal aims at: yaw = {yaw_relative_to_gimbal:.2f} radians, pitch = {pitch_relative_to_gimbal:.2f} radians")

            # self.gimbal.publish_orientation(pitch_offset + pitch_adjustment, yaw_offset + yaw_adjustment)
            return [euclidean_dist, yaw_relative_to_gimbal, pitch_relative_to_gimbal]
        return [None, None, None]

    def get_annotated_frame(self):
        if self.annotated_frame is not None:
            return self.annotated_frame
        return self.color_image