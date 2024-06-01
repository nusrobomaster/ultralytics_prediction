from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ultralytics.engine.results import Results
import supervision as sv
import numpy as np
from std_msgs.msg import Float32MultiArray
import cv2

# Subscribes to the yolov8 detection output
# converts the detections to Supervision and perform object tracking
# Determine the euclidean distance of the detected objects from the camera
# Determine the yaw and pitch adjustments for the gimbal
# Publishes this as the gimbal orientation to rotate relative to the camera, together with the euclidean distance
class DetectionProcessor(Node):
    def __init__(self, gimbal, spatial_calculator):
        super().__init__('detection_processor')
        self.yolov8_subscription = self.create_subscription(
            Detection2DArray,
            'detections_output',
            self.listener_callback,
            10)
        self.yolov8_subscription  # prevent unused variable warning
        self.results = None
        
        self.target_information_publisher = self.create_publisher(Float32MultiArray, 'target_information', 10)
        self.timer = self.create_timer(0.05, self.publish_target_information)
        self.get_logger().info('Gimbal Publisher Node has been started.')
        
        # Supervision components
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

        self.gimbal = gimbal
        self.spatial_calculator = spatial_calculator

    def listener_callback(self, msg):
        self.results = self.convert_detections_format_for_supervision(msg)
        self.get_logger().info(f'Received {len(self.results.cls)} detections')
        
    def publish_target_information(self):
        if self.results:
            detections = self.process_supervision(self.results)
            annotated_frame = self.annotate_frames(detections, self.results, self.results.orig_img)
            # Target information is [euclidean_distance, yaw_adjustment + yaw_offset, pitch_adjustment + pitch_offset]
            target_information = self.get_target_information(detections, annotated_frame)
            self.target_information_publisher.publish(target_information)
    
    def process_supervision(self, results):
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        return detections

    def annotate_frames(self, detections, results, color_image):
        # Annotate the frame
        annotated_frame = self.box_annotator.annotate(color_image.copy(), detections=detections)
        if len(detections) > 0:
            labels = [
                f"#{tracker_id} {results.names[class_id]}"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
            ]
            if labels:
                annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame
    
    def get_target_information(self, detections, annotated_frame):
        detection_information, detection_distance_info = [], []
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
                    cv2.putText(annotated_frame, direction, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                centroid_x, centroid_y, depth_value = self.spatial_calculator.spatial_location_calculator.calc_location_relative_to_camera((xmin, ymin, xmax, ymax), depth_image)
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

        # cv2.imshow('Object Detection', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    def convert_detections_format_for_supervision(self, msg):
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