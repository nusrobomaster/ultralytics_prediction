import time
import numpy as np
import cv2
import supervision as sv
from spatial_calculator import SpatialCalculator
from camera import Camera
from gimbal import Gimbal, GimbalOrientationSubscriber
from object_detector import ObjectDetector
import rclpy

class Main:
    def __init__(self):
        self.image_width = 640
        self.image_height = 480
        # self.camera_coords = self.camera.get_camera_coords()
        self.camera_coords = np.array([0, 0, 0], dtype=np.float32)
        self.conf_threshold = 0.5
        
        # Objects
        self.camera = Camera(self.image_width, self.image_height)
        self.object_detector = ObjectDetector("yolov8n.pt")
        self.spatial_calculator = SpatialCalculator(self.image_width, self.image_height)
        self.gimbal = Gimbal(self.image_width, self.image_height, self.spatial_calculator.HFOV, self.spatial_calculator.VFOV)

        # Supervision components
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def run(self):
        num_frames_processed = 0
        start = time.time()
        elapsed_time_lst = []
        last_valid_depth_value = 0

        rclpy.init(args=None)
        gimbal_subscriber = GimbalOrientationSubscriber(self.gimbal)
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(gimbal_subscriber)

        try:
            while rclpy.ok():
                rclpy.spin_once(gimbal_subscriber, timeout_sec=0.1)
                depth_image, color_image = self.camera.get_frames()
                if depth_image is None or color_image is None:
                    continue

                num_frames_processed += 1
                results = self.object_detector.detect_objects(color_image)

                # Convert results to Supervision detections
                detections = sv.Detections.from_ultralytics(results)
                detections = self.tracker.update_with_detections(detections)
                
                # Optional smoothing
                # if detections.tracker_id:
                #     detections = self.smoother.update_with_detections(detections)

                # Annotate the frame
                annotated_frame = self.box_annotator.annotate(color_image.copy(), detections=detections)
                if len(detections) > 0:
                    labels = [
                        f"#{tracker_id} {results.names[class_id]}"
                        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
                    ]
                    if labels:
                        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

                # Process each detection
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
                    print(f"Target {object_str} is at x = {target_centroid_x:.2f}m, y = {target_centroid_y:.2f}m, z = {last_valid_depth_value:.2f}m")
                    print(f"Euclidean distance away: {detection_distance_info[target_index]:.2f}m")
                    print(f"{object_str} has coordinates x = {target_pos_x:.2f}m, y = {target_pos_y:.2f}m relative to the map")
                    print(f"Angle of target relative to camera: yaw = {yaw_adjustment:.2f} radians, pitch = {pitch_adjustment:.2f} radians")
                    print(f"Gimbal aims at: yaw = {yaw_adjustment + yaw_offset:.2f} radians, pitch = {pitch_adjustment + pitch_offset:.2f} radians")

                cv2.imshow('Object Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.camera.stop()
            gimbal_subscriber.destroy_node()
            rclpy.shutdown()
            end = time.time()
            elapsed = end - start
            fps = num_frames_processed / elapsed
            print(f"FPS: {num_frames_processed} / {elapsed:.2f} = {fps:.2f}")
            print(f"Average elapsed time: {sum(elapsed_time_lst) / len(elapsed_time_lst):.2f}")


def main(args=None):
    main_app = Main()
    main_app.run()

if __name__ == '__main__':
    main()

