from collections import defaultdict
import time
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
import supervision as sv
from spatial_location_calculator import SpatialLocationCalculator

class Camera:
    def __init__(self, image_width, image_height):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.image_width = image_width
        self.image_height = image_height
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        print("Depth Scale is: ", self.depth_scale)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
    
    def get_camera_coords(self):
        camera_coords = input("Enter coordinates for camera position and yaw in meters and radians in the format x,y,yaw: ")
        camera_coords = np.array(camera_coords.split(','), dtype=np.float32)
        return camera_coords

    def stop(self):
        self.pipeline.stop()

class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_objects(self, image):
        return self.model(image, verbose=False, classes=67)[0]

class Gimbal:
    def __init__(self, image_width, image_height, HFOV, VFOV):
        self.image_width = image_width
        self.image_height = image_height
        self.HFOV = HFOV
        self.VFOV = VFOV

    def calculate_gimbal_adjustment(self, bbox):
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        image_center_x = self.image_width / 2
        image_center_y = self.image_height / 2
        delta_x = bbox_center_x - image_center_x
        delta_y = -(bbox_center_y - image_center_y)
        yaw_adjustment = (delta_x / self.image_width) * self.HFOV
        pitch_adjustment = (delta_y / self.image_height) * self.VFOV
        return yaw_adjustment, pitch_adjustment

    def calculate_gimbal_offsets(self, direction):
        pitch_offset = -0.1
        if direction == "Moving Right":
            yaw_offset = 0.3
        elif direction == "Moving Left":
            yaw_offset = -0.3
        else:
            yaw_offset = 0
        return yaw_offset, pitch_offset

class SpatialCalculator:
    def __init__(self, image_width, image_height):
        self.spatial_location_calculator = SpatialLocationCalculator(image_width, image_height)
        self.HFOV = self.spatial_location_calculator.calc_HFOV()
        self.VFOV = self.spatial_location_calculator.calc_VFOV()
        print("Horizontal FOV: ", self.HFOV)
        print("Vertical FOV: ", self.VFOV)

        # Dictionary to store previous positions of tracked objects
        self.previous_positions = defaultdict(lambda: None)

    def calculate_object_location(self, camera_coords, relative_object_coords, object_euclidean_distance):
        cam_x, cam_y, cam_yaw = camera_coords
        obj_pos_x, obj_depth = relative_object_coords
        theta = cam_yaw + math.atan2(obj_pos_x, obj_depth)
        theta = theta % (2 * math.pi)
        vector_from_cam_to_object = np.array([
            object_euclidean_distance * math.cos(theta),
            object_euclidean_distance * math.sin(theta)
        ])
        object_coords = vector_from_cam_to_object + camera_coords[:2]
        return object_coords
    
    def determine_direction_of_movement(self, tracker_id, current_position):
        previous_position = self.previous_positions[tracker_id]
        if previous_position is None:
            self.previous_positions[tracker_id] = current_position
            return "Stationary"

        movement = current_position[0] - previous_position[0]
        self.previous_positions[tracker_id] = current_position

        movement_buffer = 2
        if movement > movement_buffer:
            return "Moving Right"
        elif movement < -movement_buffer:
            return "Moving Left"
        else:
            return "Stationary"
        
class Main:
    def __init__(self):
        self.image_width = 640
        self.image_height = 480
        self.camera = Camera(self.image_width, self.image_height)
        self.object_detector = ObjectDetector("yolov8n.pt")
        self.spatial_calculator = SpatialCalculator(self.image_width, self.image_height)
        self.gimbal = Gimbal(self.image_width, self.image_height, self.spatial_calculator.HFOV, self.spatial_calculator.VFOV)
        # self.camera_coords = self.camera.get_camera_coords()
        self.camera_coords = np.array([0, 0, 0], dtype=np.float32)
        self.conf_threshold = 0.5

        # Initialise Supervision components
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def run(self):
        num_frames_processed = 0
        start = time.time()
        elapsed_time_lst = []
        last_valid_depth_value = 0

        try:
            while True:
                depth_image, color_image = self.camera.get_frames()
                if depth_image is None or color_image is None:
                    continue

                num_frames_processed += 1
                results = self.object_detector.detect_objects(color_image)

                # Convert results to Supervision detections
                boxes = results.boxes
                detections = sv.Detections.from_ultralytics(results)
                detections = self.tracker.update_with_detections(detections)
                
                # Optional smoothing
                # if detections.tracker_id:
                #     detections = self.smoother.update_with_detections(detections)

                labels = [
                    f"#{tracker_id} {results.names[class_id]}"
                    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
                ]

                # Annotate the frame
                annotated_frame = self.box_annotator.annotate(color_image.copy(), detections=detections)
                
                if len(detections) > 0:
                    labels = [
                        f"#{tracker_id} {results.names[class_id]}"
                        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
                    ]
                    if labels:
                        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

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

                        # # Annotate object details on the frame
                        # cv2.putText(annotated_frame, str(object_str), (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        # cv2.putText(annotated_frame, confidence_str, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        # cv2.putText(annotated_frame, depth_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

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
            end = time.time()
            elapsed = end - start
            fps = num_frames_processed / elapsed
            print(f"FPS: {num_frames_processed} / {elapsed:.2f} = {fps:.2f}")
            print(f"Average elapsed time: {sum(elapsed_time_lst) / len(elapsed_time_lst):.2f}")

if __name__ == '__main__':
    main_app = Main()
    main_app.run()
