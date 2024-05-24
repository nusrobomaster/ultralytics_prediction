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

    def stop(self):
        self.pipeline.stop()

class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_objects(self, image):
        # results = self.model(image, verbose=False, classes=67)
        # return results[0].boxes
        return self.model(image, verbose=False, classes=67)[0]

class SpatialCalculator:
    def __init__(self, image_width, image_height):
        self.spatial_location_calculator = SpatialLocationCalculator(image_width, image_height)
        self.HFOV = self.spatial_location_calculator.calc_HFOV()
        self.VFOV = self.spatial_location_calculator.calc_VFOV()
        print("Horizontal FOV: ", self.HFOV)
        print("Vertical FOV: ", self.VFOV)

    def calculate_gimbal_adjustment(self, bbox):
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        image_center_x = self.spatial_location_calculator.image_width / 2
        image_center_y = self.spatial_location_calculator.image_height / 2
        delta_x = bbox_center_x - image_center_x
        delta_y = bbox_center_y - image_center_y
        yaw_adjustment = (delta_x / self.spatial_location_calculator.image_width) * self.HFOV
        pitch_adjustment = (delta_y / self.spatial_location_calculator.image_height) * self.VFOV
        return yaw_adjustment, pitch_adjustment

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

class Main:
    def __init__(self):
        self.image_width = 640
        self.image_height = 480
        self.camera = Camera(self.image_width, self.image_height)
        self.object_detector = ObjectDetector("yolov8n.pt")
        self.spatial_calculator = SpatialCalculator(self.image_width, self.image_height)
        # self.camera_coords = self.get_camera_coords()
        self.camera_coords = np.array([0, 0, 0], dtype=np.float32)
        self.conf_threshold = 0.5

        # Initialize Supervision components
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        # self.smoother = sv.DetectionsSmoother()

    # Replace with ros2 subscriber for camera coordinates
    def get_camera_coords(self):
        camera_coords = input("Enter coordinates for camera position and yaw in meters and radians in the format x,y,yaw: ")
        camera_coords = np.array(camera_coords.split(','), dtype=np.float32)
        return camera_coords

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
                
                # optional smoothing
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
                for det in boxes:
                    confidence = det.conf
                    if confidence > self.conf_threshold:
                        xmin, ymin, xmax, ymax = map(int, det.xyxy[0].tolist())
                        centroid_x, centroid_y, depth_value = self.spatial_calculator.spatial_location_calculator.calc_location_relative_to_camera((xmin, ymin, xmax, ymax), depth_image)
                        distance_from_camera = self.spatial_calculator.spatial_location_calculator.calc_distance_from_camera(centroid_x, centroid_y, last_valid_depth_value)
                        pos_x, pos_y = self.spatial_calculator.calculate_object_location(self.camera_coords, (centroid_x, last_valid_depth_value), distance_from_camera)
                        yaw_adjustment, pitch_adjustment = self.spatial_calculator.calculate_gimbal_adjustment((xmin, ymin, xmax, ymax))

                        is_valid_depth_value = not np.isnan(depth_value) and not np.isinf(depth_value)
                        if is_valid_depth_value:
                            last_valid_depth_value = depth_value
                            detection_distance_info.append(distance_from_camera)
                            detection_information.append(((centroid_x, centroid_y), (yaw_adjustment, pitch_adjustment), (pos_x, pos_y)))

                        object_str = "cls: {}".format(det.cls[0])
                        confidence_str = "conf: {:.2f}".format(confidence.tolist()[0])
                        depth_str = "distance: {:.2f} meters".format(last_valid_depth_value)
                        # cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        # cv2.putText(color_image, str(object_str), (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        # cv2.putText(color_image, confidence_str, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        # cv2.putText(color_image, depth_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                if detection_information:
                    target_index = np.argmin(detection_distance_info)
                    (target_centroid_x, target_centroid_y), (yaw_adjustment, pitch_adjustment), (target_pos_x, target_pos_y) = detection_information[target_index]
                    print("Target {} is at x = {:.2f}m, y = {:.2f}m, z = {:.2f}m".format(object_str, target_centroid_x, target_centroid_y, last_valid_depth_value))
                    print("Euclidean distance away: {:.2f}m".format(detection_distance_info[target_index]))
                    print("{} has coordinates x = {:.2f}m, y = {:.2f}m relative to the map".format(object_str, target_pos_x, target_pos_y))
                    print("Gimbal adjustments: yaw = {:.2f} radians, pitch = {:.2f} radians".format(yaw_adjustment, pitch_adjustment))

                cv2.imshow('Object Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.camera.stop()
            end = time.time()
            elapsed = end - start
            fps = num_frames_processed / elapsed
            print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))
            print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))

if __name__ == '__main__':
    main_app = Main()
    main_app.run() 
