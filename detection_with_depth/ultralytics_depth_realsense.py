from ultralytics import YOLO
import time
import pyrealsense2 as rs
import numpy as np
import cv2
from spatial_location_calculator import SpatialLocationCalculator
import math

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

spatial_location_calculator = SpatialLocationCalculator(IMAGE_WIDTH, IMAGE_HEIGHT)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Start streaming
config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)

profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

conf_threshold = 0.5

# Eventually replace with a ROS2 subscriber node which retrieves camera coordinates
# x,y are measures from the lower right corner of the map. yaw the angle measured anticlockwise from the x-axis
def get_camera_coords():        
    camera_coords = input("Enter coordinates for camera position and yaw in meters and radians in the format x,y,yaw: ")
    camera_coords = np.array(camera_coords.split(','), dtype=np.float32)
    return camera_coords

# Given the coordinates of the camera relative to the map and the coordinates of the object relative to the camera,
# compute the coordinates of the object relative to the map
def calc_location_relative_to_map(camera_coords, relative_object_coords, object_euclidean_distance):
    cam_x, cam_y, cam_yaw = camera_coords
    obj_pos_x, obj_depth = relative_object_coords

    theta = 0
    delta_angle_x = math.atan2(obj_pos_x, obj_depth)
    theta = cam_yaw + delta_angle_x
    theta = theta % (2 * math.pi) 

    vector_from_cam_to_object = np.array([
            object_euclidean_distance * math.cos(theta), 
            object_euclidean_distance * math.sin(theta)
    ])

    object_coords = vector_from_cam_to_object + camera_coords[:2]
    return object_coords   # x,y coordinates of object relative to map

camera_coords = get_camera_coords()

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    # model = YOLO("RM_130524_11pm.pt")

    num_frames_processed = 0
    last_time = start = time.time()

    elapsed_time_lst = []

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            num_frames_processed += 1

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLOv8 inference
            results = model(color_image, verbose=False, classes=67)

            # Get bounding box coordinates and depth
            for det in results[0].boxes:
                # print(det)
                confidence = det.conf
                if confidence > conf_threshold:
                    # det.xyxy has the format tensor([xmin, ymin, xmax, ymax])
                    xmin, ymin, xmax, ymax = map(int, det.xyxy[0].tolist())
                    
                    # Find centroid coordinates and depth information
                    centroid_x, centroid_y, depth_value = spatial_location_calculator.calc_location_relative_to_camera((xmin, ymin, xmax, ymax), depth_image)
                    distance_from_camera = spatial_location_calculator.calc_distance_from_camera(centroid_x, centroid_y, depth_value)
                    pos_x, pos_y = calc_location_relative_to_map(camera_coords, (centroid_x, depth_value), distance_from_camera)
                    
                    # Draw bounding box
                    object_str = "cls: {}".format(det.cls[0])
                    confidence_str = "conf: {:.2f}".format(confidence.tolist()[0])
                    depth_str = "distance: {:.2f} meters".format(depth_value)
                    cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(color_image, str(object_str), (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.putText(color_image, confidence_str, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.putText(color_image, depth_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    # Print coordinates and depth
                    print("Object {} detected at x = {:.2f}m, y = {:.2f}m, z = {:.2f}m".format(object_str, centroid_x, centroid_y, depth_value))
                    print("Euclidean distance away: {:.2f}m".format(distance_from_camera))
                    print("{} has coordinates x = {:.2f}m, y = {:.2f}m relative to the map".format(object_str, pos_x, pos_y))

            elapsed_time_lst.append(time.time() - last_time)
            # print(elapsed_time_lst[-1])
            last_time = time.time()
            
            # Display the resulting frame
            cv2.imshow('Object Detection', color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        pipeline.stop()
        end = time.time()

        elapsed = end - start
        fps = num_frames_processed / elapsed

        print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))
        print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))
    